"""window.py — dataset de ventanas de anestesia (cohorte v5).

Genera, a partir de los casos VitalDB (reales) y sintéticos v5, el dataset de
ventanas ``data/windows_v2/`` (rejilla de 5 s, forward-fill limitado por track,
imagen de 14 variables + FIO2 + 5 columnas de bolo, split por paciente, pares
contrafactuales mantenidos unidos por metadata).

Uso:
    python src/autoencoder/window.py test      # prueba de la fase A (no genera)
    python src/autoencoder/window.py generate  # fase B (solo si el gate pasa)

CHANGELOG (correcciones W1-W9 sobre la versión v3 perdida)
----------------------------------------------------------
* W1  Rutas y descubrimiento de casos adaptado a las cohortes v5:
       ``source`` en {real, synthetic_v5, vaso_reinf_v5, cf_v5}.
       real     -> data/real/cases/*.parquet + clinical_data_enriched.parquet
       synthetic_v5 -> cases/*.parquet + clinical_data.parquet
       vaso_reinf_v5 -> cases/*.parquet + clinical/*_clinical.parquet
       cf_v5   -> cases/*.parquet + clinical/*_clinical.parquet +
                  metadata/cf_pair_*.json
* W2  ``list_cases`` acepta ``exclusions_path`` y elimina los caseids antes del
       split, con log por fuente. Los pares CF se excluyen completos (el CSV ya
       los lista por pares; se verifica y, por defensa, se expande al par).
* W3  ``make_split`` construye el ``subjectid`` de los pares CF desde
       ``metadata/cf_pair_*.json`` (caseid_a, caseid_b), no por paridad. En v5 la
       paridad impar/par es uniforme (caseid_a impar, caseid_b = a+1) y se usa
       solo como comprobación; se reporta cualquier contradicción.
* W4  Rejilla acotada: [max(anestart, casestart, time.min),
       min(aneend, caseend, time.max)). Se recortan las ventanas de cabeza/cola
       en las que TODOS los tracks de la imagen tienen máscara 0; se reportan
       ``n_trimmed_head``/``n_trimmed_tail`` por caso.
* W5  ``bp_source`` con cobertura mínima: ART si cobertura ART_MBP >= 20% de la
       rejilla (forward-fill + max_age); si no NIBP_MBP si >= 20%; si no "none".
       Cuando es "nibp" las columnas ART_* se rellenan con el proxy NIBP_*.
* W6  Máscara de plausibilidad: cada track de la imagen (y FIO2) tiene un rango
       plausible ``PLAUSIBLE_RANGE``. Un valor fuera de rango se conserva en la
       columna de valor y su máscara ``m_`` pasa a 0. ``m_raw_<name>`` (solo las
       14 de la imagen) conserva la presencia sin plausibilidad.
* W7  Fila clínica: deduplicación de clinical_data_enriched (real). Se comprueba
       si los duplicados son idénticos; si difieren se reportan columnas y se
       elige la fila con más columnas no nulas. Un caso sin fila clínica se
       excluye con log (no se procesa con valores por defecto).
* W8  Registro de tracks: se eliminan los tracks de evento (bolos); la maquinaria
       de suma en intervalo permanece sin entradas. Unidades según contrato v5
       (FIO2 en %, RFTN20_RATE en mL/h). Clasificación de los tracks en familias
       (vent_setpoint, drug_pk, drug_volume, drug_rare, duplicate, advanced_monitor,
       monitor, vent_measured, other). Tracks con presencia 0% en la muestra de
       prueba se marcan ``present_in_sample=False``.
* W9  Tiempo: ``t_since_opstart`` (segundos desde opstart, NaN si falta) y
       ``phase_from_clinical`` derivada de marcas (induction/maintenance/emergence;
       pre_induction no aplica dentro de la rejilla). No sustituye a ``phase`` del
       truth.
* B1  Reincorporación de bolos: las 5 columnas de bolo vuelven al registro como
       ``kind="event"``, ``group="drug_bolus"``, con nombre corto (``ppf_bolus``,
       ``remi_bolus``, ``roc_bolus``, ``phen_bolus``, ``eph_bolus``). Valor en la
       ventana = suma en ``(t - grid_s, t]``; máscara = 1 si la columna existe en
       el caso (aunque la suma sea 0), 0 si no existe. En reales las cinco
       quedan NaN con máscara 0. Salida en ``data/windows_v2/``.

Nota: la versión previa (v3) y su informe se perdieron (src/autoencoder quedó
vacío); este módulo se reconstruye contra v5 siguiendo el contrato de la tarea y
los hechos verificados en la memoria del repositorio y en
src/anessim/scripts/_compute_real_reference.py (que define IMAGE_TRACKS, MAX_AGE
y GRID_S de window.py). La tabla PLAUSIBLE_RANGE se reconstruye (el audit
original también se perdió) con límites fisiológicos verificados contra el
criterio del gate (<= 5% de ventanas reales anuladas por plausibilidad).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Constantes del contrato de ventana (no cambian respecto a la versión v3)
# --------------------------------------------------------------------------
GRID_S = 5.0
SEED = 42
VAL_FRAC = 0.15
ART_MIN_COV = 0.20
NIBP_MIN_COV = 0.20
PART_SIZE = 200_000

OUT_DIR = ROOT / "data" / "windows_v2"
EXCLUSIONS_PATH = ROOT / "data" / "audit" / "exclusions_proposed_v5.csv"

# Alias de nombres de fuente (el CSV de exclusiones usa el nombre de directorio
# "synthetic_vaso_reinf_v5"; el valor canónico de source es "vaso_reinf_v5").
SOURCE_ALIASES: dict[str, str] = {"synthetic_vaso_reinf_v5": "vaso_reinf_v5"}


def _normalize_source(name: str) -> str:
    return SOURCE_ALIASES.get(name, name)

SOURCES: dict[str, Path] = {
    "real": ROOT / "data" / "real",
    "synthetic_v5": ROOT / "data" / "synthetic_v5",
    "vaso_reinf_v5": ROOT / "data" / "synthetic_vaso_reinf_v5",
    "cf_v5": ROOT / "data" / "cf_v5",
}

# Imagen de 14 variables (definición canónica, idéntica a la usada por la
# auditoría y por _compute_real_reference.py).
IMAGE_TRACKS: list[str] = [
    "BIS/BIS",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
    "Primus/ETCO2",
    "Primus/PEEP_MBAR",
    "Primus/PIP_MBAR",
    "Primus/MV",
    "Primus/TV",
    "Primus/RR_CO2",
    "Solar8000/BT",
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "BIS/EMG",
]

FIO2_TRACK = "Primus/FIO2"

# max_age (s) por track (semántica window.py: un punto de rejilla está "presente"
# si el último valor no nulo tiene como máximo esta edad). Tabla de
# _compute_real_reference.py; se extiende con NIBP y FIO2 (documentado).
MAX_AGE: dict[str, float] = {
    "BIS/BIS": 10.0,
    "BIS/EMG": 10.0,
    "BIS/SEF": 10.0,
    "BIS/SQI": 10.0,
    "BIS/SR": 10.0,
    "BIS/TOTPOW": 10.0,
    "Solar8000/HR": 10.0,
    "Solar8000/PLETH_HR": 10.0,
    "Solar8000/PLETH_SPO2": 10.0,
    "Primus/ETCO2": 10.0,
    "Solar8000/ETCO2": 10.0,
    "Solar8000/ART_MBP": 10.0,
    "Solar8000/ART_SBP": 10.0,
    "Solar8000/ART_DBP": 10.0,
    "Primus/PEEP_MBAR": 30.0,
    "Primus/PIP_MBAR": 30.0,
    "Primus/MV": 30.0,
    "Primus/TV": 30.0,
    "Primus/RR_CO2": 30.0,
    "Solar8000/BT": 600.0,
    # extensiones (no estaban en la tabla de referencia; documentado)
    "Solar8000/NIBP_MBP": 600.0,
    "Solar8000/NIBP_SBP": 600.0,
    "Solar8000/NIBP_DBP": 600.0,
    "Primus/FIO2": 30.0,
}

# Rango plausible por track (imagen + FIO2). Reconstruido: el audit original se
# perdió; límites fisiológicos verificados contra el gate (<= 5% de ventanas
# reales anuladas por plausibilidad).
PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
    "BIS/BIS": (0.0, 100.0),
    "BIS/EMG": (0.0, 100.0),
    "Solar8000/HR": (10.0, 250.0),
    "Solar8000/PLETH_SPO2": (40.0, 100.0),
    "Primus/ETCO2": (10.0, 80.0),
    "Primus/PEEP_MBAR": (0.0, 25.0),
    "Primus/PIP_MBAR": (5.0, 60.0),
    "Primus/MV": (0.0, 25.0),
    "Primus/TV": (50.0, 1500.0),
    "Primus/RR_CO2": (2.0, 60.0),
    "Solar8000/BT": (25.0, 45.0),
    "Solar8000/ART_MBP": (0.0, 300.0),
    "Solar8000/ART_SBP": (0.0, 400.0),
    "Solar8000/ART_DBP": (0.0, 300.0),
    "Primus/FIO2": (15.0, 100.0),
}

# Tracks de evento (bolos). Nombre corto -> columna fuente y unidad.
# El valor en la ventana es la suma en (t - grid_s, t]; la máscara es 1 si la
# columna existe en el caso (aunque la suma sea 0) y 0 si no existe.
BOLUS_SPECS: dict[str, dict[str, str]] = {
    "ppf_bolus": {"source_col": "ppf_bolus_mg", "unit": "mg"},
    "remi_bolus": {"source_col": "remi_bolus_ug", "unit": "ug"},
    "roc_bolus": {"source_col": "roc_bolus_mg", "unit": "mg"},
    "phen_bolus": {"source_col": "phen_bolus_mcg", "unit": "mcg"},
    "eph_bolus": {"source_col": "eph_bolus_mg", "unit": "mg"},
}
BOLUS_SHORT: list[str] = list(BOLUS_SPECS.keys())
BOLUS_SOURCE_COLS: dict[str, str] = {
    spec["source_col"]: short for short, spec in BOLUS_SPECS.items()
}

RARE_DRUG_RATES: set[str] = {
    "Orchestra/PHEN_RATE",
    "Orchestra/NEPI_RATE",
    "Orchestra/EPH_RATE",
    "Orchestra/ROC_RATE",
}

ADVANCED_PREFIXES: tuple[str, ...] = (
    "CardioQ",
    "EV1000",
    "Vigilance",
    "Vigileo",
    "Invos",
    "FMS",
)

DUPLICATE_OF: dict[str, str] = {
    "Solar8000/ETCO2": "Primus/ETCO2",
    "Solar8000/FIO2": "Primus/FIO2",
    "Solar8000/FEO2": "Primus/FEO2",
    "Solar8000/INCO2": "Primus/INCO2",
    "Solar8000/RR_CO2": "Primus/RR_CO2",
    "Solar8000/VENT_RR": "Primus/RR_CO2",
    "Solar8000/VENT_TV": "Primus/TV",
    "Solar8000/VENT_MV": "Primus/MV",
    "Solar8000/VENT_PIP": "Primus/PIP_MBAR",
    "Solar8000/VENT_PPLAT": "Primus/PPLAT_MBAR",
    "Solar8000/VENT_MAWP": "Primus/MAWP_MBAR",
    "Solar8000/VENT_INSP_TM": "Primus/SET_INSP_TM",
    "Solar8000/VENT_SET_TV": "Primus/SET_TV_L",
    "Solar8000/VENT_SET_PCP": "Primus/SET_PIP",
    "Solar8000/VENT_SET_FIO2": "Primus/SET_FIO2",
    "Solar8000/PLETH_HR": "Solar8000/HR",
}

# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unit(name: str) -> str:
    """Unidad (contrato v5) para el campo informativo del registro."""
    if name.endswith(("_CE", "_CP", "_CT")):
        return "ng/mL" if name.startswith("Orchestra/RFTN") else "ug/mL"
    if name.endswith("_VOL"):
        return "mL"
    if name == "Orchestra/RFTN20_RATE":
        return "mL/h"
    if name == "Orchestra/PPF20_RATE":
        return "mL/h"
    if name == "Orchestra/PHEN_RATE":
        return "ug/min"
    if name == "Orchestra/NEPI_RATE":
        return "ug/kg/min"
    if name == "Orchestra/EPH_RATE":
        return "mg/min"
    if name == "Orchestra/ROC_RATE":
        return "mg/kg/h"
    if "FIO2" in name or "FEO2" in name:
        return "%"
    if name.endswith("_BAR") or name.endswith("_MBAR") or name == "Primus/PAMB_MBAR":
        return "mbar" if "PAMB" in name else "mbar"
    if name in ("Solar8000/ART_MBP", "Solar8000/ART_SBP", "Solar8000/ART_DBP",
                "Solar8000/NIBP_MBP", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP",
                "Solar8000/CVP", "Solar8000/ETCO2", "Primus/ETCO2"):
        return "mmHg"
    if name in ("Solar8000/HR", "Solar8000/PLETH_HR"):
        return "bpm"
    if name == "Solar8000/BT":
        return "degC"
    if name == "Primus/MV":
        return "L/min"
    if name == "Primus/TV":
        return "mL"
    if name in ("Primus/RR_CO2", "Solar8000/RR", "Solar8000/RR_CO2"):
        return "1/min"
    if name == "Primus/MAC":
        return "frac"
    if "FLOW" in name or name == "Primus/SET_FRESH_FLOW":
        return "L/min"
    if name == "Primus/SET_AGE":
        return "yr"
    if "ST_" in name:
        return "mV"
    return ""


def _classify(name: str) -> tuple[str, str | None]:
    """Clasifica un track en una familia (W8). Devuelve (group, duplicate_of)."""
    if name in BOLUS_SOURCE_COLS:
        return ("drug_bolus", None)
    if name.startswith("Orchestra/") and name.endswith(("_CE", "_CP", "_CT")):
        return ("drug_pk", None)
    if name.startswith("Orchestra/") and name.endswith("_VOL"):
        return ("drug_volume", None)
    if name in RARE_DRUG_RATES:
        return ("drug_rare", None)
    if name.startswith("Orchestra/"):
        return ("drug_rate", None)
    if name.startswith("Primus/SET_"):
        return ("vent_setpoint", None)
    if name.startswith("Solar8000/VENT_SET_"):
        return ("vent_setpoint", None)
    if name.startswith("Solar8000/VENT_"):
        return ("duplicate", DUPLICATE_OF.get(name))
    if name in DUPLICATE_OF:
        return ("duplicate", DUPLICATE_OF[name])
    if any(name.startswith(p) for p in ADVANCED_PREFIXES):
        return ("advanced_monitor", None)
    if name.startswith("BIS/") or name.startswith("Solar8000/"):
        return ("monitor", None)
    if name.startswith("Primus/"):
        return ("vent_measured", None)
    return ("other", None)


# --------------------------------------------------------------------------
# Inventario y registro de tracks
# --------------------------------------------------------------------------


def _inventory_from_sources(max_per_source: int | None = None) -> dict[str, set[str]]:
    """Inventario de nombres de columna por fuente (leyendo solo los esquemas)."""
    inventory: dict[str, set[str]] = {}
    for source, base in SOURCES.items():
        case_dir = base / "cases"
        paths = sorted(case_dir.glob("*.parquet"))
        if max_per_source is not None and len(paths) > max_per_source:
            paths = paths[:max_per_source]
        names: set[str] = set()
        for p in paths:
            try:
                names.update(pq.read_schema(p).names)
            except Exception:
                continue
        inventory[source] = names
    return inventory


def build_registry(
    inventory: dict[str, set[str]],
    present_in_sample: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Construye el registro final de tracks (W8)."""
    all_names: set[str] = set()
    for names in inventory.values():
        all_names |= names
    all_names -= set(BOLUS_SOURCE_COLS.keys())  # las columnas de bolo van aparte
    all_names.discard("time")  # la columna de tiempo no es un track
    rows = []
    for name in sorted(all_names):
        group, dup_of = _classify(name)
        rows.append(
            {
                "track": name,
                "group": group,
                "kind": "continuous",
                "unit": _unit(name),
                "max_age_s": MAX_AGE.get(name, 30.0),
                "in_image": name in IMAGE_TRACKS or name == FIO2_TRACK,
                "duplicate_of": dup_of,
                "sources": ",".join(s for s, n in inventory.items() if name in n),
                "present_in_sample": (present_in_sample or {}).get(name, True),
            }
        )
    # tracks de evento (bolos), reincorporados con nombre corto y kind="event"
    for short, spec in BOLUS_SPECS.items():
        src = spec["source_col"]
        rows.append(
            {
                "track": short,
                "group": "drug_bolus",
                "kind": "event",
                "unit": spec["unit"],
                "max_age_s": float("nan"),
                "in_image": False,
                "duplicate_of": None,
                "sources": ",".join(s for s, n in inventory.items() if src in n),
                "present_in_sample": (present_in_sample or {}).get(src, True),
            }
        )
    return pd.DataFrame(rows)


def _presence_in_sample(sample_df: pd.DataFrame) -> dict[str, bool]:
    """Presencia por track sobre una muestra de casos (0% -> False)."""
    presence: dict[str, float] = {}
    n_cases = 0
    for _, row in sample_df.iterrows():
        df = _read_case_columns(int(row["caseid"]), str(row["source"]))
        if df is None:
            continue
        n_cases += 1
        for col in df.columns:
            if col == "time":
                continue
            v = df[col].to_numpy()
            frac = float(np.mean(np.isfinite(v.astype(float)))) if len(v) else 0.0
            presence[col] = presence.get(col, 0.0) + frac
    return {k: (v / max(1, n_cases)) > 0.0 for k, v in presence.items()}


# --------------------------------------------------------------------------
# Lectura de casos y clínica
# --------------------------------------------------------------------------


def _read_case_columns(caseid: int, source: str) -> pd.DataFrame | None:
    """Lee el parquet de caso con solo las columnas necesarias."""
    base = SOURCES[source]
    if source == "real":
        path = base / "cases" / f"{caseid:04d}.parquet"
    else:
        path = base / "cases" / f"{caseid}.parquet"
    if not path.exists():
        return None
    needed = (["time"] + IMAGE_TRACKS + [FIO2_TRACK,
                                         "Solar8000/NIBP_MBP",
                                         "Solar8000/NIBP_SBP",
                                         "Solar8000/NIBP_DBP"]
              + list(BOLUS_SOURCE_COLS.keys()))
    schema = pq.read_schema(path)
    cols = [c for c in needed if c in schema.names]
    try:
        return pq.read_table(path, columns=cols).to_pandas()
    except Exception:
        return None


def _dedup_clinical(clinical: pd.DataFrame) -> pd.DataFrame:
    """Deduplica la tabla clínica real (W7)."""
    dup_mask = clinical.duplicated(subset=["caseid"], keep=False)
    if not dup_mask.any():
        return clinical.copy()
    dups = clinical[dup_mask]
    nonnull = dups.notna().sum(axis=1)
    # ¿son idénticos en todas las columnas?
    identical = dups.groupby("caseid", sort=False).nunique(dropna=False).max(axis=1) == 1
    kept: list[pd.Series] = []
    diffs: list[dict] = []
    for caseid, grp in clinical.groupby("caseid", sort=False):
        if len(grp) == 1:
            kept.append(grp.iloc[0])
            continue
        if identical.get(caseid, False):
            kept.append(grp.iloc[0])
        else:
            # reporta columnas en las que difieren
            diff_cols = [
                c for c in grp.columns
                if grp[c].nunique(dropna=False) > 1
            ]
            diffs.append({"caseid": int(caseid), "n_rows": len(grp),
                          "diff_columns": ",".join(diff_cols)})
            # regla: fila con más columnas no nulas; desempate la primera
            best = grp.loc[nonnull.loc[grp.index].idxmax()]
            kept.append(best)
    return pd.DataFrame(kept), diffs


def _read_clinical(source: str) -> tuple[pd.DataFrame, list[dict]]:
    """Lee la tabla clínica de una fuente. Devuelve (df, diffs_dedup)."""
    base = SOURCES[source]
    if source == "real":
        df = pq.read_table(base / "clinical_data_enriched.parquet").to_pandas()
        df, diffs = _dedup_clinical(df)
        return df, diffs
    if source == "synthetic_v5":
        df = pq.read_table(base / "clinical_data.parquet").to_pandas()
        return df, []
    # vaso_reinf_v5 y cf_v5: fila clínica por caso
    case_dir = base / "cases"
    rows = []
    for p in sorted(case_dir.glob("*.parquet")):
        caseid = int(p.stem)
        clin = base / "clinical" / f"{caseid}_clinical.parquet"
        if clin.exists():
            try:
                t = pq.read_table(clin).to_pandas()
                if len(t):
                    rows.append(t.iloc[0])
            except Exception:
                pass
    df = pd.DataFrame(rows)
    return df, []


def list_cases(
    source: str, exclusions: dict[str, set[int]] | None = None
) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """Devuelve (cases_df, logs, dedup_diffs) para una fuente.

    cases_df: caseid, subjectid, source, casestart, caseend, anestart, aneend,
              opstart, opend, cf_pair_id, caseid_a, caseid_b, split_t.
    """
    logs: list[str] = []
    clinical, diffs = _read_clinical(source)
    base = SOURCES[source]
    case_paths = sorted(base.glob("cases/*.parquet"))
    disk_ids = {int(p.stem) for p in case_paths}
    ex = exclusions or {}

    clin = clinical[clinical["caseid"].isin(disk_ids)].copy()
    missing = disk_ids - set(clin["caseid"].astype(int).tolist())
    for cid in sorted(missing):
        logs.append(f"{source}: caso {cid} sin fila clínica -> excluido")

    if source == "cf_v5":
        # pares por metadata
        pairs: list[dict] = []
        for p in sorted((base / "metadata").glob("cf_pair_*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            pairs.append(
                {
                    "caseid_a": int(d["caseid_a"]),
                    "caseid_b": int(d["caseid_b"]),
                    "split_t": float(d.get("split_t", d.get("split_t_s", np.nan))),
                    "lever": d.get("lever"),
                    "collection": d.get("collection"),
                }
            )
        pair_by_case: dict[int, dict] = {}
        parity_mismatches = 0
        for pr in pairs:
            a, b = pr["caseid_a"], pr["caseid_b"]
            pair_by_case[a] = pr
            pair_by_case[b] = pr
            if b != a + 1 or a % 2 == 0:
                parity_mismatches += 1
        if parity_mismatches:
            logs.append(f"cf_v5: {parity_mismatches} pares cuya metadata contradice la paridad impar/par")
        clin = clin.copy()
        clin["cf_pair_id"] = clin["caseid"].map(
            lambda cid: min(pair_by_case[cid]["caseid_a"], pair_by_case[cid]["caseid_b"])
            if cid in pair_by_case else None
        )
        clin["caseid_a"] = clin["caseid"].map(lambda cid: pair_by_case[cid]["caseid_a"] if cid in pair_by_case else None)
        clin["caseid_b"] = clin["caseid"].map(lambda cid: pair_by_case[cid]["caseid_b"] if cid in pair_by_case else None)
        clin["split_t"] = clin["caseid"].map(lambda cid: pair_by_case[cid]["split_t"] if cid in pair_by_case else None)
        clin["lever"] = clin["caseid"].map(lambda cid: pair_by_case[cid]["lever"] if cid in pair_by_case else None)
        # sujeto del par = par completo (por metadata)
        clin["subjectid"] = clin["cf_pair_id"].map(lambda p: f"cf:{int(p)}" if p is not None else None)

    # exclusiones (antes del split)
    excluded = [c for c in clin["caseid"].astype(int).tolist() if c in ex.get(source, set())]
    # expandir exclusiones CF al par completo
    if source == "cf_v5":
        excl_set = set(ex.get(source, set()))
        to_add = set()
        for cid in list(excl_set):
            row = clin[clin["caseid"] == cid]
            if len(row) and row.iloc[0]["cf_pair_id"] is not None:
                a = int(row.iloc[0]["caseid_a"])
                b = int(row.iloc[0]["caseid_b"])
                to_add.add(a)
                to_add.add(b)
        excl_set |= to_add
        excluded = sorted(excl_set)
    n_before = len(clin)
    clin = clin[~clin["caseid"].astype(int).isin(set(excluded))].copy()
    if excluded:
        logs.append(f"{source}: {n_before - len(clin)} casos excluidos por exclusions_path (de {n_before})")

    keep = ["caseid", "subjectid", "casestart", "caseend",
            "anestart", "aneend", "opstart", "opend"]
    for extra in ("cf_pair_id", "caseid_a", "caseid_b", "split_t", "lever"):
        if extra in clin.columns:
            keep.append(extra)
    out = clin[keep].copy()
    out["source"] = source
    out["caseid"] = out["caseid"].astype(np.int64)
    return out, logs, diffs


def make_split(cases: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Asigna split por paciente (subjectid), estratificado por fuente, 85/15."""
    rng = np.random.default_rng(seed)
    rows = []
    for source, grp in cases.groupby("source", sort=False):
        subjects = grp["subjectid"].dropna().unique().tolist()
        subjects = sorted(subjects, key=str)
        order = rng.permutation(len(subjects))
        n_val = int(round(len(subjects) * VAL_FRAC))
        val_set = {subjects[i] for i in order[:n_val]}
        for sid in subjects:
            rows.append({"subjectid": sid, "source": source,
                         "split": "val" if sid in val_set else "train"})
    split_df = pd.DataFrame(rows)
    return cases.merge(split_df, on=["subjectid", "source"], how="left")


# --------------------------------------------------------------------------
# Procesamiento de un caso
# --------------------------------------------------------------------------


def _resample(
    t: np.ndarray, v: np.ndarray, grid: np.ndarray, max_age: float
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill con edad máxima sobre la rejilla. Devuelve (valores, mask)."""
    vf = v.astype(np.float64)
    nonnull = np.isfinite(vf)
    if not nonnull.any():
        return (
            np.full(len(grid), np.nan, dtype=np.float32),
            np.zeros(len(grid), dtype=np.uint8),
        )
    tt = t[nonnull]
    vv = vf[nonnull]
    idx = np.searchsorted(tt, grid, side="right") - 1
    valid = (idx >= 0) & ((grid - tt[np.maximum(idx, 0)]) <= max_age)
    vals = np.full(len(grid), np.nan, dtype=np.float32)
    vals[valid] = vv[idx[valid]]
    return vals, valid.astype(np.uint8)


def _arrays_equal(a, b) -> bool:
    """Compara dos columnas de ventanas tolerando NaN solo en numéricas."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    if a.dtype == object or a.dtype.kind in "OUS":
        aa = pd.Series(a)
        bb = pd.Series(b)
        return bool(((aa == bb) | (aa.isna() & bb.isna())).all())
    return bool(np.array_equal(a, b))


def _phase_from_clinical(t_arr: np.ndarray, opstart, opend) -> np.ndarray:
    phase = np.full(len(t_arr), "unknown", dtype=object)
    has_op = opstart is not None and np.isfinite(opstart)
    has_end = opend is not None and np.isfinite(opend)
    if has_op:
        phase[t_arr < opstart] = "induction"
        if has_end:
            phase[(t_arr >= opstart) & (t_arr < opend)] = "maintenance"
            phase[t_arr >= opend] = "emergence"
        else:
            phase[t_arr >= opstart] = "maintenance"
    elif has_end:
        phase[t_arr >= opend] = "emergence"
    return phase


def _process_case(case_row: dict) -> tuple[pd.DataFrame, dict]:
    """Procesa un caso. Devuelve (windows_df, stats)."""
    caseid = int(case_row["caseid"])
    source = str(case_row["source"])
    t0 = _time.time()

    df = _read_case_columns(caseid, source)
    if df is None:
        raise RuntimeError(f"no se pudo leer el parquet de {source}/{caseid}")

    t = df["time"].to_numpy(dtype=np.float64)
    if len(t) == 0:
        return pd.DataFrame(), {"caseid": caseid, "n_windows": 0,
                                "n_trimmed_head": 0, "n_trimmed_tail": 0,
                                "art_coverage": 0.0, "bp_source": "none"}

    t_min, t_max = float(t[0]), float(t[-1])
    casestart = float(case_row.get("casestart", 0.0) or 0.0)
    caseend = float(case_row.get("caseend", 0.0) or 0.0)
    anestart = float(case_row.get("anestart", 0.0) or 0.0)
    aneend = case_row.get("aneend")
    aneend = float(aneend) if aneend is not None and np.isfinite(aneend) else t_max
    opstart = case_row.get("opstart")
    opstart = float(opstart) if opstart is not None and np.isfinite(opstart) else np.nan
    opend = case_row.get("opend")
    opend = float(opend) if opend is not None and np.isfinite(opend) else np.nan

    # W4: rejilla acotada
    a0 = max(anestart, casestart, t_min)
    a1 = min(aneend, caseend if caseend > 0 else aneend, t_max)
    if a1 <= a0:
        return pd.DataFrame(), {"caseid": caseid, "n_windows": 0,
                                "n_trimmed_head": 0, "n_trimmed_tail": 0,
                                "art_coverage": 0.0, "bp_source": "none"}
    grid = np.arange(a0, a1, GRID_S)
    n = len(grid)
    if n == 0:
        return pd.DataFrame(), {"caseid": caseid, "n_windows": 0,
                                "n_trimmed_head": 0, "n_trimmed_tail": 0,
                                "art_coverage": 0.0, "bp_source": "none"}

    raw: dict[str, np.ndarray] = {}   # valores en rejilla
    mraw: dict[str, np.ndarray] = {}  # máscara de presencia (sin plausibilidad)
    for col in IMAGE_TRACKS + [FIO2_TRACK, "Solar8000/NIBP_MBP",
                               "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP"]:
        if col in df.columns:
            vals, mask = _resample(t, df[col].to_numpy(), grid, MAX_AGE.get(col, 30.0))
            raw[col] = vals
            mraw[col] = mask
        else:
            raw[col] = np.full(n, np.nan, dtype=np.float32)
            mraw[col] = np.zeros(n, dtype=np.uint8)

    # W5: bp_source
    art_cov = float(mraw["Solar8000/ART_MBP"].mean()) if n else 0.0
    nibp_cov = float(mraw["Solar8000/NIBP_MBP"].mean()) if n else 0.0
    if art_cov >= ART_MIN_COV:
        bp_source = "art"
    elif nibp_cov >= NIBP_MIN_COV:
        bp_source = "nibp"
        for col in ("Solar8000/ART_MBP", "Solar8000/ART_SBP", "Solar8000/ART_DBP"):
            src = col.replace("ART", "NIBP")
            raw[col] = raw.get(src, np.full(n, np.nan, dtype=np.float32))
            mraw[col] = mraw.get(src, np.zeros(n, dtype=np.uint8))
    else:
        bp_source = "none"
        for col in ("Solar8000/ART_MBP", "Solar8000/ART_SBP", "Solar8000/ART_DBP"):
            raw[col] = np.full(n, np.nan, dtype=np.float32)
            mraw[col] = np.zeros(n, dtype=np.uint8)

    # W6: máscara de plausibilidad
    mfin: dict[str, np.ndarray] = {}
    for col in IMAGE_TRACKS + [FIO2_TRACK]:
        lo, hi = PLAUSIBLE_RANGE.get(col, (-np.inf, np.inf))
        val = raw[col]
        ok = np.isfinite(val) & (val >= lo) & (val <= hi)
        mfin[col] = (mraw[col].astype(bool) & ok).astype(np.uint8)

    # W4: recorte de extremos donde TODOS los tracks de la imagen tienen máscara 0
    union = np.zeros(n, dtype=bool)
    for col in IMAGE_TRACKS:
        union |= mfin[col].astype(bool)
    idx_present = np.where(union)[0]
    if idx_present.size == 0:
        return pd.DataFrame(), {"caseid": caseid, "n_windows": 0,
                                "n_trimmed_head": n, "n_trimmed_tail": n,
                                "art_coverage": art_cov, "bp_source": bp_source}
    first, last = int(idx_present[0]), int(idx_present[-1])
    sl = slice(first, last + 1)
    n_trim_head = first
    n_trim_tail = n - 1 - last

    t_win = np.round(grid[sl]).astype(np.int32)
    n_win = len(t_win)

    data: dict[str, np.ndarray] = {
        "caseid": np.full(n_win, caseid, dtype=np.int32),
        "t": t_win,
        "source": np.full(n_win, source, dtype=object),
        "split": np.full(n_win, str(case_row.get("split", "train")), dtype=object),
        "bp_source": np.full(n_win, bp_source, dtype=object),
        "phase_from_clinical": _phase_from_clinical(
            t_win.astype(np.float64), opstart, opend
        ),
        "t_since_opstart": (
            (t_win.astype(np.float32) - np.float32(opstart))
            if np.isfinite(opstart) else np.full(n_win, np.nan, dtype=np.float32)
        ),
    }
    for col in IMAGE_TRACKS:
        data[col] = raw[col][sl].astype(np.float32)
        data[f"m_{col}"] = mfin[col][sl].astype(np.uint8)
        data[f"m_raw_{col}"] = mraw[col][sl].astype(np.uint8)
    data[FIO2_TRACK] = raw[FIO2_TRACK][sl].astype(np.float32)
    data[f"m_{FIO2_TRACK}"] = mfin[FIO2_TRACK][sl].astype(np.uint8)

    # B1: bolos (eventos). Valor = suma en (t - grid_s, t]; máscara = 1 si la
    # columna existe en el caso (aunque la suma sea 0), 0 si no existe.
    for short, spec in BOLUS_SPECS.items():
        src = spec["source_col"]
        if src in df.columns:
            col = df[src].to_numpy(dtype=np.float64)
            spk = np.isfinite(col) & (col > 0.0)
            out = np.zeros(n, dtype=np.float64)
            if spk.any():
                tb = t[spk]
                vv = col[spk]
                idx = np.searchsorted(grid, tb, side="left")
                ok = (idx >= 0) & (idx < n)
                np.add.at(out, idx[ok], vv[ok])
            data[short] = out[sl].astype(np.float32)
            data[f"m_{short}"] = np.ones(n_win, dtype=np.uint8)
        else:
            data[short] = np.full(n_win, np.nan, dtype=np.float32)
            data[f"m_{short}"] = np.zeros(n_win, dtype=np.uint8)

    stats = {
        "caseid": caseid,
        "source": source,
        "split": str(case_row.get("split", "train")),
        "n_windows": n_win,
        "n_trimmed_head": n_trim_head,
        "n_trimmed_tail": n_trim_tail,
        "art_coverage": round(art_cov, 6),
        "bp_source": bp_source,
        "process_time_s": round(_time.time() - t0, 4),
    }
    return pd.DataFrame(data), stats


# --------------------------------------------------------------------------
# Prueba de la fase A
# --------------------------------------------------------------------------


def _sample_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """Muestra de la prueba: 40 reales + 20 v5 + 10 vaso + 5 pares CF."""
    rng = np.random.default_rng(20240917)
    real = cases[cases.source == "real"]
    synth = cases[cases.source == "synthetic_v5"]
    vaso = cases[cases.source == "vaso_reinf_v5"]
    cf = cases[cases.source == "cf_v5"]

    real_s = real.iloc[np.sort(rng.choice(len(real), 40, replace=False))]
    synth_s = synth.iloc[np.sort(rng.choice(len(synth), 20, replace=False))]
    vaso_s = vaso.iloc[np.sort(rng.choice(len(vaso), 10, replace=False))]
    # 5 pares CF (10 casos), al menos 3 pares completos y 3 palancas distintas
    pairs = sorted(cf["cf_pair_id"].dropna().astype(int).unique().tolist())
    sel_pairs = [int(p) for p in rng.choice(pairs, 5, replace=False)]
    cf_s = cf[cf["cf_pair_id"].astype(float).isin([float(p) for p in sel_pairs])].copy()
    return pd.concat([real_s, synth_s, vaso_s, cf_s], ignore_index=True)


def _estimate_projected(cases: pd.DataFrame, sample_caseids: set[int],
                        sample_n_windows: int, bytes_per_window: float) -> dict:
    """Proyecta el tamaño total del dataset por fuente (GB)."""
    def grid_len(row):
        a0 = max(float(row.get("anestart", 0) or 0), float(row.get("casestart", 0) or 0))
        ane = row.get("aneend")
        ane = float(ane) if ane is not None and np.isfinite(ane) else float(row.get("caseend") or 0)
        ce = float(row.get("caseend") or 0)
        a1 = min(ane, ce if ce > 0 else ane)
        return max(0, int((a1 - a0) / GRID_S))

    sample_cases = cases[cases["caseid"].astype(int).isin(sample_caseids)]
    sample_grid = sum(grid_len(r) for _, r in sample_cases.iterrows())
    trim_ratio = (sample_n_windows / sample_grid) if sample_grid else 1.0

    out = {}
    for source, grp in cases.groupby("source", sort=False):
        total_grid = int(grp.apply(grid_len, axis=1).sum())
        proj = total_grid * trim_ratio
        out[source] = proj * bytes_per_window / 1e9
    return out


def _load_exclusions() -> dict[str, set[int]]:
    """Lee las exclusiones, normalizando el nombre de fuente."""
    exclusions: dict[str, set[int]] = {}
    if EXCLUSIONS_PATH.exists():
        exc = pd.read_csv(EXCLUSIONS_PATH)
        for source, grp in exc.groupby("source"):
            exclusions[_normalize_source(str(source))] = set(int(c) for c in grp["caseid"])
    return exclusions


def run_phase_a_test() -> dict:
    """Ejecuta la prueba de la fase A y devuelve un dict con todos los números."""
    print("=" * 78)
    print("FASE A — prueba de window.py (v5)")
    print("=" * 78)

    exclusions = _load_exclusions()
    if EXCLUSIONS_PATH.exists():
        exc = pd.read_csv(EXCLUSIONS_PATH)
        print(f"exclusions_path: {EXCLUSIONS_PATH.name} -> {len(exc)} casos")

    all_cases, all_diffs, all_logs = [], [], []
    for source in SOURCES:
        df, logs, diffs = list_cases(source, exclusions)
        all_cases.append(df)
        all_diffs.extend(diffs)
        all_logs.extend(logs)
        for line in logs:
            print("  [log]", line)
    cases = pd.concat(all_cases, ignore_index=True)
    cases = make_split(cases, SEED)

    # ---- punto 1: inventario y registro ----
    inventory = _inventory_from_sources()
    sample_df = _sample_cases(cases)
    sample_ids = set(int(c) for c in sample_df["caseid"])
    presence = _presence_in_sample(sample_df)
    registry = build_registry(inventory, presence)
    print(f"\n[1] Inventario de tracks (por fuente): "
          + ", ".join(f"{s}={len(v)}" for s, v in sorted(inventory.items())))
    print(f"    Registro final: {len(registry)} tracks (eventos excluidos).")
    print("    Por familia:", dict(registry.group.value_counts().sort_index()))
    others = registry[registry.group == "other"]
    print(f"    Tracks en 'other': {len(others)}")
    for _, r in others.iterrows():
        print(f"      - {r['track']}")

    # ---- punto 2: split ----
    print("\n[2] Split por paciente (unidad subjectid):")
    for source, grp in cases.groupby("source", sort=False):
        n_subj = grp["subjectid"].nunique()
        n_cases = len(grp)
        tr = int((grp["split"] == "train").sum())
        va = int((grp["split"] == "val").sum())
        print(f"    {source}: {n_cases} casos / {n_subj} sujetos -> train {tr}, val {va}")
    real = cases[cases.source == "real"]
    multi = real.groupby("subjectid").caseid.nunique()
    multi = multi[multi > 1]
    print(f"    Sujetos reales con más de un caso: {len(multi)}")
    # pares CF: ninguno partido
    cf = cases[cases.source == "cf_v5"]
    cf_pairs = cf.groupby("cf_pair_id")["split"].nunique()
    broken = int((cf_pairs > 1).sum())
    print(f"    Pares CF partidos entre splits: {broken}")
    both = set(cases[cases.source == "real"]["subjectid"]) & set(cases[cases.source != "real"]["subjectid"])
    print(f"    Sujetos presentes en >1 fuente (colisión): {len(both)}")

    # ---- procesar muestra ----
    print("\n    Procesando muestra (80 casos)...")
    sample_stats = []
    sample_windows = []
    for _, row in sample_df.iterrows():
        w, s = _process_case(dict(row))
        sample_stats.append(s)
        if len(w):
            sample_windows.append(w)
    sample_windows_df = pd.concat(sample_windows, ignore_index=True) if sample_windows else pd.DataFrame()

    # ---- punto 3: ventanas, recortes, cobertura ART ----
    print("\n[3] Ventanas por caso:")
    n_win = sum(s["n_windows"] for s in sample_stats)
    print(f"    Total ventanas muestra: {n_win}")
    print(f"    n_trimmed_head: media {np.mean([s['n_trimmed_head'] for s in sample_stats]):.1f}, "
          f"p50 {np.percentile([s['n_trimmed_head'] for s in sample_stats], 50):.0f}, "
          f"max {max(s['n_trimmed_head'] for s in sample_stats)}")
    print(f"    n_trimmed_tail: media {np.mean([s['n_trimmed_tail'] for s in sample_stats]):.1f}, "
          f"p50 {np.percentile([s['n_trimmed_tail'] for s in sample_stats], 50):.0f}, "
          f"max {max(s['n_trimmed_tail'] for s in sample_stats)}")
    src_map = dict(zip(sample_df["caseid"].astype(int), sample_df["source"].astype(str)))
    real_stats = [s for s in sample_stats if src_map[s["caseid"]] == "real"]
    art_real = [s["art_coverage"] for s in real_stats]
    print(f"    Cobertura ART real (n={len(art_real)}): p5 {np.percentile(art_real,5):.3f}, "
          f"p50 {np.percentile(art_real,50):.3f}, p95 {np.percentile(art_real,95):.3f}")
    print("    bp_source (muestra):", dict(sample_windows_df.groupby("bp_source").size()) if len(sample_windows_df) else {})

    # ---- punto 4: tasa de máscara por variable ----
    print("\n[4] Tasa de máscara por variable de la imagen (real vs sintético):")
    mask_rates = []
    for col in IMAGE_TRACKS:
        for grp_name in ("real", "synthetic"):
            is_real = grp_name == "real"
            sub = sample_windows_df[
                (sample_windows_df["source"].map(
                    lambda s: (s == "real")) == is_real)
            ]
            if len(sub) == 0:
                continue
            present = sub[f"m_raw_{col}"].mean()
            plausible = sub[f"m_{col}"].mean()
            mask_rates.append({
                "track": col, "source": grp_name,
                "presence_raw": present, "mask_plausible": plausible,
                "frac_implausible_observed": 1 - plausible / present if present > 0 else 0.0,
            })
    mr = pd.DataFrame(mask_rates)
    for col in IMAGE_TRACKS:
        r = mr[(mr.track == col) & (mr.source == "real")]
        s = mr[(mr.track == col) & (mr.source == "synthetic")]
        if len(r) and len(s):
            d = s.iloc[0]["presence_raw"] - r.iloc[0]["presence_raw"]
            print(f"    {col:26s} real {r.iloc[0]['presence_raw']:.3f}  "
                  f"sint {s.iloc[0]['presence_raw']:.3f}  diff {d:+.3f}  "
                  f"implaus_obs real {r.iloc[0]['frac_implausible_observed']:.4f} "
                  f"sint {s.iloc[0]['frac_implausible_observed']:.4f}")

    # ---- punto 5: rangos ----
    print("\n[5] Rangos por variable (real vs sintético):")
    for col in IMAGE_TRACKS + [FIO2_TRACK]:
        row = []
        for grp_name in ("real", "synthetic"):
            is_real = grp_name == "real"
            sub = sample_windows_df[
                (sample_windows_df["source"].map(lambda s: s == "real") == is_real)
            ]
            v = sub.loc[sub[f"m_{col}"] == 1, col].astype(float).dropna()
            if len(v):
                row.append(f"{grp_name} n={len(v)} min={v.min():.1f} p1={np.percentile(v,1):.1f} "
                           f"p50={np.percentile(v,50):.1f} p99={np.percentile(v,99):.1f} max={v.max():.1f}")
            else:
                row.append(f"{grp_name} (sin datos)")
        print(f"    {col:26s} " + " | ".join(row))

    # ---- punto 6: prefijo CF idéntico ----
    print("\n[6] Pares CF: prefijo idéntico bit a bit hasta split_t:")
    cf_sample = sample_df[sample_df.source == "cf_v5"]
    prefix_ok = 0
    prefix_total = 0
    for pair_id in sorted(cf_sample["cf_pair_id"].unique(), key=int):
        pair = cf_sample[cf_sample.cf_pair_id == pair_id]
        if len(pair) != 2:
            continue
        a = pair.iloc[0]
        b = pair.iloc[1]
        split_t = float(a["split_t"]) if np.isfinite(a["split_t"]) else float("inf")
        wa = sample_windows_df[(sample_windows_df.caseid == a["caseid"]) &
                               (sample_windows_df.t < split_t)]
        wb = sample_windows_df[(sample_windows_df.caseid == b["caseid"]) &
                               (sample_windows_df.t < split_t)]
        cols = sorted((set(wa.columns) & set(wb.columns)) - {"caseid", "subjectid"})
        if len(wa) == len(wb):
            same = True
            for c in cols:
                if not _arrays_equal(wa[c].to_numpy(), wb[c].to_numpy()):
                    same = False
                    break
        else:
            same = False
        prefix_total += 1
        prefix_ok += int(same)
        print(f"    par {pair_id} (lever={a['lever']}, split_t={split_t:.1f}s): "
              f"{'OK' if same else 'DIFF'}")
    print(f"    Prefijo idéntico: {prefix_ok}/{prefix_total}")

    # ---- punto 7: duplicados clínicos ----
    print("\n[7] Duplicados clínicos (real):")
    print(f"    filas duplicadas no idénticas: {len(all_diffs)}")
    for d in all_diffs[:10]:
        print(f"      caseid {d['caseid']}: {d['n_rows']} filas, columnas: {d['diff_columns']}")

    # ---- punto 8: estimación de tamaño ----
    print("\n[8] Estimación de tamaño del dataset completo:")
    # medir bytes por ventana en memoria con zstd
    if len(sample_windows_df):
        tbl = _windows_to_table(sample_windows_df)
        sink = pa.BufferOutputStream()
        pq.write_table(tbl, sink, compression="zstd")
        n_bytes = sink.getvalue().size
        bytes_per_win = n_bytes / len(sample_windows_df)
        print(f"    bytes/ventana (zstd, medido): {bytes_per_win:.2f}")
        proj = _estimate_projected(cases, sample_ids, n_win, bytes_per_win)
        total = sum(proj.values())
        for source, gb in proj.items():
            print(f"    {source}: {gb:.2f} GB")
        print(f"    TOTAL proyectado: {total:.2f} GB")
    else:
        proj, total, bytes_per_win = {}, 0.0, 0.0
        print("    (sin ventanas en muestra)")

    # ---- tabla del gate ----
    print("\n[GATE]")
    real_pres = {t: float(mr[(mr.track == t) & (mr.source == "real")]["presence_raw"].iloc[0])
                 for t in ("Solar8000/HR", "Solar8000/PLETH_SPO2", "BIS/BIS", "Primus/ETCO2")}
    synth_pres = {t: float(mr[(mr.track == t) & (mr.source == "synthetic")]["presence_raw"].iloc[0])
                  for t in ("Solar8000/HR", "Solar8000/PLETH_SPO2", "BIS/BIS", "Primus/ETCO2")}
    pres_ok = all(abs(synth_pres[t] - real_pres[t]) <= 0.10 for t in real_pres)
    no_shared_subject = len(both) == 0
    no_broken_cf = broken == 0
    no_missing_clinical = not any("sin fila clínica" in l for l in all_logs)
    max_implaus_real = 0.0
    for col in IMAGE_TRACKS:
        r = mr[(mr.track == col) & (mr.source == "real")]
        if len(r):
            max_implaus_real = max(max_implaus_real, r.iloc[0]["frac_implausible_observed"])
    implaus_ok = max_implaus_real <= 0.05
    size_ok = total <= 25.0

    gate_rows = [
        ("Ningún subjectid en ambos splits", no_shared_subject, no_shared_subject),
        ("Ningún par CF partido", no_broken_cf, no_broken_cf),
        ("0 casos sin fila clínica", no_missing_clinical, no_missing_clinical),
        ("Prefijo CF idéntico bit a bit (todos los pares)", prefix_ok == prefix_total,
         prefix_ok == prefix_total),
        ("Presencia HR/SpO2/BIS/EtCO2 sint ±0.10 de real", pres_ok, pres_ok),
        ("Plausibilidad no anula >5% de ventanas reales", implaus_ok, implaus_ok),
        ("Tamaño proyectado <= 25 GB", size_ok, size_ok),
    ]
    for name, val, ok in gate_rows:
        print(f"    {name:52s} -> {val}  {'OK' if ok else 'FALLA'}")
    gate_pass = all(ok for _, _, ok in gate_rows)
    print(f"    RESULTADO GATE: {'PASA' if gate_pass else 'FALLA'}")

    return {
        "registry": registry,
        "cases": cases,
        "sample_stats": sample_stats,
        "mask_rates": mr,
        "prefix_ok": prefix_ok,
        "prefix_total": prefix_total,
        "broken_cf": broken,
        "no_shared_subject": no_shared_subject,
        "max_implaus_real": max_implaus_real,
        "presence_real": real_pres,
        "presence_synth": synth_pres,
        "proj_gb": proj,
        "total_gb": total,
        "bytes_per_window": bytes_per_win,
        "gate_pass": gate_pass,
        "diffs": all_diffs,
    }


# --------------------------------------------------------------------------
# Conversión a tabla parquet (dtypes fijos)
# --------------------------------------------------------------------------


def _windows_to_table(df: pd.DataFrame) -> pa.Table:
    arrays = []
    for col in df.columns:
        series = df[col]
        if col == "caseid":
            arr = pa.array(series.to_numpy(dtype=np.int32), type=pa.int32())
        elif col == "t":
            arr = pa.array(series.to_numpy(dtype=np.int32), type=pa.int32())
        elif col in ("source", "split", "bp_source", "phase_from_clinical"):
            arr = pa.array(series.astype(str).to_numpy(), type=pa.dictionary(pa.int32(), pa.string()))
        elif col.startswith("m_") or col.startswith("m_raw_"):
            arr = pa.array(series.to_numpy(dtype=np.uint8), type=pa.uint8())
        else:
            arr = pa.array(series.to_numpy(dtype=np.float32), type=pa.float32())
        arrays.append((col, arr))
    return pa.Table.from_arrays([a for _, a in arrays], names=[n for n, _ in arrays])


# --------------------------------------------------------------------------
# Fase B: generación
# --------------------------------------------------------------------------


def _worker_process(case_row: dict) -> tuple[dict, pd.DataFrame | None]:
    try:
        w, s = _process_case(case_row)
        return s, (w if len(w) else None)
    except Exception as e:  # noqa: BLE001
        return {"caseid": int(case_row["caseid"]), "error": repr(e),
                "source": str(case_row["source"])}, None


def generate(max_workers: int = 14) -> None:
    """Fase B: genera el dataset completo en data/windows_v1 (reanudable)."""
    print("=" * 78)
    print("FASE B — generación del dataset")
    print("=" * 78)

    exclusions = _load_exclusions()
    if EXCLUSIONS_PATH.exists():
        exc = pd.read_csv(EXCLUSIONS_PATH)
        print(f"exclusions_path: {EXCLUSIONS_PATH.name} -> {len(exc)} casos")

    all_cases = []
    for source in SOURCES:
        df, logs, _ = list_cases(source, exclusions)
        all_cases.append(df)
        for line in logs:
            print("  [log]", line)
    cases = make_split(pd.concat(all_cases, ignore_index=True), SEED)
    cases = cases.sort_values(["source", "split", "caseid"], kind="stable").reset_index(drop=True)

    # inventario + registro (present_in_sample sobre una muestra fija)
    inventory = _inventory_from_sources()
    sample_df = _sample_cases(cases)
    presence = _presence_in_sample(sample_df)
    registry = build_registry(inventory, presence)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    win_dir = OUT_DIR / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)

    cases_path = OUT_DIR / "cases.parquet"
    errors_path = OUT_DIR / "errors.csv"
    done: set[int] = set()
    if cases_path.exists():
        try:
            done = set(int(c) for c in pq.read_table(cases_path, columns=["caseid"])["caseid"].to_pylist())
            print(f"    resume: {len(done)} casos ya procesados")
        except Exception:
            done = set()
    todo = cases[~cases["caseid"].astype(int).isin(done)].reset_index(drop=True)
    print(f"    casos a procesar: {len(todo)} (total {len(cases)})")

    case_rows = [dict(r) for _, r in todo.iterrows()]

    # buffers por (source, split)
    buffers: dict[tuple[str, str], list[pd.DataFrame]] = {}
    buffer_rows: dict[tuple[str, str], int] = {}
    part_idx: dict[tuple[str, str], int] = {}
    # resume: continuar la numeración de partes existentes
    for key in [("real", "train"), ("real", "val"), ("synthetic_v5", "train"),
                ("synthetic_v5", "val"), ("vaso_reinf_v5", "train"),
                ("vaso_reinf_v5", "val"), ("cf_v5", "train"), ("cf_v5", "val")]:
        sub = win_dir / f"source={key[0]}" / f"split={key[1]}"
        if sub.exists():
            existing = sorted(sub.glob("part-*.parquet"))
            if existing:
                part_idx[key] = max(int(p.stem.split("-")[1]) for p in existing) + 1
    cases_out: list[dict] = []
    errors_out: list[dict] = []
    n_done = 0
    t_start = _time.time()

    def flush(key: tuple[str, str], final: bool = False) -> None:
        if key not in buffers or not buffers[key]:
            return
        df = pd.concat(buffers[key], ignore_index=True)
        buffers[key] = []
        buffer_rows[key] = 0
        if key not in part_idx:
            part_idx[key] = 0
        out_sub = win_dir / f"source={key[0]}" / f"split={key[1]}"
        out_sub.mkdir(parents=True, exist_ok=True)
        path = out_sub / f"part-{part_idx[key]:05d}.parquet"
        pq.write_table(_windows_to_table(df), path, compression="zstd")
        part_idx[key] += 1

    if len(case_rows) == 0:
        print("    nada que procesar (resume completo)")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            # ex.map conserva el orden de case_rows (ordenado por source, split, caseid)
            for i, (stat, w) in enumerate(ex.map(_worker_process, case_rows, chunksize=8)):
                n_done += 1
                if "error" in stat:
                    errors_out.append({"caseid": stat["caseid"],
                                       "source": stat.get("source", ""),
                                       "error": stat["error"]})
                else:
                    stat["subjectid"] = str(case_rows[i].get("subjectid"))
                    stat["source"] = str(case_rows[i].get("source"))
                    stat["split"] = str(case_rows[i].get("split"))
                    stat["t_first"] = int(w["t"].iloc[0]) if w is not None and len(w) else None
                    stat["t_last"] = int(w["t"].iloc[-1]) if w is not None and len(w) else None
                    cases_out.append(stat)
                    if w is not None and len(w):
                        key = (str(case_rows[i]["source"]), str(case_rows[i]["split"]))
                        buffers.setdefault(key, []).append(w)
                        buffer_rows[key] = buffer_rows.get(key, 0) + len(w)
                        if buffer_rows[key] >= PART_SIZE:
                            flush(key)
                if n_done % 500 == 0:
                    print(f"    {n_done}/{len(case_rows)} casos, "
                          f"{_time.time() - t_start:.0f}s")

        for key in list(buffers.keys()):
            flush(key)

    # casos ya hechos (resume) + nuevos
    if done:
        try:
            old_cases = pq.read_table(cases_path).to_pandas()
            all_cases_out = pd.concat([old_cases, pd.DataFrame(cases_out)], ignore_index=True)
        except Exception:
            all_cases_out = pd.DataFrame(cases_out)
    else:
        all_cases_out = pd.DataFrame(cases_out)
    if len(all_cases_out):
        all_cases_out = all_cases_out.drop_duplicates(subset=["caseid"], keep="last")
        all_cases_out.to_parquet(cases_path, index=False)

    if errors_out:
        pd.DataFrame(errors_out).to_csv(errors_path, index=False)
        print(f"    errores: {len(errors_out)} en {errors_path.name}")

    # split.parquet
    split_df = cases[["subjectid", "source", "split"]].drop_duplicates().copy()
    split_df["subjectid"] = split_df["subjectid"].astype(str)
    split_df.to_parquet(OUT_DIR / "split.parquet", index=False)

    # registry.parquet
    reg_out = registry.copy()
    if "duplicate_of" in reg_out.columns:
        reg_out["duplicate_of"] = reg_out["duplicate_of"].fillna("").astype(str)
    reg_out.to_parquet(OUT_DIR / "registry.parquet", index=False)

    # manifest.json
    manifest = {
        "date": pd.Timestamp.now().isoformat(),
        "seed": SEED,
        "grid_s": GRID_S,
        "max_age": MAX_AGE,
        "plausible_range": {k: list(v) for k, v in PLAUSIBLE_RANGE.items()},
        "image_tracks": IMAGE_TRACKS,
        "bolus_tracks": BOLUS_SHORT,
        "bolus_specs": {s: dict(spec) for s, spec in BOLUS_SPECS.items()},
        "exclusions_path": str(EXCLUSIONS_PATH),
        "exclusions_sha256": _sha256(EXCLUSIONS_PATH) if EXCLUSIONS_PATH.exists() else None,
        "n_cases_by_source_split": {
            f"{s}|{sp}": int(n)
            for (s, sp), n in cases.groupby(["source", "split"]).size().items()
        },
        "n_windows_by_source_split": _count_windows_by_partition(win_dir),
        "total_bytes": _dir_bytes(OUT_DIR),
        "window_py_sha256": _sha256(Path(__file__)),
        "n_cases_processed": int(len(pq.read_table(cases_path))),
        "n_errors": int(len(errors_out)),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("    manifest.json escrito")
    print(f"    TOTAL: {len(cases_out)} casos procesados, {len(errors_out)} errores, "
          f"{_time.time() - t_start:.0f}s")
    # verificación de cobertura: casos en cases.parquet + errors.csv == lista de entrada
    n_total = len(cases)
    done_ids = set(done) | ({int(s["caseid"]) for s in cases_out} if cases_out else set())
    err_ids = {int(e["caseid"]) for e in errors_out}
    input_ids = {int(c) for c in cases["caseid"]}
    covered = done_ids | err_ids
    missing = sorted(input_ids - covered)
    if missing:
        print(f"    AVISO: {len(missing)} casos sin procesar y sin error: {missing[:10]}")
    print(f"    cobertura cases.parquet + errors.csv vs lista de entrada: {len(covered)}/{n_total}")


def _count_windows_by_partition(win_dir: Path) -> dict:
    out = {}
    for p in sorted(win_dir.glob("source=*/split=*/part-*.parquet")):
        try:
            md = pq.read_metadata(p)
            key = str(p.relative_to(win_dir).parent).replace("\\", "/")
            out[key] = out.get(key, 0) + int(md.num_rows)
        except Exception:
            pass
    return out


def _dir_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def integrity_check(out_dir: Path | None = None) -> bool:
    """Chequeo de integridad sobre el dataset completo."""
    out_dir = out_dir or OUT_DIR
    win_dir = out_dir / "windows"
    print("=" * 78)
    print("CHEQUEO DE INTEGRIDAD")
    print("=" * 78)
    ok = True
    parts = sorted(win_dir.glob("source=*/split=*/part-*.parquet"))
    print(f"particiones: {len(parts)}")

    # 1) unicidad de (caseid, t) y 2) ningún caseid en más de un split
    seen = {}
    dup_ck = 0
    n_rows = 0
    for p in parts:
        tbl = pq.read_table(p, columns=["caseid", "t", "source", "split"])
        df = tbl.to_pandas()
        n_rows += len(df)
        src = df["source"].astype(str).tolist()
        spl = df["split"].astype(str).tolist()
        cid = df["caseid"].tolist()
        tt = df["t"].tolist()
        for s_, p_, c_, t_ in zip(src, spl, cid, tt):
            key = (s_, p_)
            seen.setdefault(key, set())
            pair = (int(c_), int(t_))
            if pair in seen[key]:
                dup_ck += 1
            seen[key].add(pair)
    caseid_split = {}
    for (source, split), pairs in seen.items():
        for cid, _ in pairs:
            caseid_split.setdefault(int(cid), set()).add((source, split))
    cross_split = sum(1 for cid, s in caseid_split.items() if len(s) > 1)
    print(f"filas: {n_rows}, duplicados (caseid,t): {dup_ck}, caseids en >1 split: {cross_split}")
    ok &= dup_ck == 0 and cross_split == 0

    # 3) conteos coherentes con manifest
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    counts = _count_windows_by_partition(win_dir)
    mani_counts = manifest.get("n_windows_by_source_split", {})
    same_counts = all(counts.get(k, 0) == v for k, v in mani_counts.items())
    print(f"conteos por partición vs manifest: {'OK' if same_counts else 'DIFF'}")
    ok &= same_counts

    # 4) columnas de la imagen y sus máscaras con mismo dtype en todas las particiones
    expected = (["caseid", "t", "source", "split", "bp_source",
                 "phase_from_clinical", "t_since_opstart"]
                + IMAGE_TRACKS
                + [f"m_{c}" for c in IMAGE_TRACKS]
                + [f"m_raw_{c}" for c in IMAGE_TRACKS]
                + [FIO2_TRACK, f"m_{FIO2_TRACK}"]
                + BOLUS_SHORT
                + [f"m_{s}" for s in BOLUS_SHORT])
    dtype_report = {}
    all_ok_cols = True
    for p in parts:
        sc = pq.read_schema(p)
        for c in expected:
            if c not in sc.names:
                all_ok_cols = False
                dtype_report.setdefault(c, set()).add("MISSING")
                continue
            dtype_report.setdefault(c, set()).add(str(sc.field(c).type))
    for c in expected:
        dts = dtype_report.get(c, set())
        if len(dts) != 1:
            all_ok_cols = False
            print(f"    columna {c}: dtypes inconsistentes {dts}")
    print(f"columnas de la imagen + máscaras presentes y con dtype uniforme: "
          f"{'OK' if all_ok_cols else 'FALLA'}")
    ok &= all_ok_cols

    # errores.csv
    errors_path = out_dir / "errors.csv"
    if errors_path.exists():
        err = pd.read_csv(errors_path)
        print(f"errors.csv: {len(err)} casos")
        for _, r in err.iterrows():
            print(f"    {r['caseid']} ({r.get('source','')}): {r['error']}")
    else:
        print("errors.csv: ausente (0 errores)")

    cases_df = pq.read_table(out_dir / "cases.parquet").to_pandas()
    print(f"cases.parquet: {len(cases_df)} filas")
    print("RESULTADO INTEGRIDAD:", "OK" if ok else "FALLA")
    return ok


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="window.py")
    ap.add_argument("command", choices=["test", "generate", "check"])
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    if args.command == "test":
        run_phase_a_test()
    elif args.command == "generate":
        generate(max_workers=args.workers)
    elif args.command == "check":
        integrity_check()


if __name__ == "__main__":
    main()
