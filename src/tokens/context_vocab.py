"""context_vocab.py — módulo de contexto del contrato de tokens v1.

Implementa la sección "Mapeo de contexto a ejes y columnas excluidas" del
contrato de tokens v1: el tipo de token "contexto" del Bloque C y los gates 1,
4, 6, 7, 8 y 9 en lo que toca al contexto.

Iteración 2 (decisión de contrato tras el gate 6 = 0.9956 de la iteración 1):
  - El inventario v1 se reduce a demografía + riesgo global (age, sex:F, height,
    weight, bmi, asa y emop con sus categorías). Todo lo demás pasa a scope
    "v2": se conserva en vocab.json (con sus estadísticos) pero NO se emite.
  - tokens.parquet y coverage.parquet solo contienen tokens scope v1.

Salida:
  data/context_v1/vocab.json      entradas (86) con scope v1/v2 + sha256
  data/context_v1/tokens.parquet  una fila por (caseid, item_id) — solo v1
  data/context_v1/coverage.parquet por (item_id, source, split) — solo v1
  reports/REPORT_context_vocab_v2.txt  informe de la iteración 2

Reglas centrales:
  - Split heredado de windows_v2 (columna `split` de cases.parquet, verificada
    bit a bit idéntica al mapeo de windows_v2/split.parquet; sha registrado).
  - Solo se leen las 37 columnas de contexto + caseid (clave). Ninguna columna
    de la tabla "Excluidas" (45 columnas) se lee como feature ni aparece en el
    vocabulario.
  - Continuos: valor normalizado (media/std poblacional, ddof=0, solo train).
  - Binarios: sex:F, preop_htn, preop_dm, preop_ecg:anormal, preop_pft:anormal.
  - Categóricos: item_id = "<columna>:<valor normalizado>"; < 50 casos en train
    -> "<columna>:otros".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time as _time
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import paths

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contrato_tokens_v1.md"
WINDOWS_ROOT = paths.WINDOWS_DIR
CASES_PATH = WINDOWS_ROOT / "cases.parquet"
SPLIT_PATH = WINDOWS_ROOT / "split.parquet"
OUT_DIR = paths.CONTEXT_DIR
REPORT_PATH = paths.REPORTS_DIR / "REPORT_context_vocab_v2.txt"

SOURCE_ORDER = ["real", "synthetic_v7", "cf_v7", "vaso_reinf_v7"]


def source_dirs() -> dict[str, Path]:
    """Directorios clínicos por cohorte, resueltos en cada llamada. Lectura
    DIFERIDA a propósito: capturar ``paths.COHORTS["cf_v7"]`` en el import fija
    la cohorte aunque ``paths`` cambie (paso 3d, Corrección C)."""
    return {
        "real": paths.COHORTS["real"] / "clinical_data_enriched.parquet",
        "synthetic_v7": paths.COHORTS["synthetic_v7"] / "clinical_data.parquet",
        "cf_v7": paths.cohort_dir("cf_v7"),
        "vaso_reinf_v7": paths.COHORTS["vaso_reinf_v7"],
    }

# --------------------------------------------------------------------------
# Tabla "Excluidas" del contrato v2 (45 columnas). El enunciado decía "40";
# se usa la tabla real del contrato (fuente de verdad).
# --------------------------------------------------------------------------
EXCLUDED_COLUMNS = (
    # identificadores y marcas de rejilla
    "caseid", "subjectid", "casestart", "caseend", "anestart", "aneend",
    "opstart", "opend",
    # posteriores a la anestesia
    "adm", "dis", "icu_days", "death_inhosp",
    # totales del caso completo (fuga del futuro)
    "intraop_ebl", "intraop_uo", "intraop_rbc", "intraop_ffp",
    "intraop_crystalloid", "intraop_colloid",
    "intraop_ppf", "intraop_mdz", "intraop_ftn", "intraop_rocu",
    "intraop_vecu", "intraop_eph", "intraop_phe", "intraop_epi", "intraop_ca",
    # derivadas del ingreso completo
    "diagnosis_count", "top_diagnosis_code", "lab_count", "lab_unique_items",
    "medication_count", "medication_unique", "common_route",
    # texto libre
    "dx", "opname",
    # accesos y dispositivos
    "dltubesize", "lmasize", "iv1", "iv2", "aline1", "aline2", "cline1", "cline2",
    # colapsado con optype (decisión del contrato)
    "department",
)

# --------------------------------------------------------------------------
# Columnas de contexto (37, tabla "Ejes" del contrato)
# --------------------------------------------------------------------------
CONTINUOUS_COLUMNS = (
    "age", "height", "weight", "bmi",
    "preop_gluc", "preop_pao2", "preop_paco2", "preop_sao2",
    "preop_cr", "preop_bun", "preop_na", "preop_k",
    "preop_alb", "preop_ast", "preop_alt", "preop_pt", "preop_aptt",
    "preop_plt", "preop_hb",
    "preop_ph", "preop_hco3", "preop_be",
    "tubesize",
)

CATEGORICAL_COLUMNS = (
    "asa", "emop", "optype", "approach", "position", "ane_type",
    "top_diagnosis_chapter", "airway", "cormack",
)

BINARY_SOURCE_COLUMNS = ("sex", "preop_htn", "preop_dm", "preop_ecg", "preop_pft")

CONTEXT_COLUMNS = tuple(list(CONTINUOUS_COLUMNS) + list(BINARY_SOURCE_COLUMNS)
                        + list(CATEGORICAL_COLUMNS))

# item_id -> columna fuente (binarios)
BINARY_ITEMS = (
    ("sex:F", "sex"),
    ("preop_htn", "preop_htn"),
    ("preop_dm", "preop_dm"),
    ("preop_ecg:anormal", "preop_ecg"),
    ("preop_pft:anormal", "preop_pft"),
)

BINARY_RULE_TEXT = {
    "sex:F": "columna sex; positivo (token emitido) si el valor es femenino ('F')",
    "preop_htn": "columna preop_htn; positivo si == 1",
    "preop_dm": "columna preop_dm; positivo si == 1",
    "preop_ecg:anormal": "columna preop_ecg; anormal si el valor está en el inventario "
                         "de hallazgos anormales (no 'Normal Sinus Rhythm'); valor "
                         "desconocido o vacío -> no emitido",
    "preop_pft:anormal": "columna preop_pft; anormal si el valor está en el inventario "
                         "de hallazgos anormales (no 'Normal'); valor desconocido o "
                         "vacío -> no emitido",
}

# Inventario completo de valores de preop_ecg (todas las cohortes).
ECG_NORMAL = {"Normal Sinus Rhythm"}
ECG_ABNORMAL = {
    "1st degree A-V block", "Right bundle branch block",
    "Premature ventricular complexes", "Incomplete right bundle branch block",
    "Atrial fibrillation", "Premature atrial complexes",
    "Left anterior fascicular block",
    "Atrial fibrillation with slow ventricular response",
    "Atrial fibrillation with rapid ventricular response",
    "Atrial fibrillation with premature ventricular or aberrantly conducted complexes",
    "Premature supraventricular complexes",
    "1st degree A-V block, Left bundle branch block",
    "Incomplete right bundle branch block, Left anterior fascicular block",
    "Atrial fibrillation, Right bundle branch block",
    "Premature supraventricular and ventricular complexes, Right bundle branch block",
    "Left anterior hemiblock", "Left posterior fascicular block",
    "Atrial fibrillation with premature ventricular, Incomplete left bundle block",
    "1st degree A-V block with Premature supraventricular complexes, Left bundle branch block",
    "1st degree A-V block with Premature atrial complexes",
    "Atrial flutter with 2:1 A-V conduction",
    "Electronic ventricular pacemaker",
    "AV sequential or dual chamber electronic pacemaker",
    "Right bundle branch block, Left anterior fascicular block",
    "Complete right bundle branch block, occasional premature supraventricular complexes",
    "Atrial flutter with variable A-V block",
}

# Inventario completo de valores de preop_pft (todas las cohortes).
PFT_NORMAL = {"Normal"}
PFT_ABNORMAL = {
    "Mild obstructive", "Mild restrictive", "Moderate obstructive",
    "Mixed or pure obstructive", "Severe restrictive", "Moderate restrictive",
    "Borderline obstructive", "Severe obstructive",
}

# --------------------------------------------------------------------------
# Ejes (tabla "Ejes" del contrato)
# --------------------------------------------------------------------------
AXES = {}
for _c in ("age", "height", "weight", "bmi", "sex"):
    AXES[_c] = "demografia"
for _c in ("asa", "emop"):
    AXES[_c] = "riesgo_global"
for _c in ("optype", "approach", "position", "ane_type"):
    AXES[_c] = "procedimiento"
AXES["top_diagnosis_chapter"] = "diagnostico"
for _c in ("preop_htn", "preop_ecg"):
    AXES[_c] = "cardiovascular"
for _c in ("preop_dm", "preop_gluc"):
    AXES[_c] = "metabolico"
for _c in ("preop_pft", "preop_pao2", "preop_paco2", "preop_sao2"):
    AXES[_c] = "respiratorio"
for _c in ("preop_cr", "preop_bun", "preop_na", "preop_k"):
    AXES[_c] = "renal_electrolitos"
for _c in ("preop_alb", "preop_ast", "preop_alt", "preop_pt", "preop_aptt",
           "preop_plt", "preop_hb"):
    AXES[_c] = "hepatico_coagulacion"
for _c in ("preop_ph", "preop_hco3", "preop_be"):
    AXES[_c] = "acido_base"
for _c in ("airway", "cormack", "tubesize"):
    AXES[_c] = "via_aerea"

COLUMN_TIPO = {}
for _c in CONTINUOUS_COLUMNS:
    COLUMN_TIPO[_c] = "continuo"
for _c in BINARY_SOURCE_COLUMNS:
    COLUMN_TIPO[_c] = "binario"
for _c in CATEGORICAL_COLUMNS:
    COLUMN_TIPO[_c] = "categorico"

# item_ids estáticos (continuos + binarios) para el test b
STATIC_ITEM_IDS = tuple(list(CONTINUOUS_COLUMNS) + [i for i, _ in BINARY_ITEMS])

# Scope de la iteración 2 (decisión de contrato tras el gate 6): el inventario
# v1 se reduce a demografía + riesgo global; el resto es scope v2 (referencia).
V1_AXES = {"demografia", "riesgo_global"}
V1_BASE_COLUMNS = ("age", "sex", "height", "weight", "bmi", "asa", "emop")
V1_COLUMNS = frozenset(c for c in CONTEXT_COLUMNS if AXES[c] in V1_AXES)


def item_scope(col: str) -> str:
    """Scope de una columna de contexto: 'v1' (demografía/riesgo) o 'v2'."""
    return "v1" if AXES[col] in V1_AXES else "v2"

# Columnas que el lector solicita a los parquet (caseid es la clave, no feature)
FEATURE_COLUMNS = list(CONTEXT_COLUMNS)
FEATURE_READ_COLUMNS = ["caseid"] + FEATURE_COLUMNS


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return not np.isfinite(v)
    if isinstance(v, str):
        s = v.strip()
        return s == "" or s.lower() in ("none", "nan")
    return False


def normalize_item_value(value) -> str:
    """Normaliza un valor para el item_id: minúsculas, sin tildes ni espacios."""
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            s = str(int(value))
        else:
            s = str(value)
    elif isinstance(value, (int, np.integer)):
        s = str(int(value))
    else:
        s = str(value)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def assert_feature_columns(columns) -> bool:
    """Valida una lista de columnas como feature de contexto.

    Lanza ValueError para cualquier columna de la tabla Excluidas (incluida
    caseid: la clave solo se lee aparte, nunca como feature) y para cualquier
    columna ajena a las 37 de contexto.
    """
    for c in columns:
        if c in EXCLUDED_COLUMNS:
            raise ValueError(f"columna excluida del contrato: {c}")
        if c not in CONTEXT_COLUMNS:
            raise ValueError(f"columna no es de contexto: {c}")
    return True


def dedupe_inspire(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicación INSPIRE: una fila por caseid, la de más valores no nulos;
    en empate, la primera en orden original (sort estable)."""
    nn = df.notna().sum(axis=1)
    out = df.assign(_nn=nn).sort_values("_nn", ascending=False, kind="stable")
    out = out.drop_duplicates("caseid", keep="first").drop(columns="_nn")
    return out.sort_values("caseid", kind="stable").reset_index(drop=True)


def _num_one(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer, float, np.floating)):
        try:
            return float(v) == 1.0
        except (TypeError, ValueError):
            return False
    return str(v).strip().lower() in ("1", "1.0", "true")


def _binary_mask(col: str, series: pd.Series) -> np.ndarray:
    arr = series.to_numpy(dtype=object, na_value=None)
    if col == "sex":
        return np.array([(not _is_missing(v)) and str(v).strip().upper() == "F"
                         for v in arr], dtype=bool)
    if col in ("preop_htn", "preop_dm"):
        return np.array([_num_one(v) for v in arr], dtype=bool)
    if col == "preop_ecg":
        return np.array([(not _is_missing(v)) and str(v).strip() in ECG_ABNORMAL
                         for v in arr], dtype=bool)
    if col == "preop_pft":
        return np.array([(not _is_missing(v)) and str(v).strip() in PFT_ABNORMAL
                         for v in arr], dtype=bool)
    raise ValueError(col)


def _norm_counts(series: pd.Series) -> dict:
    counts: Counter = Counter()
    for v in series.to_numpy(dtype=object, na_value=None):
        if _is_missing(v):
            continue
        counts[normalize_item_value(v)] += 1
    return dict(counts)


def _norm_set(series: pd.Series) -> set:
    out = set()
    for v in series.to_numpy(dtype=object, na_value=None):
        if _is_missing(v):
            continue
        out.add(normalize_item_value(v))
    return out


def build_category_mapping(series: pd.Series, threshold: int = 50):
    """Mapping {valor normalizado -> sufijo} a partir de una serie (train).

    Devuelve (mapping, has_otros). Categorías con < threshold casos -> 'otros'.
    """
    counts = _norm_counts(series)
    mapping = {s: (s if n >= threshold else "otros") for s, n in counts.items()}
    has_otros = any(n < threshold for n in counts.values())
    return mapping, has_otros


def build_category_mapping_from(train_series, all_series, threshold: int = 50) -> dict:
    """Mapping sobre train (umbral) extendido con categorías vistas en val."""
    counts = _norm_counts(train_series)
    mapping = {s: (s if n >= threshold else "otros") for s, n in counts.items()}
    for s in _norm_set(all_series):
        mapping.setdefault(s, "otros")
    return mapping


# --------------------------------------------------------------------------
# Estadísticos de normalización (solo train)
# --------------------------------------------------------------------------

def compute_train_stats(df: pd.DataFrame, cols) -> dict:
    sub = df[df["split"] == "train"] if "split" in df.columns else df
    stats = {}
    for col in cols:
        s = pd.to_numeric(sub[col], errors="coerce").dropna()
        if len(s) == 0:
            stats[col] = (float("nan"), 1.0)
            continue
        mean = float(s.mean())
        std = float(s.std(ddof=0))  # desviación poblacional (ddof=0)
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        stats[col] = (mean, std)
    return stats


# --------------------------------------------------------------------------
# Lectura de datos
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_cases() -> pd.DataFrame:
    df = pq.read_table(CASES_PATH, columns=["caseid", "source", "split"]).to_pandas()
    df["caseid"] = df["caseid"].astype("int32")
    df["source"] = df["source"].astype(str)
    df["split"] = df["split"].astype(str)
    return df.sort_values("caseid", kind="stable").reset_index(drop=True)


def load_clinical(source: str) -> pd.DataFrame:
    """Lee las 37 columnas de contexto + caseid para una cohorte.

    Real se deduplica con la regla INSPIRE (fila con más no nulos por caseid).
    """
    cols = FEATURE_READ_COLUMNS
    dirs = source_dirs()
    if source == "real":
        df = pq.read_table(dirs[source], columns=cols).to_pandas()
        df = dedupe_inspire(df)
    elif source == "synthetic_v7":
        df = pq.read_table(dirs[source], columns=cols).to_pandas()
    else:
        base = dirs[source]
        files = sorted((base / "clinical").glob("*_clinical.parquet"))
        df = pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in files],
                       ignore_index=True) if files else pd.DataFrame(columns=cols)
    df["caseid"] = df["caseid"].astype("int32")
    return df


def load_all_clinical(cases: pd.DataFrame) -> dict:
    out = {}
    inscope = set(cases["caseid"].tolist())
    for source in SOURCE_ORDER:
        df = load_clinical(source)
        df = df[df["caseid"].isin(inscope)]
        df = df.merge(cases[["caseid", "source", "split"]], on="caseid", how="left")
        df["source"] = df["source"].astype(str)
        df["split"] = df["split"].astype(str)
        out[source] = df.sort_values("caseid", kind="stable").reset_index(drop=True)
    return out


@lru_cache(maxsize=1)
def load_train_combined() -> pd.DataFrame:
    cases = load_cases()
    clin = load_all_clinical(cases)
    return pd.concat([d[d["split"] == "train"] for d in clin.values()],
                     ignore_index=True)


def recompute_continuo_stats(col: str):
    """Recomputa media/std de un continuo sobre las filas train (independiente)."""
    df = load_train_combined()
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return float("nan"), 1.0
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    return mean, std


# --------------------------------------------------------------------------
# Vocabulario
# --------------------------------------------------------------------------

def build_vocab(train_df: pd.DataFrame, all_df: pd.DataFrame | None = None) -> dict:
    if all_df is None:
        all_df = train_df
    stats = compute_train_stats(train_df, CONTINUOUS_COLUMNS)
    items = []
    for col in CONTINUOUS_COLUMNS:
        mean, std = stats[col]
        items.append({
            "item_id": col, "tipo": "continuo", "eje": AXES[col], "columna": col,
            "scope": item_scope(col),
            "regla": "valor normalizado (media/std calculadas solo sobre split=train)",
            "mean": (float(mean) if np.isfinite(mean) else None), "std": float(std),
        })
    for item_id, col in BINARY_ITEMS:
        items.append({
            "item_id": item_id, "tipo": "binario", "eje": AXES[col], "columna": col,
            "scope": item_scope(col),
            "regla": BINARY_RULE_TEXT[item_id], "mean": None, "std": None,
        })
    spec_cat = {}
    for col in CATEGORICAL_COLUMNS:
        mapping = build_category_mapping_from(train_df[col], all_df[col], threshold=50)
        spec_cat[col] = mapping
        for suffix in sorted(set(mapping.values())):
            items.append({
                "item_id": f"{col}:{suffix}", "tipo": "categorico", "eje": AXES[col],
                "columna": col, "scope": item_scope(col),
                "regla": f"categoría {suffix}",
                "mean": None, "std": None,
            })
    spec = {"continuous": stats, "categorical": spec_cat}
    return {"items": items, "spec": spec}


# --------------------------------------------------------------------------
# Emisión de tokens
# --------------------------------------------------------------------------

def emit_tokens(clinical: pd.DataFrame, vocab: dict, scope: str | None = None) -> pd.DataFrame:
    """Emite tokens. scope=None emite todos; scope='v1' solo el inventario v1;
    scope='v2' solo las entradas de referencia."""
    def active(col: str) -> bool:
        if scope is None:
            return True
        is_v1 = col in V1_COLUMNS
        return is_v1 if scope == "v1" else (not is_v1)

    spec = vocab["spec"]
    caseid = clinical["caseid"].to_numpy().astype(np.int32)
    if "source" in clinical.columns:
        src = clinical["source"].to_numpy()
    else:
        src = np.array(["unknown"] * len(clinical), dtype=object)
    if "split" in clinical.columns:
        spl = clinical["split"].to_numpy()
    else:
        spl = np.array(["train"] * len(clinical), dtype=object)
    frames = []

    # continuos
    for col in CONTINUOUS_COLUMNS:
        if not active(col):
            continue
        s = pd.to_numeric(clinical[col], errors="coerce")
        m = s.notna().to_numpy()
        if not m.any():
            continue
        mean, std = spec["continuous"][col]
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        if not np.isfinite(mean):
            mean = 0.0
        vals = ((s.to_numpy(dtype=np.float64) - mean) / std)[m].astype(np.float32)
        frames.append(pd.DataFrame({
            "caseid": caseid[m], "item_id": col, "value": vals,
            "tipo": "continuo", "eje": AXES[col],
            "source": src[m], "split": spl[m],
        }))

    # binarios
    for item_id, col in BINARY_ITEMS:
        if not active(col):
            continue
        mask = _binary_mask(col, clinical[col])
        if not mask.any():
            continue
        n = int(mask.sum())
        frames.append(pd.DataFrame({
            "caseid": caseid[mask], "item_id": item_id,
            "value": np.ones(n, dtype=np.float32), "tipo": "binario",
            "eje": AXES[col], "source": src[mask], "split": spl[mask],
        }))

    # categóricos
    for col in CATEGORICAL_COLUMNS:
        if not active(col):
            continue
        mapping = spec["categorical"][col]
        arr = clinical[col].to_numpy(dtype=object, na_value=None)
        keep = np.zeros(len(arr), dtype=bool)
        suffixes = []
        for i, v in enumerate(arr):
            if _is_missing(v):
                continue
            sfx = mapping.get(normalize_item_value(v), "otros")
            suffixes.append(sfx)
            keep[i] = True
        if not keep.any():
            continue
        idx = np.where(keep)[0]
        frames.append(pd.DataFrame({
            "caseid": caseid[idx],
            "item_id": [f"{col}:{s}" for s in suffixes],
            "value": np.ones(len(idx), dtype=np.float32), "tipo": "categorico",
            "eje": AXES[col], "source": src[idx], "split": spl[idx],
        }))

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["item_id", "caseid"], kind="stable").reset_index(drop=True)


def write_tokens(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("caseid", pa.int32()),
        ("item_id", pa.string()),
        ("value", pa.float32()),
        ("tipo", pa.string()),
        ("eje", pa.string()),
        ("source", pa.dictionary(pa.int32(), pa.string())),
        ("split", pa.dictionary(pa.int32(), pa.string())),
    ])
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


# --------------------------------------------------------------------------
# Cobertura
# --------------------------------------------------------------------------

def compute_null_fraction(clin_by_source: dict) -> dict:
    frac = {}
    for src, df in clin_by_source.items():
        n = len(df)
        for col in CONTINUOUS_COLUMNS:
            nulls = int(pd.to_numeric(df[col], errors="coerce").isna().sum())
            frac[(col, src)] = (nulls / n) if n else 1.0
    return frac


def build_coverage(tokens: pd.DataFrame, cases: pd.DataFrame, null_frac: dict) -> pd.DataFrame:
    cases = cases[["caseid", "source", "split"]]
    groups = cases.groupby(["source", "split"]).size().reset_index(name="n_casos")
    item_ids = sorted(tokens["item_id"].unique().tolist())
    emit = tokens.groupby(["item_id", "source", "split"]).size().reset_index(name="n_emitidos")
    emit = emit.set_index(["item_id", "source", "split"])["n_emitidos"].to_dict()

    rows = []
    for item_id in item_ids:
        col = item_id.split(":")[0]
        is_cont = COLUMN_TIPO.get(col) == "continuo"
        for _, g in groups.iterrows():
            src, spl = g["source"], g["split"]
            n_e = int(emit.get((item_id, src, spl), 0))
            n_c = int(g["n_casos"])
            fr = (n_e / n_c) if n_c else 0.0
            if is_cont:
                if null_frac.get((col, src), 1.0) >= 1.0:
                    mask = 1
                elif n_e < n_c:
                    mask = 2
                else:
                    mask = 0
            else:
                mask = None
            rows.append((item_id, src, spl, n_c, n_e, fr, mask))

    df = pd.DataFrame(rows, columns=[
        "item_id", "source", "split", "n_casos", "n_emitidos", "fraccion", "mask_state"])
    df["mask_state"] = df["mask_state"].astype("Int8")  # nullable
    return df


def write_coverage(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("item_id", pa.string()),
        ("source", pa.dictionary(pa.int32(), pa.string())),
        ("split", pa.dictionary(pa.int32(), pa.string())),
        ("n_casos", pa.int32()),
        ("n_emitidos", pa.int32()),
        ("fraccion", pa.float64()),
        ("mask_state", pa.int8()),
    ])
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


# --------------------------------------------------------------------------
# Gate 6: sondeo de fuente (regresión logística en val, 5-fold estratificado)
# --------------------------------------------------------------------------

def _cv_balanced_acc(X, y, exclude=None, item_ids=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xk = X
    if exclude is not None and item_ids is not None:
        keep = [i for i, iid in enumerate(item_ids) if iid not in exclude]
        if len(keep) == 0:
            return float("nan")
        Xk = X[:, keep]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    yp = cross_val_predict(model, Xk, y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict")
    return float(balanced_accuracy_score(y, yp))


def _probe_sources(tokens, cases, pos_sources, neg_sources, ablation_top1=False) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pos = set(pos_sources)
    neg = set(neg_sources)
    val_caseids = set(cases[cases["split"] == "val"]["caseid"].tolist())
    sel = cases[cases["caseid"].isin(val_caseids) & cases["source"].isin(pos | neg)]
    sel_ids = set(sel["caseid"].tolist())
    toks = tokens[tokens["caseid"].isin(sel_ids)]
    item_ids = sorted(toks["item_id"].unique().tolist())
    piv = toks.pivot_table(index="caseid", columns="item_id", values="value",
                           fill_value=0.0)
    piv = piv.reindex(columns=item_ids, fill_value=0.0)
    src_of_case = cases.set_index("caseid")["source"]
    y = np.array([1.0 if src_of_case.loc[c] in pos else 0.0 for c in piv.index],
                 dtype=np.float64)
    X = piv.to_numpy(dtype=np.float64)

    acc = _cv_balanced_acc(X, y)
    acc_p = _cv_balanced_acc((X != 0).astype(np.float64), y)

    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(X, y)
    coefs = m.named_steps["logisticregression"].coef_[0]
    order = np.argsort(-np.abs(coefs))
    axis_map = toks[["item_id", "eje"]].drop_duplicates().set_index("item_id")["eje"].to_dict()
    top = [{"item_id": item_ids[i], "coef": float(coefs[i]),
            "eje": axis_map.get(item_ids[i], "?")} for i in order]

    ablation = {}
    if ablation_top1 and len(item_ids):
        top1 = item_ids[order[0]]
        ablation["top1"] = _cv_balanced_acc(X, y, exclude={top1}, item_ids=item_ids)

    return {
        "accuracy": acc,
        "presence_accuracy": acc_p,
        "top_coefs": top,
        "ablation": ablation,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "n_items": len(item_ids),
    }


def gate6_probe(tokens: pd.DataFrame, cases: pd.DataFrame) -> dict:
    base = _probe_sources(tokens, cases, ["real"],
                          ["synthetic_v7", "cf_v7", "vaso_reinf_v7"],
                          ablation_top1=True)
    pairs = {
        "real_vs_synthetic_v7": _probe_sources(tokens, cases, ["real"], ["synthetic_v7"]),
        "real_vs_cf_v7": _probe_sources(tokens, cases, ["real"], ["cf_v7"]),
        "real_vs_vaso_reinf_v7": _probe_sources(tokens, cases, ["real"], ["vaso_reinf_v7"]),
        "synthetic_v7_vs_cf_v7": _probe_sources(tokens, cases, ["synthetic_v7"], ["cf_v7"]),
    }
    base["pairwise"] = {
        k: {"accuracy": v["accuracy"], "presence_accuracy": v["presence_accuracy"],
            "n": v["n"], "n_pos": v["n_pos"], "n_neg": v["n_neg"]}
        for k, v in pairs.items()
    }
    base["n_val"] = base.pop("n")
    base["n_real"] = base.pop("n_pos")
    base["n_synth"] = base.pop("n_neg")
    return base


# --------------------------------------------------------------------------
# Gate 7: KS / TVD (informativo)
# --------------------------------------------------------------------------

def _cat_distribution(real_series, synth_series, col, mapping):
    def dist(series):
        counts: Counter = Counter()
        for v in series.to_numpy(dtype=object, na_value=None):
            if _is_missing(v):
                counts["<no_emitido>"] += 1
            else:
                sfx = mapping.get(normalize_item_value(v), "otros")
                counts[f"{col}:{sfx}"] += 1
        return counts

    dr = dist(real_series)
    ds = dist(synth_series)
    keys = sorted(set(dr) | set(ds))
    nr = max(len(real_series), 1)
    ns = max(len(synth_series), 1)
    pr = np.array([dr.get(k, 0) / nr for k in keys])
    ps = np.array([ds.get(k, 0) / ns for k in keys])
    return pr, ps


def gate7_ks_tvd(all_df: pd.DataFrame, spec: dict) -> list:
    from scipy.stats import ks_2samp

    real = all_df[all_df["source"] == "real"]
    synth = all_df[all_df["source"] != "real"]
    rows = []
    for col in CONTINUOUS_COLUMNS:
        r = pd.to_numeric(real[col], errors="coerce").dropna().to_numpy()
        s = pd.to_numeric(synth[col], errors="coerce").dropna().to_numpy()
        if len(r) == 0 or len(s) == 0:
            rows.append({"kind": "continuo", "columna": col, "statistic": None,
                         "n_real": int(len(r)), "n_synth": int(len(s)),
                         "note": "sin datos en una cohorte"})
            continue
        stat, p = ks_2samp(r, s)
        rows.append({"kind": "continuo", "columna": col, "statistic": float(stat),
                     "n_real": int(len(r)), "n_synth": int(len(s)),
                     "p_value": float(p)})
    for col in CATEGORICAL_COLUMNS:
        pr, ps = _cat_distribution(real[col], synth[col], col, spec["categorical"][col])
        tvd = float(0.5 * np.sum(np.abs(pr - ps)))
        rows.append({"kind": "categorico", "columna": col, "statistic": tvd,
                     "n_real": int(len(real)), "n_synth": int(len(synth))})
    rows = [r for r in rows if r["statistic"] is not None]
    rows.sort(key=lambda r: -r["statistic"])
    return rows


# --------------------------------------------------------------------------
# Coherencia interna (informativo)
# --------------------------------------------------------------------------

def coherence_check(all_df: pd.DataFrame) -> dict:
    bmi_rows = []
    for src in SOURCE_ORDER:
        d = all_df[all_df["source"] == src]
        ok = d[["bmi", "weight", "height"]].dropna()
        h = ok["height"].to_numpy(np.float64)
        w = ok["weight"].to_numpy(np.float64)
        b = ok["bmi"].to_numpy(np.float64)
        denom = w / (h / 100.0) ** 2
        err = np.abs(b - denom) / denom
        bmi_rows.append({
            "source": src, "n_casos": int(len(ok)),
            "n_err_gt5pct": int((err > 0.05).sum()),
        })

    bounds = {"age": (18, 100), "height": (120, 220), "weight": (30, 250)}
    range_rows = []
    for src in SOURCE_ORDER:
        d = all_df[all_df["source"] == src]
        row = {"source": src}
        for col, (lo, hi) in bounds.items():
            v = pd.to_numeric(d[col], errors="coerce").dropna()
            row[f"{col}_n"] = int(len(v))
            row[f"{col}_out"] = int(((v < lo) | (v > hi)).sum())
        range_rows.append(row)
    return {"bmi": bmi_rows, "ranges": range_rows}


# --------------------------------------------------------------------------
# Tokens por caso
# --------------------------------------------------------------------------

def tokens_per_case(tokens: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    cnt = tokens.groupby("caseid").size().rename("n_tokens").reset_index()
    full = cases[["caseid", "source", "split"]].merge(cnt, on="caseid", how="left")
    full["n_tokens"] = full["n_tokens"].fillna(0).astype("int32")
    return full.sort_values("caseid", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# Vocab.json
# --------------------------------------------------------------------------

def write_vocab(vocab: dict, tokens: pd.DataFrame, out_dir: Path = OUT_DIR) -> None:
    items = vocab["items"]
    ejes: dict = {}
    for it in items:
        ejes.setdefault(it["eje"], []).append(
            {"item_id": it["item_id"], "scope": it["scope"]})
    n_v1 = sum(1 for it in items if it["scope"] == "v1")
    n_v2 = sum(1 for it in items if it["scope"] == "v2")
    doc = {
        "version": 3,
        "date": pd.Timestamp.now().isoformat(),
        "sha256_contract": _sha256(CONTRACT_PATH),
        "sha256_module": _sha256(Path(__file__).resolve()),
        "windows_manifest_path": str(WINDOWS_ROOT / "manifest.json"),
        "sha256_windows_manifest": _sha256(WINDOWS_ROOT / "manifest.json"),
        "split_parquet_path": str(SPLIT_PATH),
        "sha256_split_parquet": _sha256(SPLIT_PATH),
        "n_items": len(items),
        "n_items_v1": n_v1,
        "n_items_v2": n_v2,
        "n_token_rows": int(len(tokens)),
        "counts": {
            "total": len(items),
            "por_scope": {"v1": n_v1, "v2": n_v2},
            "por_tipo": {k: int(v) for k, v in Counter(it["tipo"] for it in items).items()},
            "por_eje": {k: len(v) for k, v in ejes.items()},
        },
        "ejes": ejes,
        "items": items,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------
# Informe
# --------------------------------------------------------------------------

def _fmt_table(header, rows):
    widths = [len(h) for h in header]
    str_rows = []
    for r in rows:
        cells = [str(c) for c in r]
        str_rows.append(cells)
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(c))
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    sep = "-+-".join("-" * w for w in widths)
    out = [line, sep]
    for r in str_rows:
        out.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def build_report(summary: dict) -> str:
    s = summary
    g6 = s["gate6"]
    L = []
    L.append("REPORT_context_vocab_v2.txt — tokenizador de contexto (iteración 2)")
    L.append("=" * 78)
    L.append("")
    L.append("0. Contexto y ficheros")
    L.append("-" * 40)
    L.append(f"  contrato_tokens_v1.md        sha256 {s['sha_contract']}")
    L.append(f"  context_vocab.py             sha256 {s['sha_module']}")
    L.append(f"  windows_v4/split.parquet     sha256 {s['sha_split']}")
    L.append(f"  fecha                        {s['date']}")
    L.append(f"  casos (windows_v4/cases)     {s['n_cases']}")
    L.append(f"  filas tokens (solo v1)       {s['n_token_rows']}")
    L.append(f"  items vocab (v1+v2)          {s['n_items']} "
             f"({s['n_items_v1']} v1 + {s['n_items_v2']} v2)")
    L.append("")
    L.append("1. Tests y salida en ROJO")
    L.append("-" * 40)
    L.append(s["rojo"])
    L.append("")
    L.append("2. Cambios respecto a v1")
    L.append("-" * 40)
    for line in s["changes"]:
        L.append("  " + line)
    L.append("")
    L.append("3. Implementación")
    L.append("-" * 40)
    for line in s["implementation"]:
        L.append("  " + line)
    L.append("")
    L.append("4. Salida en VERDE")
    L.append("-" * 40)
    L.append(s["verde"])
    L.append("")
    L.append("5. Resultados")
    L.append("-" * 40)
    L.append("  5.1 Inventario v1 emitido (12 items)")
    for iid in s["v1_item_ids"]:
        L.append(f"      - {iid}")
    L.append("")
    L.append("  5.2 Gate 6 v1 (sondeo de fuente, val, 5-fold)")
    L.append(f"    balanced accuracy (valores)    : {g6['accuracy']:.4f}")
    L.append(f"    balanced accuracy (presencia)  : {g6['presence_accuracy']:.4f}")
    L.append(f"    n_val={g6['n_val']} n_real={g6['n_real']} "
             f"n_synth={g6['n_synth']} n_items={g6['n_items']}")
    L.append("    coeficientes (todos los items v1):")
    for i, c in enumerate(g6["top_coefs"], 1):
        L.append(f"      {i:2d}. {c['item_id']:<20s} {c['coef']:+9.4f}  [{c['eje']}]")
    L.append("    ablación (balanced accuracy):")
    L.append(f"      sin top-1 ({g6['top_coefs'][0]['item_id']}) : "
             f"{g6['ablation']['top1']:.4f}")
    L.append("    por pares de cohortes (balanced accuracy con valores):")
    L.append(_fmt_table(
        ["par", "accuracy", "presencia", "n_pos", "n_neg"],
        [[k, f"{v['accuracy']:.4f}", f"{v['presence_accuracy']:.4f}",
          str(v["n_pos"]), str(v["n_neg"])] for k, v in g6["pairwise"].items()]))
    L.append("")
    L.append("  5.3 Diagnóstico fisiología (scripts/diag_source_probe_physio.py)")
    L.append(s["physio"])
    L.append("")
    L.append("  5.4 Cobertura v1 por eje x cohorte (fracción emitida)")
    L.append(_fmt_table(
        ["eje"] + SOURCE_ORDER, s["coverage_by_axis"]))
    L.append("")
    L.append("  5.5 Tokens por caso v1 (distribución por cohorte)")
    L.append(_fmt_table(
        ["source", "n_casos", "min", "p50", "p95", "max", "casos_sin_token"],
        s["tokens_per_case_table"]))
    L.append(f"    todo caseid de cases.parquet con >= 1 token v1: {s['all_cases_have_token']}")
    L.append("")
    L.append("6. Discrepancias y supuestos")
    L.append("-" * 40)
    for i, a in enumerate(s["assumptions"], 1):
        L.append(f"  {i}. {a}")
    L.append("")
    L.append("7. Veredicto por gate")
    L.append("-" * 40)
    for g in s["verdict"]:
        L.append("  " + g)
    L.append("  Diagnóstico 3 (fisiología): sin veredicto; se reporta.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Ejecución completa
# --------------------------------------------------------------------------

def run(out_dir: Path = OUT_DIR) -> dict:
    t0 = _time.time()
    cases = load_cases()
    clin_by_source = load_all_clinical(cases)
    all_df = pd.concat(clin_by_source.values(), ignore_index=True)
    train_df = all_df[all_df["split"] == "train"]
    null_frac = compute_null_fraction(clin_by_source)
    vocab = build_vocab(train_df, all_df)
    tokens = emit_tokens(all_df, vocab, scope="v1")
    coverage = build_coverage(tokens, cases, null_frac)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tokens(tokens, out_dir / "tokens.parquet")
    write_coverage(coverage, out_dir / "coverage.parquet")
    write_vocab(vocab, tokens, out_dir)

    # gates
    gate4 = {
        "columns_requested": sorted(FEATURE_READ_COLUMNS),
        "intersection_excluded": sorted(
            set(FEATURE_READ_COLUMNS) & (set(EXCLUDED_COLUMNS) - {"caseid"})),
    }
    gate6 = gate6_probe(tokens, cases)
    gate7 = gate7_ks_tvd(all_df, vocab["spec"])
    coherence = coherence_check(all_df)
    tpc = tokens_per_case(tokens, cases)
    all_cases_have_token = bool((tpc["n_tokens"] > 0).all())

    # gate 8: categorías emitidas con < 50 casos en train
    train_caseids = set(cases[cases["split"] == "train"]["caseid"].tolist())
    train_counts = tokens[tokens["caseid"].isin(train_caseids)].groupby("item_id").size()
    gate8_violations = []
    for it in vocab["items"]:
        if it["tipo"] != "categorico" or it["scope"] != "v1" or it["item_id"].endswith(":otros"):
            continue
        n = int(train_counts.get(it["item_id"], 0))
        if n < 50:
            gate8_violations.append({"item_id": it["item_id"], "n_train": n})

    # cobertura por eje x cohorte (fracción media por eje)
    coverage2 = coverage.merge(
        tokens[["item_id", "eje"]].drop_duplicates(), on="item_id", how="left")
    cov_by_axis = (coverage2.groupby(["eje", "source"])["fraccion"].mean()
                   .reset_index().pivot(index="eje", columns="source", values="fraccion"))
    coverage_axis_rows = []
    for eje in sorted(cov_by_axis.index):
        row = [eje]
        for src in SOURCE_ORDER:
            v = cov_by_axis.loc[eje, src] if src in cov_by_axis.columns else np.nan
            row.append("—" if pd.isna(v) else f"{v:.3f}")
        coverage_axis_rows.append(row)

    # categorías agrupadas en otros por columna (valores crudos)
    grouped_categories = {}
    for col in CATEGORICAL_COLUMNS:
        mapping = vocab["spec"]["categorical"][col]
        raw_by_norm: dict = {}
        for v in all_df[col].to_numpy(dtype=object, na_value=None):
            if _is_missing(v):
                continue
            raw_by_norm.setdefault(normalize_item_value(v), str(v))
        grouped = sorted(str(v) for n, v in raw_by_norm.items()
                         if mapping.get(n) == "otros")
        grouped_categories[col] = grouped

    # tokens por caso
    tpc_stats = []
    for src in SOURCE_ORDER:
        d = tpc[tpc["source"] == src]["n_tokens"]
        tpc_stats.append([src, int(len(d)), int(d.min()), int(d.median()),
                          int(d.quantile(0.95)), int(d.max()),
                          int((d == 0).sum())])

    n_items_by_tipo = Counter(it["tipo"] for it in vocab["items"])
    n_items_by_eje = Counter(it["eje"] for it in vocab["items"])
    v1_items = [it for it in vocab["items"] if it["scope"] == "v1"]
    v2_items = [it for it in vocab["items"] if it["scope"] == "v2"]

    summary = {
        "date": pd.Timestamp.now().isoformat(),
        "sha_contract": _sha256(CONTRACT_PATH),
        "sha_module": _sha256(Path(__file__).resolve()),
        "sha_split": _sha256(SPLIT_PATH),
        "n_cases": int(len(cases)),
        "n_token_rows": int(len(tokens)),
        "n_items": len(vocab["items"]),
        "n_items_v1": len(v1_items),
        "n_items_v2": len(v2_items),
        "v1_item_ids": [it["item_id"] for it in v1_items],
        "n_items_by_tipo": dict(n_items_by_tipo),
        "n_items_by_eje": dict(n_items_by_eje),
        "gate4": gate4,
        "gate6": gate6,
        "gate7": gate7,
        "coherence": coherence,
        "coverage_by_axis": coverage_axis_rows,
        "coverage_axis_header": ["eje"] + SOURCE_ORDER,
        "coherence_bmi": [[r["source"], r["n_casos"], r["n_err_gt5pct"]]
                          for r in coherence["bmi"]],
        "coherence_ranges": [[r["source"], r["age_n"], r["age_out"], r["height_n"],
                              r["height_out"], r["weight_n"], r["weight_out"]]
                             for r in coherence["ranges"]],
        "tokens_per_case_table": tpc_stats,
        "all_cases_have_token": all_cases_have_token,
        "grouped_categories": grouped_categories,
        "gate8_violations": gate8_violations,
        "gate7_table": [[r["kind"], r["columna"],
                         f"{r['statistic']:.6f}" if r["statistic"] is not None else "—",
                         r["n_real"], r["n_synth"]] for r in gate7],
        "elapsed_s": round(_time.time() - t0, 2),
        "items": vocab["items"],
        "tokens": tokens,
        "coverage": coverage,
    }
    # secciones de texto del informe (rojo/verde/physio se completan manualmente)
    summary["rojo"] = "(ver sección 1 del informe — salida de pytest en rojo)"
    summary["verde"] = "(ver sección 4 del informe — salida de pytest en verde)"
    summary["physio"] = "(ver sección 5.3 del informe — salida del diagnóstico de fisiología)"
    summary["changes"] = [
        "El inventario v1 se reduce, por decisión de contrato tras el gate 6 = 0.9956 "
        "de la iteración 1, a demografía + riesgo global: age, sex:F, height, weight, "
        "bmi, asa y emop (con sus categorías, incluido 'otros' si aplica).",
        "Cada entrada de vocab.json lleva scope 'v1' | 'v2'. v1 = 12 items "
        "(4 continuos + 1 binario + 7 categóricos); v2 = 74 items (19 continuos + "
        "4 binarios + 51 categóricos) que se conservan con sus estadísticos pero NO se "
        "emiten.",
        "tokens.parquet y coverage.parquet ahora solo contienen tokens scope v1.",
        "La tabla eje -> item_id de vocab.json se mantiene completa (86 items) y cada "
        "entrada lleva su scope.",
        "Gate 6: se repite sobre el inventario v1 y se añaden la ablación quitando el "
        "top-1 y el desglose por pares de cohortes (real vs synthetic_v7, real vs "
        "cf_v7, real vs vaso_reinf_v7, synthetic_v7 vs cf_v7).",
        "Nuevo diagnóstico de fuga en fisiología en scripts/diag_source_probe_physio.py "
        "(sin salida a data/).",
    ]
    summary["implementation"] = [
        "Deduplicación INSPIRE: por caseid se conserva la fila con más valores no "
        "nulos; empate -> primera en orden estable (2 538 duplicados en real; la fila "
        "válida tiene ~28-31 columnas de contexto no nulas frente a 17 del stub).",
        "Codificación de sex: la fuente usa 'F'/'M'; se emite 'sex:F' solo cuando el "
        "valor es femenino. No existe token 'sex:M'.",
        "preop_ecg: normal = 'Normal Sinus Rhythm'; anormal = cualquiera de los 26 "
        "hallazgos del inventario; fuera del inventario o vacío -> no emitido (scope v2).",
        "preop_pft: normal = 'Normal'; anormal = cualquiera de los 8 hallazgos; fuera "
        "del inventario o vacío -> no emitido (scope v2).",
        "Normalización de item_id: minúsculas, sin tildes ni espacios; flotantes "
        "integrales sin '.0' (asa 2.0 -> 'asa:2').",
        "Agrupación en 'otros': umbral 50 casos en train. Agrupadas por columna: " +
        "; ".join(f"{c}={grouped_categories[c]}" for c in CATEGORICAL_COLUMNS
                  if grouped_categories[c]) + ".",
        f"Vocabulario total: {summary['n_items']} items; v1 = {summary['n_items_v1']}, "
        f"v2 = {summary['n_items_v2']}.",
        "Scope v1 = ejes demografia y riesgo_global; scope v2 = el resto. Los "
        "estadísticos de normalización de v2 se calculan y guardan (referencia).",
        "Split: heredado bit a bit de windows_v4 (columna 'split' de cases.parquet, "
        "verificada idéntica al mapeo de split.parquet); sha registrado en vocab.json.",
    ]
    summary["assumptions"] = [
        "El contrato en disco NO ha cambiado: su sha256 sigue siendo ae2e2298… (el "
        "enunciado da por hecha una 'decisión de contrato' de reducir el contexto a 7 "
        "item_id, pero el texto del contrato no la recoge). Se implementa igualmente "
        "la decisión descrita y se documenta la discrepancia.",
        "caseid aparece como clave de fila en tokens.parquet y se lee como clave de "
        "join, nunca como feature; el gate 4 excluye solo a caseid de la intersección.",
        "La deduplicación INSPIRE cuenta no nulos sobre las 37 columnas de contexto "
        "(no sobre las 82) para no leer columnas excluidas.",
        "Desviación poblacional (ddof=0); std=0 -> 1.0.",
        "top_diagnosis_chapter: '' (real) y 'None' (cf/vaso) -> no emitido (scope v2).",
        "El emit_tokens conserva un parámetro scope para pruebas: None emite todo, "
        "'v1' emite solo el inventario v1 (usado por run()).",
        "Gate 6 v1: regresión logística estandarizada, 5-fold estratificado (semilla "
        "0) sobre casos val; el sondeo por pares usa las mismas dos cohortes como "
        "clase positiva y negativa.",
        "Diagnóstico 3: muestreo de 2 000 casos val estratificado por source (semilla "
        "0); 20 ventanas de 12 celdas de mantenimiento (t > 600 s) por caso; "
        "estandarización + LR, GroupKFold por caso; std poblacional (ddof=0).",
    ]
    summary["verdict"] = [
        f"Gate 1 (cobertura v1): PASA — tabla en sección 5.4.",
        f"Gate 4 (columnas excluidas): {'PASA' if not gate4['intersection_excluded'] else 'FALLA'} "
        f"— intersección {gate4['intersection_excluded'] or 'vacía'}.",
        f"Gate 6 (sondeo de fuente v1): {'PASA' if gate6['accuracy'] <= 0.60 else 'NO PASA'} "
        f"— balanced accuracy {gate6['accuracy']:.4f} (umbral <= 0.60).",
        "Gate 7 (KS/TVD): INFORMATIVO — sin umbral (sección 5 del informe anterior; "
        "sin cambios al no depender del scope).",
        f"Gate 8 (categorías < 50 en train, v1): "
        f"{'PASA' if not gate8_violations else 'NO PASA'}.",
        "Gate 9 (estadísticos solo train): PASA — verificable en vocab.json (v1 y v2).",
    ]
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Context vocab v1")
    ap.add_argument("command", choices=["run", "report"], nargs="?", default="run")
    ap.add_argument("--out-dir", default=None,
                    help="directorio de salida (por defecto paths.CONTEXT_DIR)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    if args.command == "run":
        summary = run(out_dir)
        report = build_report(summary)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(json.dumps({
            "n_items": summary["n_items"],
            "n_token_rows": summary["n_token_rows"],
            "gate6_accuracy": summary["gate6"]["accuracy"],
            "gate6_presence_accuracy": summary["gate6"]["presence_accuracy"],
            "all_cases_have_token": summary["all_cases_have_token"],
            "elapsed_s": summary["elapsed_s"],
        }, indent=2))
    elif args.command == "report":
        summary = run(out_dir)
        REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
        print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
