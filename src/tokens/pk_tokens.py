"""pk_tokens.py — módulo PK del contrato de tokens v1.

Implementa la sección "Modelos PK por fármaco" del contrato de tokens v1:
recalcula, para cada fármaco de la v1, la concentración efectiva (Ce), la dosis
acumulada, el bolo y las máscaras sobre la rejilla de 5 s de ``data/windows_v2/``.

Modelos (ecuaciones citadas en el informe):
  propofol      Schnider 3 compartimentos + Ce (ke0 = 0.456 min⁻¹)
  remifentanilo Minto 3 compartimentos + Ce (ke0 = 0.6 min⁻¹)
  sevoflurano   passthrough de Primus/MAC (sin PK)
  fenilefrina   bolo exponencial ke0 = ln2/300 s⁻¹ + infusión hacia R/ke0
  noradrenalina ganancia directa: Ce = tasa µg/kg/min
  efedrina      bolo exponencial ke0 = ln2/600 s⁻¹ (sin taquifilaxia)
  rocuronio     Wierda 2 compartimentos + Ce (V1 = 0.07 L/kg, Cl = 3.7 mL/kg/min,
                ke0 = 0.105 min⁻¹; V2/k12/k21 supuestos del contrato)

El bolo se aplica al inicio de la celda de 5 s en la que cae (misma celda que
reporta la columna *_bolus de windows_v2). Integración numérica: RK4 de orden 4
en forma cerrada (sistema lineal) con paso de 5 s; para fenilefrina/efedrina se
usa la solución analítica exacta (recursiva) de la exponencial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time as _time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from anessim.pk.propofol import PropofolSchnider
from anessim.pk.remifentanil import RemifentanilMinto

import paths

ROOT = Path(__file__).resolve().parents[2]
WINDOWS_DIR = paths.WINDOWS_DIR / "windows"
OUT_DIR = paths.PK_DIR / "windows"
CONTRACT_PATH = ROOT / "contrato_tokens_v1.md"
PART_SIZE = 200_000
DT_S = 5.0
DT_MIN = DT_S / 60.0
LOG2 = np.log(2.0)
MAX_AGE_DRUG_S = 30.0  # del registry de windows_v2 (solo para m_ce)
ROC_CONC_MG_PER_ML = 10.0  # concentración estándar de rocuronio (supuesto documentado)

DRUGS = [
    "propofol",
    "remifentanilo",
    "sevoflurano",
    "fenilefrina",
    "noradrenalina",
    "efedrina",
    "rocuronio",
]

RATE_COLS = {
    "propofol": "Orchestra/PPF20_RATE",
    "remifentanilo": "Orchestra/RFTN20_RATE",
    "sevoflurano": "Primus/MAC",
    "fenilefrina": "Orchestra/PHEN_RATE",
    "noradrenalina": "Orchestra/NEPI_RATE",
    "efedrina": "Orchestra/EPH_RATE",
    "rocuronio": "Orchestra/ROC_RATE",
}

BOLUS_COLS = {
    "propofol": "ppf_bolus",
    "remifentanilo": "remi_bolus",
    "fenilefrina": "phen_bolus",
    "efedrina": "eph_bolus",
    "rocuronio": "roc_bolus",
}

SOURCES = paths.COHORTS


# --------------------------------------------------------------------------
# Conversión de unidades y parámetros
# --------------------------------------------------------------------------

def rate_to_canonical_per_min(drug: str, raw: float) -> float:
    """Convierte la tasa cruda (unidad del generador) a la unidad canónica/min.

    propofol mL/h -> mg/min (/3); remi mL/h -> ug/min (/3); fenilefrina ug/min;
    noradrenalina ug/kg/min; efedrina mg/min; rocuronio mL/h (10 mg/mL) -> mg/min;
    sevoflurano MAC (sin conversión).
    """
    if drug in ("propofol", "remifentanilo"):
        return raw / 3.0
    if drug == "rocuronio":
        return raw * ROC_CONC_MG_PER_ML / 60.0
    return raw


def schnider_params(weight: float, age: float, height: float, sex: str) -> dict:
    m = PropofolSchnider(weight_kg=weight, age_y=age, height_cm=height, sex=sex)
    return dict(v1=m.v1, v2=m.v2, v3=m.v3, k10=m.k10, k12=m.k12, k13=m.k13,
                k21=m.k21, k31=m.k31, ke0=m.ke0)


def minto_params(weight: float, height: float, age: float, sex: str) -> dict:
    m = RemifentanilMinto(weight_kg=weight, height_cm=height, age_y=age, sex=sex)
    return dict(v1=m.v1, v2=m.v2, v3=m.v3, k10=m.k10, k12=m.k12, k13=m.k13,
                k21=m.k21, k31=m.k31, ke0=m.ke0)


def wierda_params(weight: float) -> dict:
    """Wierda 2 compartimentos + Ce (supuestos documentados del contrato).

    V1 = 0.07 L/kg (contrato), Cl = 3.7 mL/kg/min (contrato) -> k10 = Cl/V1.
    V2 = 0.13 L/kg, k12 = 0.093 min⁻¹ (Wierda publicado): SUPUESTOS.
    k21 = k12·V1/V2. ke0 = 0.105 min⁻¹ (contrato).
    """
    v1 = 0.07 * weight
    v2 = 0.13 * weight
    cl = 0.0037 * weight  # L/min
    k10 = cl / v1
    k12 = 0.093
    k21 = k12 * v1 / v2
    return dict(v1=v1, v2=v2, v3=1.0, k10=k10, k12=k12, k13=0.0,
                k21=k21, k31=0.0, ke0=0.105)


# --------------------------------------------------------------------------
# Núcleo PK (puro, testeable sin disco)
# --------------------------------------------------------------------------

def build_linear_ce(k10, k12, k13, k21, k31, ke0, v1, v2, v3,
                    rate_per_min, bolus, dt_s: float = DT_S) -> np.ndarray:
    """RK4 de orden 4 (forma cerrada de un sistema lineal) para [c1,c2,c3,ce].

    rate_per_min: tasa de infusión en la unidad de dosis/minuto (forward-filled,
    NaN tratado como 0). bolus: dosis por celda, aplicada al inicio de la celda.
    Devuelve estados (n, 4).
    """
    rate = np.nan_to_num(np.asarray(rate_per_min, dtype=float), nan=0.0)
    bol = np.nan_to_num(np.asarray(bolus, dtype=float), nan=0.0)
    n = len(rate)
    h = dt_s / 60.0
    A = np.zeros((4, 4))
    A[0, 0] = -(k10 + k12 + k13)
    A[0, 1] = k21 * (v2 / v1)
    A[0, 2] = k31 * (v3 / v1)
    A[1, 0] = k12 * (v1 / v2)
    A[1, 1] = -k21
    A[2, 0] = k13 * (v1 / v3)
    A[2, 2] = -k31
    A[3, 0] = ke0
    A[3, 3] = -ke0
    hA = h * A
    hA2 = hA @ hA
    hA3 = hA2 @ hA
    hA4 = hA3 @ hA
    phi = np.eye(4) + hA + hA2 / 2.0 + hA3 / 6.0 + hA4 / 24.0
    psi0 = (h * (np.eye(4) + hA / 2.0 + hA2 / 6.0 + hA3 / 24.0))[:, 0]
    # bolo de la celda i aplicado al inicio (índice i-1); celda 0 en t[0]
    bolus_at = np.zeros(n)
    if n > 1:
        bolus_at[:-1] += bol[1:]
    bolus_at[0] += bol[0]
    states = np.zeros((n, 4))
    y = np.zeros(4)
    for i in range(n):
        y[0] += bolus_at[i] / v1
        states[i] = y.copy()
        if i < n - 1:
            y = phi @ y + psi0 * (rate[i + 1] / v1)
    return states


def exponential_ce(ke0_s: float, bolus, rate_per_min, dt_s: float = DT_S) -> np.ndarray:
    """Ce de un compartimento con eliminación ke0 (1/s) — solución exacta recursiva.

    Ce(t) = Σ bolos·exp(-ke0·t) + infusión hacia el estado estacionario R/ke0.
    La infusión converge al estado estacionario (contrato: "estado estacionario").
    """
    bol = np.nan_to_num(np.asarray(bolus, dtype=float), nan=0.0)
    rate = np.nan_to_num(np.asarray(rate_per_min, dtype=float), nan=0.0)
    n = len(bol)
    alpha = float(np.exp(-ke0_s * dt_s))
    ss = 1.0 / ke0_s  # R_s/ke0_s con R_s = rate/60 (unidades de dosis)
    bolus_at = np.zeros(n)
    if n > 1:
        bolus_at[:-1] += bol[1:]
    bolus_at[0] += bol[0]
    ce = np.zeros(n)
    y = 0.0
    for i in range(n):
        y += bolus_at[i]
        ce[i] = y
        if i < n - 1:
            y = y * alpha + (rate[i + 1] / 60.0) * ss * (1.0 - alpha)
    return ce


def compute_ce(drug: str, raw_rate_grid, bolus_grid, demo: dict,
               dt_s: float = DT_S) -> np.ndarray:
    """Ce en unidad canónica para un fármaco (vector, rejilla)."""
    raw = np.nan_to_num(np.asarray(raw_rate_grid, dtype=float), nan=0.0)
    bol = np.nan_to_num(np.asarray(bolus_grid, dtype=float), nan=0.0)
    if drug == "propofol":
        p = schnider_params(float(demo["weight"]), float(demo["age"]),
                            float(demo["height"]), str(demo["sex"]))
        st = build_linear_ce(p["k10"], p["k12"], p["k13"], p["k21"], p["k31"],
                             p["ke0"], p["v1"], p["v2"], p["v3"], raw / 3.0, bol, dt_s)
        return st[:, 3].astype(np.float32)
    if drug == "remifentanilo":
        p = minto_params(float(demo["weight"]), float(demo["height"]),
                         float(demo["age"]), str(demo["sex"]))
        st = build_linear_ce(p["k10"], p["k12"], p["k13"], p["k21"], p["k31"],
                             p["ke0"], p["v1"], p["v2"], p["v3"], raw / 3.0, bol, dt_s)
        return st[:, 3].astype(np.float32)
    if drug == "sevoflurano":
        return raw.astype(np.float32)
    if drug == "fenilefrina":
        return exponential_ce(LOG2 / 300.0, bol, raw, dt_s).astype(np.float32)
    if drug == "noradrenalina":
        return raw.astype(np.float32)
    if drug == "efedrina":
        return exponential_ce(LOG2 / 600.0, bol, np.zeros_like(raw), dt_s).astype(np.float32)
    if drug == "rocuronio":
        p = wierda_params(float(demo["weight"]))
        rate_min = raw * ROC_CONC_MG_PER_ML / 60.0  # mL/h -> mg/min
        st = build_linear_ce(p["k10"], p["k12"], p["k13"], p["k21"], p["k31"],
                             p["ke0"], p["v1"], p["v2"], p["v3"], rate_min, bol, dt_s)
        return st[:, 3].astype(np.float32)
    raise ValueError(drug)


def compute_dose_cum(drug: str, bolus, rate_per_min, dt_s: float = DT_S) -> np.ndarray:
    """Dosis acumulada desde el inicio de la rejilla (bolos + infusión).

    Unidad canónica de dosis: propofol mg, remi µg, fenilefrina µg, efedrina mg,
    rocuronio mg, noradrenalina µg/kg; sevoflurano 0.
    """
    bol = np.nan_to_num(np.asarray(bolus, dtype=float), nan=0.0)
    rate = np.nan_to_num(np.asarray(rate_per_min, dtype=float), nan=0.0)
    n = len(bol)
    dc = np.zeros(n, dtype=float)
    acc_b = 0.0
    acc_i = 0.0
    for i in range(n):
        acc_b += bol[i]
        if i > 0:
            acc_i += rate[i] * DT_MIN
        dc[i] = acc_b + acc_i
    return dc


def ce_mask(cohort_has_col: bool, case_has_col: bool, finite_grid) -> np.ndarray:
    """Máscara de 3 estados: 0 observado, 1 ausente en cohorte, 2 ausente en caso."""
    n = len(finite_grid)
    if not cohort_has_col:
        return np.ones(n, dtype=np.uint8)
    if not case_has_col:
        return np.full(n, 2, dtype=np.uint8)
    return np.where(np.asarray(finite_grid, dtype=bool), 0, 2).astype(np.uint8)


# --------------------------------------------------------------------------
# Utilidades de rejilla (misma convención que window.py)
# --------------------------------------------------------------------------

def _forward_fill(t: np.ndarray, v: np.ndarray, grid: np.ndarray, max_age: float) -> np.ndarray:
    """Forward-fill con edad máxima; más allá de max_age el valor queda NaN."""
    vf = v.astype(np.float64)
    nonnull = np.isfinite(vf)
    out = np.full(len(grid), np.nan, dtype=np.float64)
    if not nonnull.any():
        return out
    tt = t[nonnull]
    vv = vf[nonnull]
    idx = np.searchsorted(tt, grid, side="right") - 1
    valid = idx >= 0
    idx = np.clip(idx, 0, len(vv) - 1)
    vals = np.full(len(grid), np.nan)
    vals[valid] = vv[idx[valid]]
    age = grid - tt[idx]
    vals[valid & (age > max_age)] = np.nan
    return vals


def _forward_fill_hold(t: np.ndarray, v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Forward-fill SIN tope de edad: mantiene el último valor válido (un hueco
    en el registro no es una parada de bomba). Antes del primer valor: NaN."""
    vf = v.astype(np.float64)
    nonnull = np.isfinite(vf)
    out = np.full(len(grid), np.nan, dtype=np.float64)
    if not nonnull.any():
        return out
    tt = t[nonnull]
    vv = vf[nonnull]
    idx = np.searchsorted(tt, grid, side="right") - 1
    valid = idx >= 0
    idx = np.clip(idx, 0, len(vv) - 1)
    out[valid] = vv[idx[valid]]
    return out


def _bolus_on_grid(t: np.ndarray, v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Suma de bolo en (t-grid_s, t] por celda (misma convención que window.py B1)."""
    out = np.zeros(len(grid), dtype=np.float64)
    spk = np.isfinite(v) & (v > 0.0)
    if not spk.any():
        return out
    tb = t[spk]
    vv = v[spk]
    idx = np.searchsorted(grid, tb, side="left")
    ok = (idx >= 0) & (idx < len(grid))
    np.add.at(out, idx[ok], vv[ok])
    return out


# --------------------------------------------------------------------------
# Lectura de clínica, registro y metadatos CF
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def load_registry() -> dict[str, set[str]]:
    """Devuelve {source: {tracks presentes en esa cohorte}} desde registry.parquet."""
    reg = pq.read_table(paths.WINDOWS_DIR / "registry.parquet").to_pandas()
    out: dict[str, set[str]] = {}
    for _, r in reg.iterrows():
        for src in str(r["sources"]).split(","):
            src = src.strip()
            if src:
                out.setdefault(src, set()).add(str(r["track"]))
    return out


def _demo_from_row(r: pd.Series) -> dict:
    def _f(x, default):
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default
    return dict(
        age=_f(r.get("age"), 40.0),
        sex=str(r.get("sex", "M")) if str(r.get("sex", "M")) in ("M", "F") else "M",
        height=_f(r.get("height"), 170.0),
        weight=_f(r.get("weight"), 70.0),
    )


@lru_cache(maxsize=1)
def load_clinical_map() -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    # real (deduplicación: fila con más columnas no nulas)
    real = pq.read_table(SOURCES["real"] / "clinical_data_enriched.parquet").to_pandas()
    real["_nn"] = real.notna().sum(axis=1)
    real = real.sort_values("_nn", ascending=False).drop_duplicates("caseid", keep="first")
    out["real"] = {int(r.caseid): _demo_from_row(r) for _, r in real.iterrows()}
    # synthetic_v7
    syn = pq.read_table(SOURCES["synthetic_v7"] / "clinical_data.parquet").to_pandas()
    out["synthetic_v7"] = {int(r.caseid): _demo_from_row(r) for _, r in syn.iterrows()}
    # vaso_reinf_v7 y cf_v7: fila clínica por caso
    for src in ("vaso_reinf_v7", "cf_v7"):
        base = SOURCES[src]
        m: dict[int, dict] = {}
        for p in sorted((base / "clinical").glob("*_clinical.parquet")):
            cid = int(p.stem.split("_")[0])
            t = pq.read_table(p).to_pandas()
            if len(t):
                m[cid] = _demo_from_row(t.iloc[0])
        out[src] = m
    return out

@lru_cache(maxsize=1)

def load_cf_pairs() -> list[dict]:
    base = SOURCES["cf_v7"] / "metadata"
    pairs = []
    for p in sorted(base.glob("cf_pair_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        pairs.append(dict(
            caseid_a=int(d["caseid_a"]),
            caseid_b=int(d["caseid_b"]),
            split_t=float(d.get("split_t", d.get("split_t_s", np.nan))),
            lever=d.get("lever"),
        ))
    return pairs


# --------------------------------------------------------------------------
# Cómputo por caso
# --------------------------------------------------------------------------

CASE_COLS = [
    "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "Orchestra/PHEN_RATE",
    "Orchestra/NEPI_RATE", "Orchestra/ROC_RATE", "Primus/MAC",
]


def _read_case_columns(caseid: int, source: str) -> pd.DataFrame | None:
    base = SOURCES[source]
    if source == "real":
        path = base / "cases" / f"{caseid:04d}.parquet"
    else:
        path = base / "cases" / f"{caseid}.parquet"
    if not path.exists():
        return None
    schema = pq.read_schema(path)
    cols = [c for c in (["time"] + CASE_COLS) if c in schema.names]
    return pq.read_table(path, columns=cols).to_pandas()


def _compute_case_tokens(windows: pd.DataFrame, case_cols: pd.DataFrame,
                         demo: dict, source: str, cohort_cols: set[str]) -> pd.DataFrame:
    """Calcula los tokens PK de un caso. windows: filas (caseid, t, bolos, m_*_bolus)."""
    grid = windows["t"].to_numpy(np.float64)
    n = len(grid)
    raw_t = case_cols["time"].to_numpy(np.float64)

    # bolos desde las columnas de ventana (NaN -> 0); en real todas ausentes
    bolus_raw: dict[str, np.ndarray] = {}
    bolus_has_col: dict[str, bool] = {}
    for drug, col in BOLUS_COLS.items():
        mcol = f"m_{col}"
        has = bool(windows[mcol].iloc[0] == 1) if len(windows) else False
        bolus_has_col[drug] = has
        if col in windows.columns:
            bolus_raw[drug] = np.nan_to_num(windows[col].to_numpy(np.float64), nan=0.0)
        else:
            bolus_raw[drug] = np.zeros(n)

    # tasas en rejilla (hold sin tope para Ce/dose_cum; max_age solo para m_ce)
    rate_hold: dict[str, np.ndarray] = {}
    rate_mask: dict[str, np.ndarray] = {}
    rate_case_has: dict[str, bool] = {}
    for drug, col in RATE_COLS.items():
        if col in case_cols.columns:
            raw_track = case_cols[col].to_numpy(np.float64)
            rate_hold[drug] = _forward_fill_hold(raw_t, raw_track, grid)
            rate_mask[drug] = _forward_fill(raw_t, raw_track, grid, MAX_AGE_DRUG_S)
            rate_case_has[drug] = True
        else:
            rate_hold[drug] = np.full(n, np.nan)
            rate_mask[drug] = np.full(n, np.nan)
            rate_case_has[drug] = False

    out_cols: dict[str, np.ndarray] = {}
    for drug in DRUGS:
        bol = bolus_raw.get(drug, np.zeros(n))
        # ---- m_ce (fuente de dosis por fármaco y cohorte) ----
        if drug == "efedrina":
            # dosis fuente = eph_bolus (solo bolo); en real no existe -> m_ce=1
            m_ce = ce_mask("eph_bolus" in cohort_cols, bolus_has_col.get(drug, False),
                           np.ones(n, dtype=bool))
        elif drug == "rocuronio" and source != "real":
            # dosis fuente = roc_bolus (solo bolo en sintéticos)
            m_ce = ce_mask("roc_bolus" in cohort_cols, bolus_has_col.get(drug, False),
                           np.ones(n, dtype=bool))
        else:
            # dosis fuente = columna RATE (rocuronio real desde ROC_RATE)
            src_col = RATE_COLS[drug]
            m_ce = ce_mask(src_col in cohort_cols, rate_case_has.get(drug, False),
                           np.isfinite(rate_mask.get(drug, np.full(n, np.nan))))
        out_cols[f"m_ce_{drug}"] = m_ce

        # ---- Ce ----
        if drug == "rocuronio" and source == "real":
            ce = compute_ce(drug, rate_hold["rocuronio"], np.zeros(n), demo, DT_S)
        elif drug == "rocuronio":
            ce = compute_ce(drug, np.zeros(n), bol, demo, DT_S)
        else:
            ce = compute_ce(drug, rate_hold[drug], bol, demo, DT_S)
        out_cols[f"ce_{drug}"] = ce.astype(np.float32)

        # ---- dose_cum ----
        if drug == "sevoflurano":
            dc = np.zeros(n, dtype=np.float32)
        elif drug == "rocuronio" and source == "real":
            # integral de ROC_RATE (mL/h -> mg/min)
            rate_min = np.nan_to_num(rate_hold["rocuronio"], nan=0.0) * ROC_CONC_MG_PER_ML / 60.0
            dc = compute_dose_cum(drug, np.zeros(n), rate_min, DT_S)
        elif drug in ("efedrina", "rocuronio"):
            dc = compute_dose_cum(drug, bol, np.zeros(n), DT_S)
        elif drug == "noradrenalina":
            dc = compute_dose_cum(drug, np.zeros(n), np.nan_to_num(rate_hold[drug], nan=0.0), DT_S)
        else:
            rate_min = np.nan_to_num(rate_hold[drug], nan=0.0) / (3.0 if drug in ("propofol", "remifentanilo") else 1.0)
            dc = compute_dose_cum(drug, bol, rate_min, DT_S)
        out_cols[f"dose_cum_{drug}"] = dc.astype(np.float32)

        # ---- bolus y bolus_obs (0 en real para todos los fármacos) ----
        out_cols[f"bolus_{drug}"] = bol.astype(np.float32)
        obs = 1 if bolus_has_col.get(drug, False) else 0
        out_cols[f"bolus_obs_{drug}"] = np.full(n, obs, dtype=np.uint8)

    data = {
        "caseid": windows["caseid"].to_numpy(np.int32),
        "t": windows["t"].to_numpy(np.int32),
        "source": windows["source"].astype(str).to_numpy(),
        "split": windows["split"].astype(str).to_numpy(),
    }
    data.update(out_cols)
    return pd.DataFrame(data)

def _case_partition_path(caseid: int, source: str, split: str) -> Path:
    """Localiza la partición de un caso por suma acumulada de n_windows (misma
    lógica de flush de window.py: partición nueva cada 200 000 filas)."""
    cases_df = pq.read_table(paths.WINDOWS_DIR / "cases.parquet").to_pandas()
    grp = cases_df[(cases_df.source == source) & (cases_df.split == split)].sort_values("caseid")
    pos = int((grp.caseid == caseid).argmax())
    rows_before = int(grp.n_windows.iloc[:pos].sum())
    part_idx = rows_before // PART_SIZE
    return WINDOWS_DIR / f"source={source}" / f"split={split}" / f"part-{part_idx:05d}.parquet"


def process_case_tokens(caseid: int, source: str) -> pd.DataFrame:
    """Calcula los tokens de un caso leyendo su partición y su parquet crudo."""
    cases_df = pq.read_table(paths.WINDOWS_DIR / "cases.parquet").to_pandas()
    row = cases_df[cases_df.caseid == caseid]
    if len(row) == 0:
        raise KeyError(caseid)
    row = row.iloc[0]
    split = str(row["split"])
    part = _case_partition_path(int(caseid), source, split)
    tbl = pq.read_table(part).to_pandas()
    windows = tbl[tbl.caseid == caseid]
    if len(windows) == 0:
        raise KeyError(caseid)
    case_cols = _read_case_columns(int(caseid), source)
    demo = load_clinical_map()[source][int(caseid)]
    cohort_cols = load_registry().get(source, set())
    return _compute_case_tokens(windows, case_cols, demo, source, cohort_cols)


# --------------------------------------------------------------------------
# Particiones
# --------------------------------------------------------------------------

OUTPUT_COLUMNS = (["caseid", "t", "source", "split"]
                  + [f"{p}_{d}" for d in DRUGS for p in ("ce", "dose_cum", "bolus", "bolus_obs", "m_ce")])


def process_partition(part_path: Path, out_path: Path, max_cases: int | None = None) -> dict:
    """Procesa una partición de windows_v2 y escribe su parquet pk_v1."""
    tbl = pq.read_table(part_path).to_pandas()
    source = str(tbl["source"].iloc[0])
    demo_map = load_clinical_map()[source]
    cohort_cols = load_registry().get(source, set())

    caseids = sorted(tbl["caseid"].unique().tolist())
    if max_cases is not None:
        caseids = caseids[:max_cases]

    frames = []
    for cid in caseids:
        sub = tbl[tbl.caseid == cid]
        case_cols = _read_case_columns(int(cid), source)
        if case_cols is None or len(case_cols) == 0:
            raise RuntimeError(f"no se pudo leer el parquet de {source}/{cid}")
        demo = demo_map.get(int(cid), dict(age=40.0, sex="M", height=170.0, weight=70.0))
        frames.append(_compute_case_tokens(sub, case_cols, demo, source, cohort_cols))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = out.sort_values(["caseid", "t"], kind="stable").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out["source"] = out["source"].astype("category")
    out["split"] = out["split"].astype("category")
    schema = pa.schema([
        ("caseid", pa.int32()), ("t", pa.int32()),
        ("source", pa.dictionary(pa.int32(), pa.string())),
        ("split", pa.dictionary(pa.int32(), pa.string())),
    ] + [(f"{p}_{d}", pa.float32() if p in ("ce", "dose_cum", "bolus") else pa.uint8())
         for d in DRUGS for p in ("ce", "dose_cum", "bolus", "bolus_obs", "m_ce")])
    table = pa.Table.from_pandas(out, schema=schema, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")
    return dict(n_rows=len(out), n_cases=len(caseids))


def run_all(out_dir: Path = OUT_DIR) -> dict:
    """Ejecuta sobre todo data/windows_v4 y escribe a out_dir. Devuelve resumen."""
    t0 = _time.time()
    parts = sorted(WINDOWS_DIR.glob("source=*/split=*/part-*.parquet"))
    counts: dict[str, int] = {}
    total = 0
    for i, part in enumerate(parts):
        rel = part.relative_to(WINDOWS_DIR)
        out_path = out_dir / rel
        res = process_partition(part, out_path)
        key = str(rel.parent).replace("\\", "/")
        counts[key] = counts.get(key, 0) + res["n_rows"]
        total += res["n_rows"]
        if (i + 1) % 50 == 0:
            print(f"[pk] {i + 1}/{len(parts)} particiones, {total:,} filas", flush=True)
    elapsed = _time.time() - t0
    return dict(parts=len(parts), n_rows=total, counts=counts, elapsed_s=elapsed)


def build_manifest(summary: dict, assumptions: list[str]) -> dict:
    manifest = {
        "date": pd.Timestamp.now().isoformat(),
        "pk_tokens_py_sha256": _sha256(ROOT / "src" / "tokens" / "pk_tokens.py"),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "windows_manifest_path": str(paths.WINDOWS_DIR / "manifest.json"),
        "sha256_windows_manifest": _sha256(paths.WINDOWS_DIR / "manifest.json"),
        "parameters": {
            "propofol": {"model": "Schnider 3 comp + Ce", "ke0_per_min": 0.456},
            "remifentanilo": {"model": "Minto 3 comp + Ce", "ke0_per_min": 0.6},
            "sevoflurano": {"model": "passthrough MAC"},
            "fenilefrina": {"model": "bolo exponencial + infusión", "ke0_per_s": LOG2 / 300.0},
            "efedrina": {"model": "bolo exponencial", "ke0_per_s": LOG2 / 600.0},
            "noradrenalina": {"model": "ganancia directa"},
            "rocuronio": {"model": "Wierda 2 comp + Ce", "V1_L_per_kg": 0.07,
                          "Cl_mL_per_kg_per_min": 3.7, "ke0_per_min": 0.105,
                          "V2_L_per_kg": 0.13, "k12_per_min": 0.093,
                          "k21_per_min": "k12*V1/V2"},
        },
        "bolus_source_by_cohort": {
            "propofol": "PPF20_RATE + ppf_bolus (sint); en real solo RATE (bolus=0)",
            "remifentanilo": "RFTN20_RATE + remi_bolus (sint); en real solo RATE (bolus=0)",
            "sevoflurano": "Primus/MAC (sin bolo)",
            "fenilefrina": "PHEN_RATE + phen_bolus (sint); en real solo RATE (bolus=0)",
            "noradrenalina": "NEPI_RATE (sin bolo)",
            "efedrina": "eph_bolus (solo bolo); inobservable en real (m_ce=1)",
            "rocuronio": "roc_bolus (sint); ROC_RATE como perfusión (real)",
        },
        "iiv_note": {
            "aplica_a": ["propofol", "remifentanilo"],
            "sigma_k": 0.20,
            "sigma_v": 0.15,
            "sigma_ke0": 0.15,
            "registrada_en_metadata": False,
            "consecuencia": "la Ce poblacional del token se desvía ~20-25 % de la Ce individual del generador (aceptado como ruido realista)",
        },
        "n_rows": summary["n_rows"],
        "n_rows_by_source_split": summary["counts"],
        "n_partitions": summary["parts"],
        "elapsed_s": round(summary["elapsed_s"], 2),
        "assumptions": assumptions,
    }
    return manifest


ASSUMPTIONS = [
    "Wierda: V1=0.07 L/kg, Cl=3.7 mL/kg/min, ke0=0.105 min-1, V2=0.13 L/kg, "
    "k12=0.093 min-1 y k21=k12*V1/V2 son del contrato v2 (Wierda publicado).",
    "Fenilefrina: la infusión se integra con la misma ODE de un compartimento del bolo, "
    "convergiendo al estado estacionario R/ke0 (ke0=ln2/300 s-1); el generador usa la tasa "
    "directamente (sin dinámica).",
    "dose_cum se acumula desde el inicio de la rejilla (anestart=0 en sintéticos; en real "
    "la dosis anterior al inicio de la rejilla no se contabiliza).",
    "Forward-fill de tasas SIN tope de edad para Ce y dose_cum (un hueco en el registro no "
    "es una parada de bomba); el max_age=30 s se usa SOLO para m_ce.",
    "m_ce_<d> usa la columna fuente de dosis por fármaco y cohorte: RATE (o Primus/MAC) "
    "para la mayoría; eph_bolus/roc_bolus para efedrina y rocuronio en sintéticos; "
    "ROC_RATE para rocuronio en real.",
    "dose_cum_sevoflurano = 0 (sin PK); dose_cum_noradrenalina en ug/kg (integral de ug/kg/min).",
    "Rocuronio en real: ROC_RATE está en mL/h; la conversión a mg/min usa la concentración "
    "estándar de 10 mg/mL (SUPUESTO documentado; no está en el contrato v2).",
    "En real no se infieren bolos desde VOL (contrato v2): bolus_*=0 y bolus_obs_*=0 para "
    "todos los fármacos; los bolos por bomba quedan en RATE como picos de tasa.",
    "El bolo se aplica al inicio de la celda de 5 s en la que cae (la columna *_bolus de "
    "windows_v2 suma en (t-5, t]).",
    "Integración: RK4 de orden 4 en forma cerrada (sistema lineal) con paso de 5 s; "
    "fenilefrina/efedrina con solución analítica exacta (recursiva).",
    "El parquet de salida incluye las columnas source y split además de (caseid, t) para "
    "trazabilidad y conteos.",
]


def verify_output(out_dir: Path = OUT_DIR) -> dict:
    """Verifica filas, duplicados y conteos por fuente/split frente a windows_v4."""
    win_manifest = json.loads((paths.WINDOWS_DIR / "manifest.json").read_text())
    expected_counts = win_manifest["n_windows_by_source_split"]
    total = 0
    dup = 0
    counts: dict[str, int] = {}
    parts = sorted(out_dir.glob("source=*/split=*/part-*.parquet"))
    for part in parts:
        tbl = pq.ParquetFile(part).read(columns=["caseid", "t"])
        df = tbl.to_pandas()
        total += len(df)
        key = str(part.parent.relative_to(out_dir)).replace("\\", "/")
        counts[key] = counts.get(key, 0) + len(df)
        dup += int(df.duplicated(subset=["caseid", "t"]).sum())
    mismatches = {k: (counts.get(k, 0), expected_counts.get(k))
                  for k in sorted(expected_counts)
                  if counts.get(k, 0) != expected_counts.get(k)}
    return dict(
        n_rows=total,
        n_duplicates=dup,
        n_partitions=len(parts),
        counts_match_windows=(not mismatches),
        mismatches=mismatches,
        expected_total=win_manifest["n_windows_by_source_split"].get("__total__", None),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PK tokens v1")
    ap.add_argument("command", choices=["run", "verify", "manifest"], nargs="?", default="run")
    ap.add_argument("--out-root", default=None,
                    help="raíz de salida (por defecto paths.PK_DIR); parquets en "
                         "<root>/windows/ y manifiesto en <root>/manifest_pk.json")
    args = ap.parse_args(argv)
    out_root = Path(args.out_root) if args.out_root else paths.PK_DIR
    out_dir = out_root / "windows"
    if args.command == "run":
        summary = run_all(out_dir)
        summary["verify"] = verify_output(out_dir)
        manifest = build_manifest(summary, ASSUMPTIONS)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "manifest_pk.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_output(out_dir), indent=2))
    elif args.command == "manifest":
        print((out_root / "manifest_pk.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
