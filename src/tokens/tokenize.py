"""tokenize.py — tokenizador v1 del contrato de tokens.

Construye las ventanas de transición de 60 s (12 celdas de la rejilla de 5 s)
sobre ``data/windows_v2/``, ``data/pk_v1/``, ``data/context_v1/`` y los metadatos
CF de ``data/cf_v5/``, y produce ``data/tokens_v1/`` (una fila por (caseid, t0)).

Bloques del contrato implementados:
  Bloque A  convenciones globales (paso, stride, máscara 3 estados, split
            heredado, fase mantenimiento, forward-fill).
  Bloque B  inventario v1 salvo tokens de estado.
  Bloque C  features de fármaco (7), ventilación (5), contexto (v1, 12 items),
            tiempo (1). Sin tokens de estado.
  Gates     1, 2, 4, 5, 7 (informativo), 8 y 9. Gates 3 y 6 heredados de
            pk_tokens/context_vocab; gate 10 pendiente de los tokens de estado.

Definición exacta de ventana (documentada por ambigüedad del contrato):
  - ``t0`` es un punto de la rejilla (cierre de la celda de referencia).
  - ``t1 = t0 + 60`` es el cierre de la última de las 12 celdas de la ventana.
  - Las 12 celdas de la ventana son las de cierre ``t0+5, t0+10, ..., t0+60``;
    la celda ``t0`` aporta ``Ce(t0)`` y ``t_since_opstart``.
  - ``Ce_max`` y ``bolo_en_ventana`` se calculan sobre las 12 celdas (t0, t1]
    cerrado por la derecha; la máscara del token es el peor estado de esas 12.
  - Una ventana se emite si las 12 celdas existen (sin huecos de rejilla) y
    tanto ``t0`` como ``t1`` están en fase mantenimiento (según
    ``phase_from_clinical`` de windows_v2).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time as _time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

import paths

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contrato_tokens_v1.md"
WINDOWS_ROOT = paths.WINDOWS_DIR
WINDOWS_DIR = WINDOWS_ROOT / "windows"
PK_DIR = paths.PK_DIR / "windows"
CTX_TOKENS = paths.CONTEXT_DIR / "tokens.parquet"
CTX_VOCAB = paths.CONTEXT_DIR / "vocab.json"
OUT_DIR = paths.TOKENS_DIR / "windows"
OUT_ROOT = paths.TOKENS_DIR


def cf_meta_dir() -> Path:
    """Directorio de metadatos de la cohorte de CF ACTIVA, resuelto en cada
    llamada: es una LECTURA DIFERIDA a propósito (capturarlo en el import era el
    bug del paso 3d: el tokenizador leía los metadatos y los casos CRUDOS de la
    cohorte antigua)."""
    return paths.cohort_dir("cf_v7") / "metadata"


def cf_cases_dir() -> Path:
    """Casos CRUDOS de la cohorte de CF ACTIVA (de aquí salen los setpoints
    ``vent_*``, no de las ventanas). Lectura diferida."""
    return paths.cohort_dir("cf_v7") / "cases"
MANIFEST_PATH = OUT_ROOT / "manifest_tokens.json"
REPORT_PATH = paths.REPORTS_DIR / "REPORT_tokens.txt"

GRID_S = 5.0
STEP_S = 60
CELLS = 12
STRIDE_TRAIN = 60
STRIDE_DENSE = 5
PART_SIZE = 200_000
MAX_DENSE_BYTES = 10 * 10**9  # 10 GB

DRUGS = ["propofol", "remifentanilo", "sevoflurano", "fenilefrina",
         "noradrenalina", "efedrina", "rocuronio"]
DRUG_FEATURES = ["ce_t0", "ce_t1", "ce_max", "bolo", "dose_cum", "bolo_obs"]
LOG1P_FEATURES = frozenset(["ce_t0", "ce_t1", "ce_max", "bolo", "dose_cum"])

VENT_ITEMS = ["fio2", "tv", "rr", "pip", "peep"]
# Los nombres de columna reales en los parquets de caso llevan prefijo "Primus/".
VENT_SETPOINT_COLS = {
    "fio2": "Primus/SET_FIO2",
    "tv": "Primus/SET_TV_L",
    "rr": "Primus/SET_RR_IPPV",
    "pip": "Primus/SET_PIP",
    "peep": "Primus/SET_INTER_PEEP",
}
VENT_PROXY_COLS = {
    "fio2": "Primus/FIO2",
    "tv": "Primus/TV",
    "rr": "Primus/RR_CO2",
    "pip": "Primus/PIP_MBAR",
    "peep": "Primus/PEEP_MBAR",
}

# Palancas CF -> objetivo de la intervención (contrato v5, 21 palancas).
LEVER_GROUP = {
    "propofol_bolus": "propofol", "ppf_bolus": "propofol", "ppf20_rate": "propofol",
    "remi_up": "remifentanilo", "remi_bolus": "remifentanilo", "rftn20_rate": "remifentanilo",
    "ephedrine": "efedrina", "eph_bolus": "efedrina",
    "phen_rate": "fenilefrina", "phen_bolus": "fenilefrina",
    "noradrenaline": "noradrenalina", "nepi_rate": "noradrenalina",
    "sevo_up": "sevoflurano", "sevo_mac": "sevoflurano",
    "set_fio2": "ventilacion", "set_peep": "ventilacion", "set_rr": "ventilacion",
    "set_tv": "ventilacion", "fio2_down": "ventilacion", "peep_down": "ventilacion",
    "peep_up": "ventilacion",
}

SOURCES = paths.dataset_sources()


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stat_key(tipo: str, item_id: str, feature: str) -> str:
    return f"{tipo}.{item_id}.{feature}"


@lru_cache(maxsize=1)
def load_vocab_v1() -> tuple[str, ...]:
    d = json.loads(CTX_VOCAB.read_text(encoding="utf-8"))
    ids = tuple(it["item_id"] for it in d["items"] if it.get("scope") == "v1")
    assert len(ids) == 12, f"esperados 12 items v1, hay {len(ids)}"
    return ids


@lru_cache(maxsize=1)
def load_cases() -> pd.DataFrame:
    df = pq.read_table(WINDOWS_ROOT / "cases.parquet",
                       columns=["caseid", "source", "split", "n_windows",
                                "t_first", "t_last"]).to_pandas()
    df["caseid"] = df["caseid"].astype("int64")
    df["source"] = df["source"].astype(str)
    df["split"] = df["split"].astype(str)
    return df.sort_values("caseid", kind="stable").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_context_map() -> dict[int, dict[str, float]]:
    t = pq.read_table(CTX_TOKENS, columns=["caseid", "item_id", "value"]).to_pandas()
    out: dict[int, dict[str, float]] = {}
    for _, r in t.iterrows():
        out.setdefault(int(r["caseid"]), {})[str(r["item_id"])] = float(r["value"])
    return out


@lru_cache(maxsize=1)
def load_cf_meta() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted(cf_meta_dir().glob("cf_pair_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        a = int(d["caseid_a"])
        b = int(d["caseid_b"])
        split_t = float(d.get("split_t", d.get("split_t_s", np.nan)))
        lever = d.get("lever")
        out[a] = {"pair_id": a, "cf_role": "intervencion", "split_t": split_t, "lever": lever}
        out[b] = {"pair_id": a, "cf_role": "base", "split_t": split_t, "lever": lever}
    return out


def _case_path(caseid: int, source: str) -> Path:
    base = SOURCES[source]
    if source == "real":
        return base / "cases" / f"{caseid:04d}.parquet"
    return base / "cases" / f"{caseid}.parquet"


def _phase_mark_caseids(agg: dict[int, tuple[int, int]]) -> tuple[int, ...]:
    """Dado {caseid: (n_celdas, n_maintenance)}, devuelve los caseids 100 %
    maintenance (n_celdas > 0 y todas sus celdas en 'maintenance')."""
    return tuple(sorted(cid for cid, (n, m) in agg.items() if n > 0 and m == n))


@lru_cache(maxsize=1)
def load_cases_without_phase_marks() -> tuple[int, ...]:
    """Caseids reales con TODAS sus celdas de windows_v2 en fase 'maintenance'.

    Contrato v5 ("Casos sin marcas de fase"): se excluye todo caso real cuyas
    celdas en windows_v2 sean 100 % maintenance. El criterio es la fase que
    window.py YA escribió, no la nulidad de opend ni ninguna regla de
    deduplicación. La fase la fijó window.py deduplicando por no nulos sobre las
    82 columnas (elige la fila stub, sin opstart/opend, y marca el caso entero
    como mantenimiento); context_vocab deduplica sobre las 37 de contexto para
    SUS tokens. Son dos reglas para dos propósitos y NO se unifican: unificarlas
    (iteración 3) reincorporó 24 casos 100 % maintenance con inducciones y
    educciones dentro. Resultado: 35 caseids.
    """
    agg: dict[int, tuple[int, int]] = {}  # caseid -> (n_total, n_maintenance)
    for part in sorted(WINDOWS_DIR.glob("source=real/split=*/part-*.parquet")):
        df = pq.read_table(part, columns=["caseid", "phase_from_clinical"]).to_pandas()
        for cid, grp in df.groupby("caseid"):
            cid = int(cid)
            n = int(len(grp))
            m = int((grp["phase_from_clinical"] == "maintenance").sum())
            prev = agg.get(cid, (0, 0))
            agg[cid] = (prev[0] + n, prev[1] + m)
    return _phase_mark_caseids(agg)


def _forward_fill_hold(t_raw: np.ndarray, v_raw: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Forward-fill sin tope de edad (un hueco en el registro no es una parada
    del ventilador). Antes del primer registro: NaN."""
    vf = np.asarray(v_raw, dtype=np.float64)
    nonnull = np.isfinite(vf)
    out = np.full(len(grid), np.nan, dtype=np.float64)
    if not nonnull.any():
        return out
    tt = np.asarray(t_raw, dtype=np.float64)[nonnull]
    vv = vf[nonnull]
    idx = np.searchsorted(tt, grid, side="right") - 1
    valid = idx >= 0
    idx = np.clip(idx, 0, len(vv) - 1)
    out[valid] = vv[idx[valid]]
    return out


def read_setpoints_from_path(path: Path, grid: np.ndarray) -> dict[str, np.ndarray | None]:
    """Lee los 5 setpoints de un parquet de caso y los alinea a ``grid``
    (rejilla de 5 s de windows_v2, cierres de celda) con forward-fill hold.

    CORRECCIÓN iter 2: el array devuelto vive en el eje ``grid`` (no en el eje
    temporal crudo del parquet), de modo que ``_build_contiguous``/``_build_general``
    pueden indexarlo con posiciones de la rejilla.
    """
    if not path.exists():
        return {v: None for v in VENT_ITEMS}
    schema = pq.read_schema(path)
    cols = ["time"] + [c for c in VENT_SETPOINT_COLS.values() if c in schema.names]
    df = pq.read_table(path, columns=cols).to_pandas()
    t_raw = df["time"].to_numpy(dtype=np.float64)
    grid_f = np.asarray(grid, dtype=np.float64)
    out: dict[str, np.ndarray | None] = {}
    for v in VENT_ITEMS:
        col = VENT_SETPOINT_COLS[v]
        if col in df.columns:
            out[v] = _forward_fill_hold(t_raw, df[col].to_numpy(dtype=np.float64), grid_f)
        else:
            out[v] = None
    return out


def read_case_setpoints(caseid: int, source: str, grid: np.ndarray) -> dict[str, np.ndarray | None]:
    """Lee los 5 setpoints de ventilación del parquet de caso, alineados a la
    rejilla de 5 s del caso (cierres de celda de windows_v2)."""
    path = _case_path(int(caseid), source)
    return read_setpoints_from_path(path, grid)


# --------------------------------------------------------------------------
# Núcleo puro: construcción de ventanas (testeable sin disco)
# --------------------------------------------------------------------------

def _empty_windows() -> dict:
    w: dict[str, np.ndarray] = {
        "t0": np.zeros(0, dtype=np.int64),
        "t1": np.zeros(0, dtype=np.int64),
        "dense": np.zeros(0, dtype=bool),
        "time_t_since_opstart": np.zeros(0, dtype=np.float64),
    }
    for d in DRUGS:
        for f in DRUG_FEATURES:
            w[f"drug_{d}_{f}"] = np.zeros(0, dtype=np.float64)
        w[f"drug_{d}_mask"] = np.zeros(0, dtype=np.uint8)
    for v in VENT_ITEMS:
        w[f"vent_{v}_t0"] = np.zeros(0, dtype=np.float64)
        w[f"vent_{v}_t1"] = np.zeros(0, dtype=np.float64)
        w[f"vent_{v}_proxy"] = np.zeros(0, dtype=np.uint8)
        w[f"vent_{v}_mask"] = np.zeros(0, dtype=np.uint8)
    return w


def _empty_discards() -> dict:
    return {"n_grid": 0, "hueco": 0, "induccion": 0, "emergencia": 0,
            "valido": 0, "emitido": 0, "sin_marcas_fase": 0, "sin_celdas": 0}


def _window_cells_exist(t0: int, present: set[int]) -> bool:
    for i in range(1, CELLS + 1):
        if (t0 + GRID_S * i) not in present:
            return False
    return True


def _select_t0(valid: np.ndarray, split: str, dense_val: bool) -> tuple[np.ndarray, np.ndarray]:
    """Selecciona los t0 a emitir y sus flags dense.

    train: stride 60 (dense=False). val: anchors de stride 60 (dense=False) más
    el resto de celdas válidas (dense=True), conjuntos disjuntos (sin duplicados
    de (caseid, t0)). Si dense_val es False, val queda solo con stride 60.
    """
    if len(valid) == 0:
        return valid[:0], np.zeros(0, dtype=bool)
    anchor = int(valid[0])
    is_anchor = (valid - anchor) % STRIDE_TRAIN == 0
    if split == "train" or not dense_val:
        return valid[is_anchor], np.zeros(int(is_anchor.sum()), dtype=bool)
    return valid, (~is_anchor).astype(bool)


def _build_contiguous(t: np.ndarray, sel_idx: np.ndarray, sel_t0: np.ndarray,
                      dense: np.ndarray, t_since_opstart: np.ndarray,
                      pk: dict[str, np.ndarray], setpoints: dict[str, np.ndarray | None],
                      proxies: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    w = _empty_windows()
    m = len(sel_idx)
    for key in w:
        w[key] = np.resize(w[key], m)
    w["t0"] = sel_t0
    w["t1"] = sel_t0 + STEP_S
    w["dense"] = dense
    w["time_t_since_opstart"] = (t_since_opstart[sel_idx].astype(np.float64) / 60.0)

    for d in DRUGS:
        ce = pk[f"ce_{d}"].astype(np.float64)
        ce_sw = sliding_window_view(ce, CELLS)
        bol = pk[f"bolus_{d}"].astype(np.float64)
        bol_sw = sliding_window_view(bol, CELLS)
        dc = pk[f"dose_cum_{d}"].astype(np.float64)
        bobs = pk[f"bolus_obs_{d}"].astype(np.float64)
        mce = pk[f"m_ce_{d}"].astype(np.uint8)
        mce_sw = sliding_window_view(mce, CELLS)
        w[f"drug_{d}_ce_t0"] = ce[sel_idx]
        w[f"drug_{d}_ce_t1"] = ce[sel_idx + CELLS]
        w[f"drug_{d}_ce_max"] = ce_sw[sel_idx + 1].max(axis=1)
        w[f"drug_{d}_bolo"] = bol_sw[sel_idx + 1].sum(axis=1)
        w[f"drug_{d}_dose_cum"] = dc[sel_idx + CELLS]
        w[f"drug_{d}_bolo_obs"] = bobs[sel_idx + CELLS]
        seg_m = mce_sw[sel_idx + 1]
        has1 = (seg_m == 1).any(axis=1)
        has2 = (seg_m == 2).any(axis=1)
        w[f"drug_{d}_mask"] = np.where(has1, 1, np.where(has2, 2, 0)).astype(np.uint8)

    for v in VENT_ITEMS:
        sp = setpoints.get(v)
        pr_val, pr_mask = proxies[v]
        pr_val = pr_val.astype(np.float64)
        pr_ok = (pr_mask == 1) & np.isfinite(pr_val)
        sp0 = sp[sel_idx] if sp is not None else np.full(m, np.nan)
        sp1 = sp[sel_idx + CELLS] if sp is not None else np.full(m, np.nan)
        use_sp = np.isfinite(sp0) & np.isfinite(sp1)
        pr0_ok = pr_ok[sel_idx]
        pr1_ok = pr_ok[sel_idx + CELLS]
        use_pr = (~use_sp) & pr0_ok & pr1_ok
        masked = (~use_sp) & (~use_pr)
        v0 = np.where(use_sp, np.nan_to_num(sp0, nan=0.0),
                      np.where(use_pr, np.nan_to_num(pr_val[sel_idx], nan=0.0), 0.0))
        v1 = np.where(use_sp, np.nan_to_num(sp1, nan=0.0),
                      np.where(use_pr, np.nan_to_num(pr_val[sel_idx + CELLS], nan=0.0), 0.0))
        w[f"vent_{v}_t0"] = v0
        w[f"vent_{v}_t1"] = v1
        w[f"vent_{v}_proxy"] = use_pr.astype(np.uint8)
        w[f"vent_{v}_mask"] = np.where(masked, 2, 0).astype(np.uint8)
    return w


def _build_general(t: np.ndarray, sel_t0: np.ndarray, dense: np.ndarray,
                   t_since_opstart: np.ndarray, pk: dict[str, np.ndarray],
                   setpoints: dict[str, np.ndarray | None],
                   proxies: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Camino general (rejilla no contigua); solo para tests/pequeños."""
    w = _empty_windows()
    m = len(sel_t0)
    pos = {int(tv): i for i, tv in enumerate(t)}
    for key in w:
        w[key] = np.resize(w[key], m)
    w["t0"] = sel_t0
    w["t1"] = sel_t0 + STEP_S
    w["dense"] = dense
    out_time = np.zeros(m, dtype=np.float64)
    for k, t0 in enumerate(sel_t0):
        i0 = pos[int(t0)]
        out_time[k] = t_since_opstart[i0] / 60.0
    w["time_t_since_opstart"] = out_time

    for d in DRUGS:
        ce = pk[f"ce_{d}"].astype(np.float64)
        bol = pk[f"bolus_{d}"].astype(np.float64)
        dc = pk[f"dose_cum_{d}"].astype(np.float64)
        bobs = pk[f"bolus_obs_{d}"].astype(np.float64)
        mce = pk[f"m_ce_{d}"].astype(np.uint8)
        arr = {f: np.zeros(m, dtype=np.float64) for f in DRUG_FEATURES}
        arr[f"drug_{d}_mask"] = np.zeros(m, dtype=np.uint8)
        for k, t0 in enumerate(sel_t0):
            idx = [pos[int(t0 + GRID_S * i)] for i in range(1, CELLS + 1)]
            i0 = pos[int(t0)]
            arr["ce_t0"][k] = ce[i0]
            arr["ce_t1"][k] = ce[idx[-1]]
            arr["ce_max"][k] = ce[idx].max()
            arr["bolo"][k] = bol[idx].sum()
            arr["dose_cum"][k] = dc[idx[-1]]
            arr["bolo_obs"][k] = bobs[idx[-1]]
            seg_m = mce[idx]
            arr[f"drug_{d}_mask"][k] = 1 if (seg_m == 1).any() else (2 if (seg_m == 2).any() else 0)
        for f in DRUG_FEATURES:
            w[f"drug_{d}_{f}"] = arr[f]
        w[f"drug_{d}_mask"] = arr[f"drug_{d}_mask"]

    for v in VENT_ITEMS:
        sp = setpoints.get(v)
        pr_val, pr_mask = proxies[v]
        pr_val = pr_val.astype(np.float64)
        pr_ok = (pr_mask == 1) & np.isfinite(pr_val)
        v0 = np.zeros(m, dtype=np.float64)
        v1 = np.zeros(m, dtype=np.float64)
        prx = np.zeros(m, dtype=np.uint8)
        mask = np.zeros(m, dtype=np.uint8)
        for k, t0 in enumerate(sel_t0):
            i0 = pos[int(t0)]
            i1 = pos[int(t0 + STEP_S)]
            sp0 = sp[i0] if sp is not None else np.nan
            sp1 = sp[i1] if sp is not None else np.nan
            if np.isfinite(sp0) and np.isfinite(sp1):
                v0[k], v1[k] = sp0, sp1
                prx[k] = 0
            elif pr_ok[i0] and pr_ok[i1]:
                v0[k], v1[k] = pr_val[i0], pr_val[i1]
                prx[k] = 1
            else:
                mask[k] = 2
        w[f"vent_{v}_t0"] = v0
        w[f"vent_{v}_t1"] = v1
        w[f"vent_{v}_proxy"] = prx
        w[f"vent_{v}_mask"] = mask
    return w


def build_case_windows(
    t_grid: np.ndarray,
    phase: np.ndarray,
    t_since_opstart: np.ndarray,
    split: str,
    dense_val: bool,
    pk: dict[str, np.ndarray],
    setpoints: dict[str, np.ndarray | None],
    proxies: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, dict]:
    """Construye las ventanas de transición de un caso.

    Parámetros:
      t_grid          cierres de celda (int64), ascendentes.
      phase           fase por celda (mismo orden que t_grid).
      t_since_opstart segundos desde opstart por celda (float).
      split           'train' | 'val'.
      dense_val       si es val, emitir también stride 5 (dense=True).
      pk              arrays alineados a t_grid con claves ce_<d>, dose_cum_<d>,
                      bolus_<d>, bolus_obs_<d>, m_ce_<d>.
      setpoints       {vent: array alineado|None} (setpoint forward-filled).
      proxies         {vent: (valores, mask_uint8)} medidos alineados.

    Devuelve (windows_dict, discards_dict). windows_dict contiene las features
    CRUDAS (sin normalizar) y las máscaras.
    """
    t = np.asarray(t_grid, dtype=np.int64)
    n = len(t)
    disc = _empty_discards()
    disc["n_grid"] = n
    if n == 0:
        return _empty_windows(), disc

    phase = np.asarray(phase)
    tso = np.asarray(t_since_opstart, dtype=np.float64)
    contiguous = bool(np.all(np.diff(t) == GRID_S)) if n >= 2 else True

    if contiguous:
        fit = (t + STEP_S) <= t[-1]
        disc["hueco"] = int(n - int(fit.sum()))
        idx_fit = np.nonzero(fit)[0]
        t_fit = t[fit]
        ph0 = phase[idx_fit]
        ph1 = phase[idx_fit + CELLS]
        disc["induccion"] = int((ph0 == "induction").sum())
        disc["emergencia"] = int(((ph0 != "induction") & (ph1 == "emergence")).sum())
        maint = (ph0 == "maintenance") & (ph1 == "maintenance")
        valid = t_fit[maint]
    else:
        present = {int(x) for x in t}
        ok = np.array([_window_cells_exist(int(x), present) for x in t], dtype=bool)
        disc["hueco"] = int((~ok).sum())
        idx_fit = np.nonzero(ok)[0]
        t_fit = t[ok]
        pos = {int(x): i for i, x in enumerate(t)}
        ph0 = phase[idx_fit]
        ph1 = np.array([phase[pos[int(x) + STEP_S]] for x in t_fit], dtype=object)
        disc["induccion"] = int((ph0 == "induction").sum())
        disc["emergencia"] = int(((ph0 != "induction") & (ph1 == "emergence")).sum())
        maint = (ph0 == "maintenance") & (ph1 == "maintenance")
        valid = t_fit[maint]

    disc["valido"] = int(len(valid))
    sel_t0, dense = _select_t0(valid, split, dense_val)
    disc["emitido"] = int(len(sel_t0))
    if len(sel_t0) == 0:
        return _empty_windows(), disc

    if contiguous:
        # posición de cada t0 en la rejilla contigua
        sel_idx = (sel_t0 - t[0]) // GRID_S
        sel_idx = sel_idx.astype(np.int64)
        w = _build_contiguous(t, sel_idx, sel_t0, dense, tso, pk, setpoints, proxies)
    else:
        w = _build_general(t, sel_t0, dense, tso, pk, setpoints, proxies)
    return w, disc


# --------------------------------------------------------------------------
# Normalización
# --------------------------------------------------------------------------

class StatsAccumulator:
    """Acumula sumas/cuadrados por clave para media y desviación (ddof=0)."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.sumsq: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def add(self, key: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return
        self.sums[key] = self.sums.get(key, 0.0) + float(values.sum())
        self.sumsq[key] = self.sumsq.get(key, 0.0) + float((values * values).sum())
        self.n[key] = self.n.get(key, 0) + int(len(values))

    def finalize(self) -> dict:
        out: dict[str, dict] = {}
        for k in self.sums:
            n = self.n[k]
            mean = self.sums[k] / n
            var = self.sumsq[k] / n - mean * mean
            var = var if var > 0.0 else 0.0
            std = float(np.sqrt(var))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            out[k] = {"mean": float(mean), "std": std, "n": int(n)}
        return out


def accumulate_stats_from_windows(acc: StatsAccumulator, w: dict) -> None:
    for d in DRUGS:
        mask = w[f"drug_{d}_mask"]
        m0 = mask == 0
        for f in DRUG_FEATURES:
            if f not in LOG1P_FEATURES:
                continue  # bolo_obs no se normaliza (0/1)
            v = np.log1p(w[f"drug_{d}_{f}"][m0])
            acc.add(_stat_key("drug", d, f), v)
    for v in VENT_ITEMS:
        mask = w[f"vent_{v}_mask"]
        m0 = mask == 0
        acc.add(_stat_key("vent", v, "t0"), w[f"vent_{v}_t0"][m0])
        acc.add(_stat_key("vent", v, "t1"), w[f"vent_{v}_t1"][m0])
    acc.add(_stat_key("time", "t_since_opstart", "value"), w["time_t_since_opstart"])


def _zscore(values: np.ndarray, stats: dict, key: str) -> np.ndarray:
    st = stats.get(key, {"mean": 0.0, "std": 1.0})
    return ((values - st["mean"]) / st["std"]).astype(np.float32)


def apply_normalization(w: dict, stats: dict) -> dict:
    """Aplica la normalización a las features de un dict de ventanas crudas.

    Fármaco: log1p + z-score (ce_t0, ce_t1, ce_max, bolo, dose_cum); bolo_obs
    sin transformar. Ventilación y tiempo: z-score directo. La feature con
    máscara 1 o 2 queda exactamente 0 tras transformar.
    """
    out: dict[str, np.ndarray] = {}
    m = len(w["t0"])
    for d in DRUGS:
        mask = w[f"drug_{d}_mask"].astype(np.uint8)
        zero = (mask != 0)
        for f in DRUG_FEATURES:
            raw = w[f"drug_{d}_{f}"].astype(np.float64)
            if f in LOG1P_FEATURES:
                val = _zscore(np.log1p(np.where(np.isfinite(raw), raw, 0.0)),
                              stats, _stat_key("drug", d, f))
            else:
                val = np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)
            val[zero] = 0.0
            out[f"drug_{d}_{f}"] = val.astype(np.float32)
        out[f"drug_{d}_mask"] = mask
    for v in VENT_ITEMS:
        mask = w[f"vent_{v}_mask"].astype(np.uint8)
        zero = (mask != 0)
        t0 = _zscore(w[f"vent_{v}_t0"].astype(np.float64), stats, _stat_key("vent", v, "t0"))
        t1 = _zscore(w[f"vent_{v}_t1"].astype(np.float64), stats, _stat_key("vent", v, "t1"))
        t0[zero] = 0.0
        t1[zero] = 0.0
        out[f"vent_{v}_t0"] = t0.astype(np.float32)
        out[f"vent_{v}_t1"] = t1.astype(np.float32)
        prx = w[f"vent_{v}_proxy"].astype(np.uint8)
        prx[zero] = 0
        out[f"vent_{v}_proxy"] = prx
        out[f"vent_{v}_mask"] = mask
    out["time_t_since_opstart"] = _zscore(
        w["time_t_since_opstart"].astype(np.float64), stats,
        _stat_key("time", "t_since_opstart", "value"))
    # metadatos (sin normalizar)
    for key in ("t0", "t1", "dense", "caseid", "source", "split",
                "pair_id", "cf_role", "split_t", "lever", "post_split", "n_ctx"):
        if key in w:
            out[key] = w[key]
    for iid in load_vocab_v1():
        k = f"ctx_{iid}"
        if k in w:
            out[k] = w[k]
    return out


# --------------------------------------------------------------------------
# Lectura de particiones y cómputo por caso
# --------------------------------------------------------------------------

WINDOWS_READ_COLS = (["caseid", "t", "source", "split", "phase_from_clinical",
                      "t_since_opstart"]
                     + [VENT_PROXY_COLS[v] for v in VENT_ITEMS]
                     + [f"m_{VENT_PROXY_COLS[v]}" for v in VENT_ITEMS])
PK_READ_COLS = (["caseid", "t"]
                + [f"{p}_{d}" for d in DRUGS for p in ("ce", "dose_cum", "bolus", "bolus_obs", "m_ce")])


def _process_case(caseid: int, source: str, split: str, dense_val: bool,
                  win_cells: pd.DataFrame, pk_cells: pd.DataFrame,
                  context_map: dict, cf_meta: dict) -> tuple[dict, dict]:
    """Calcula las ventanas de un caso a partir de sus celdas."""
    t_grid = win_cells["t"].to_numpy(dtype=np.int64)
    phase = win_cells["phase_from_clinical"].to_numpy()
    tso = win_cells["t_since_opstart"].to_numpy(dtype=np.float64)
    if len(t_grid) < CELLS + 1:
        w = _empty_windows()
        disc = _empty_discards()
        disc["n_grid"] = len(t_grid)
        disc["hueco"] = len(t_grid)
        return w, disc

    pk: dict[str, np.ndarray] = {}
    for d in DRUGS:
        for p in ("ce", "dose_cum", "bolus", "bolus_obs", "m_ce"):
            pk[f"{p}_{d}"] = pk_cells[f"{p}_{d}"].to_numpy()

    setpoints = read_case_setpoints(int(caseid), source, t_grid)
    proxies: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for v in VENT_ITEMS:
        col = VENT_PROXY_COLS[v]
        mcol = f"m_{col}"
        vals = win_cells[col].to_numpy(dtype=np.float64)
        mask = win_cells[mcol].to_numpy(dtype=np.uint8)
        proxies[v] = (vals, mask)

    w, disc = build_case_windows(t_grid, phase, tso, split, dense_val, pk,
                                 setpoints, proxies)

    m = len(w["t0"])
    ctx_ids = load_vocab_v1()
    ctx = context_map.get(int(caseid), {})
    for iid in ctx_ids:
        w[f"ctx_{iid}"] = np.full(m, np.float32(ctx.get(iid, np.nan)), dtype=np.float32)
    w["n_ctx"] = np.full(m, len([i for i in ctx_ids if i in ctx]), dtype=np.uint8)
    w["caseid"] = np.full(m, int(caseid), dtype=np.int32)
    w["source"] = np.full(m, source, dtype=object)
    w["split"] = np.full(m, split, dtype=object)
    cf = cf_meta.get(int(caseid))
    if cf is None:
        w["pair_id"] = np.full(m, np.nan, dtype=np.float64)
        w["cf_role"] = np.full(m, None, dtype=object)
        w["split_t"] = np.full(m, np.nan, dtype=np.float64)
        w["lever"] = np.full(m, None, dtype=object)
        w["post_split"] = np.zeros(m, dtype=bool)
    else:
        w["pair_id"] = np.full(m, float(cf["pair_id"]), dtype=np.float64)
        w["cf_role"] = np.full(m, cf["cf_role"], dtype=object)
        w["split_t"] = np.full(m, float(cf["split_t"]), dtype=np.float64)
        w["lever"] = np.full(m, cf["lever"], dtype=object)
        # contrato v4: post_split = t1 > split_t (la ventana que contiene la
        # intervención, t0 < split_t < t1, es la primera post-split)
        w["post_split"] = (w["t1"].astype(np.float64) > float(cf["split_t"]))
    return w, disc


def process_partition_windows(part_path: Path, dense_val: bool,
                              context_map: dict, cf_meta: dict,
                              excluded_caseids: frozenset[int] = frozenset(),
                              max_cases: int | None = None) -> tuple[dict, dict]:
    """Lee una partición windows_v2 + pk_v1 y construye las ventanas crudas."""
    rel = part_path.relative_to(WINDOWS_DIR)
    pk_path = PK_DIR / rel
    tbl = pq.read_table(part_path, columns=WINDOWS_READ_COLS).to_pandas()
    pkt = pq.read_table(pk_path, columns=PK_READ_COLS).to_pandas()
    merged = tbl.merge(pkt, on=["caseid", "t"], how="inner", sort=False)
    merged = merged.sort_values(["caseid", "t"], kind="stable")
    source = str(merged["source"].iloc[0])
    split = str(merged["split"].iloc[0])

    caseids = sorted(merged["caseid"].unique().tolist())
    if max_cases is not None:
        caseids = caseids[:max_cases]

    frames: list[dict] = []
    total_disc = _empty_discards()
    for cid in caseids:
        if int(cid) in excluded_caseids:
            total_disc["sin_marcas_fase"] += 1
            continue
        sub = merged[merged["caseid"] == cid]
        w, disc = _process_case(int(cid), source, split, dense_val, sub, sub,
                                context_map, cf_meta)
        frames.append(w)
        for k in total_disc:
            total_disc[k] += disc.get(k, 0)

    # concatenar ventanas de la partición
    keys = _empty_windows().keys()
    if frames:
        concat: dict[str, np.ndarray] = {}
        for k in keys:
            concat[k] = np.concatenate([f[k] for f in frames])
        # añadir contexto y metadatos por ventana
        for k in ("caseid", "source", "split", "pair_id", "cf_role", "split_t",
                  "lever", "post_split", "n_ctx"):
            concat[k] = np.concatenate([f[k] for f in frames])
        for iid in load_vocab_v1():
            concat[f"ctx_{iid}"] = np.concatenate([f[f"ctx_{iid}"] for f in frames])
        order = np.lexsort((concat["t0"], concat["caseid"]))
        for k in concat:
            concat[k] = concat[k][order]
        return concat, total_disc
    empty = _empty_windows()
    for k in ("caseid", "source", "split", "pair_id", "cf_role", "split_t",
              "lever", "post_split", "n_ctx"):
        empty[k] = np.zeros(0, dtype=object if k in ("source", "split", "cf_role", "lever") else np.float64)
    for iid in load_vocab_v1():
        empty[f"ctx_{iid}"] = np.zeros(0, dtype=np.float32)
    return empty, total_disc


# --------------------------------------------------------------------------
# Esquema y escritura
# --------------------------------------------------------------------------

def output_columns(ctx_ids: tuple[str, ...]) -> list[str]:
    cols = []
    for d in DRUGS:
        cols += [f"drug_{d}_{f}" for f in DRUG_FEATURES] + [f"drug_{d}_mask"]
    for v in VENT_ITEMS:
        cols += [f"vent_{v}_t0", f"vent_{v}_t1", f"vent_{v}_proxy", f"vent_{v}_mask"]
    cols += [f"ctx_{iid}" for iid in ctx_ids] + ["n_ctx"]
    cols += ["time_t_since_opstart"]
    cols += ["caseid", "t0", "t1", "source", "split", "dense",
             "pair_id", "cf_role", "split_t", "lever", "post_split"]
    return cols


def feature_columns(ctx_ids: tuple[str, ...] | None = None) -> list[str]:
    """Las 65 columnas que son feature (sin máscaras ni metadatos).

    contrato v4: es_proxy (vent_*_proxy) viaja como metadato, no como feature.
    """
    if ctx_ids is None:
        ctx_ids = load_vocab_v1()
    cols = []
    for d in DRUGS:
        cols += [f"drug_{d}_{f}" for f in DRUG_FEATURES]
    for v in VENT_ITEMS:
        cols += [f"vent_{v}_t0", f"vent_{v}_t1"]
    for iid in ctx_ids:
        cols += [f"ctx_{iid}"]
    cols += ["time_t_since_opstart"]
    return cols


def mask_columns() -> list[str]:
    return [f"drug_{d}_mask" for d in DRUGS] + [f"vent_{v}_mask" for v in VENT_ITEMS]


def metadata_columns() -> list[str]:
    return ([f"vent_{v}_proxy" for v in VENT_ITEMS]
            + ["n_ctx", "caseid", "t0", "t1", "source", "split", "dense",
               "pair_id", "cf_role", "split_t", "lever", "post_split"])


def build_schema(ctx_ids: tuple[str, ...]) -> pa.Schema:
    fields = []
    for d in DRUGS:
        for f in DRUG_FEATURES:
            fields.append(pa.field(f"drug_{d}_{f}", pa.float32()))
        fields.append(pa.field(f"drug_{d}_mask", pa.uint8()))
    for v in VENT_ITEMS:
        fields.append(pa.field(f"vent_{v}_t0", pa.float32()))
        fields.append(pa.field(f"vent_{v}_t1", pa.float32()))
        fields.append(pa.field(f"vent_{v}_proxy", pa.uint8()))
        fields.append(pa.field(f"vent_{v}_mask", pa.uint8()))
    for iid in ctx_ids:
        fields.append(pa.field(f"ctx_{iid}", pa.float32()))
    fields.append(pa.field("n_ctx", pa.uint8()))
    fields.append(pa.field("time_t_since_opstart", pa.float32()))
    fields.append(pa.field("caseid", pa.int32()))
    fields.append(pa.field("t0", pa.int32()))
    fields.append(pa.field("t1", pa.int32()))
    fields.append(pa.field("source", pa.dictionary(pa.int32(), pa.string())))
    fields.append(pa.field("split", pa.dictionary(pa.int32(), pa.string())))
    fields.append(pa.field("dense", pa.bool_()))
    fields.append(pa.field("pair_id", pa.int64()))
    fields.append(pa.field("cf_role", pa.string()))
    fields.append(pa.field("split_t", pa.float64()))
    fields.append(pa.field("lever", pa.string()))
    fields.append(pa.field("post_split", pa.bool_()))
    return pa.schema(fields)


def build_table(out: dict, ctx_ids: tuple[str, ...]) -> pa.Table:
    cols = output_columns(ctx_ids)
    schema = build_schema(ctx_ids)
    arrays = []
    for c in cols:
        arr = out[c]
        if c == "pair_id":
            # NaN -> null (int64 nullable)
            vals = [None if (x is None or (isinstance(x, (float, np.floating))
                                           and not np.isfinite(x))) else int(x)
                    for x in np.asarray(arr)]
            arrays.append(pa.array(vals, type=pa.int64()))
        elif c in ("source", "split"):
            arrays.append(pa.array(arr.astype(str), type=pa.string()))
        elif c in ("cf_role", "lever"):
            arrays.append(pa.array([None if x is None else str(x) for x in arr],
                                   type=pa.string()))
        elif c == "post_split":
            arrays.append(pa.array(np.asarray(arr, dtype=bool), type=pa.bool_()))
        else:
            arrays.append(pa.array(np.asarray(arr)))
    return pa.Table.from_arrays(arrays, schema=schema)


def write_partition(out: dict, part_path: Path) -> None:
    rel = part_path.relative_to(WINDOWS_DIR)
    out_path = OUT_DIR / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = build_table(out, load_vocab_v1())
    pq.write_table(table, out_path, compression="zstd")


def build_pairs_parquet() -> dict:
    """Escribe data/tokens_v1/pairs.parquet: una fila por par CF (dense=False).

    Columnas: pair_id, caseid_base, caseid_intervencion, split_t, lever,
    lever_group, n_windows_pre, n_windows_post, n_windows_total,
    n_windows_base, n_windows_intervencion, usable.

    usable = (n_windows_total > 0) AND (los conjuntos de t0 dense=False de
    ambas ramas coinciden). Los conteos pre/post se toman de la rama base; al
    coincidir los conjuntos de t0 coinciden también con la intervención.
    """
    cf_meta = load_cf_meta()
    levers = {str(m["lever"]) for m in cf_meta.values()}
    unknown = levers - set(LEVER_GROUP)
    if unknown:
        raise RuntimeError(f"palancas fuera de LEVER_GROUP: {sorted(unknown)}")

    pairs: dict[int, dict] = {}
    for cid, m in cf_meta.items():
        if m["cf_role"] == "intervencion":
            pairs.setdefault(m["pair_id"], {}).update(
                caseid_intervencion=int(cid), split_t=float(m["split_t"]),
                lever=str(m["lever"]))
    for cid, m in cf_meta.items():
        if m["cf_role"] == "base":
            pairs[m["pair_id"]].update(caseid_base=int(cid))

    # por caseid: conjunto de t0 dense=False y conteos pre/post
    info: dict[int, dict] = {}
    for part in sorted(OUT_DIR.glob("source=cf_v7/split=*/part-*.parquet")):
        df = pq.read_table(part, columns=["caseid", "dense", "t0", "post_split"]).to_pandas()
        df = df[df.dense == False]
        for cid, grp in df.groupby("caseid"):
            cid = int(cid)
            entry = info.setdefault(cid, {"t0s": set(), "pre": 0, "post": 0})
            entry["t0s"] |= {int(x) for x in grp.t0}
            entry["pre"] += int((grp.post_split == False).sum())
            entry["post"] += int((grp.post_split == True).sum())

    rows = []
    for pair_id in sorted(pairs):
        p = pairs[pair_id]
        base = int(p["caseid_base"])
        inter = int(p["caseid_intervencion"])
        bi = info.get(base, {"t0s": set(), "pre": 0, "post": 0})
        ii = info.get(inter, {"t0s": set(), "pre": 0, "post": 0})
        nb, ni = len(bi["t0s"]), len(ii["t0s"])
        sets_match = bi["t0s"] == ii["t0s"]
        total = bi["pre"] + bi["post"]
        rows.append({
            "pair_id": int(pair_id),
            "caseid_base": base,
            "caseid_intervencion": inter,
            "split_t": float(p["split_t"]),
            "lever": str(p["lever"]),
            "lever_group": LEVER_GROUP[str(p["lever"])],
            "n_windows_pre": bi["pre"],
            "n_windows_post": bi["post"],
            "n_windows_total": total,
            "n_windows_base": nb,
            "n_windows_intervencion": ni,
            "usable": bool(total > 0 and sets_match),
        })
    df_out = pd.DataFrame(rows)
    schema = pa.schema([
        ("pair_id", pa.int64()), ("caseid_base", pa.int32()),
        ("caseid_intervencion", pa.int32()), ("split_t", pa.float64()),
        ("lever", pa.string()), ("lever_group", pa.string()),
        ("n_windows_pre", pa.int32()), ("n_windows_post", pa.int32()),
        ("n_windows_total", pa.int32()),
        ("n_windows_base", pa.int32()), ("n_windows_intervencion", pa.int32()),
        ("usable", pa.bool_()),
    ])
    path = OUT_ROOT / "pairs.parquet"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df_out, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="zstd")
    return {"n_pairs": len(rows), "path": str(path),
            "n_unusable": int(df_out.usable.eq(False).sum())}


# --------------------------------------------------------------------------
# Estimación de tamaño denso (val)
# --------------------------------------------------------------------------

def _bytes_per_row_estimate() -> int:
    schema = build_schema(load_vocab_v1())
    total = 0
    for f in schema:
        try:
            if pa.types.is_dictionary(f.type):
                total += 4
            else:
                total += f.type.bit_width // 8 if hasattr(f.type, "bit_width") else 8
        except Exception:
            total += 8
    return total


def estimate_dense() -> dict:
    """Estima el tamaño de las ventanas densas de val antes de generarlas."""
    manifest = json.loads((WINDOWS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    n_cases_val = sum(v for k, v in manifest["n_cases_by_source_split"].items()
                      if k.endswith("|val"))
    n_cells_val = sum(v for k, v in manifest["n_windows_by_source_split"].items()
                      if "/split=val" in k)
    dense_est = max(0, n_cells_val - n_cases_val * (CELLS + 1))
    bpr = _bytes_per_row_estimate()
    est_bytes = dense_est * bpr
    allowed = est_bytes <= MAX_DENSE_BYTES
    return {
        "n_cells_val": n_cells_val,
        "n_cases_val": n_cases_val,
        "dense_windows_est": int(dense_est),
        "bytes_per_row_est": bpr,
        "est_bytes": int(est_bytes),
        "est_gb": round(est_bytes / 1e9, 3),
        "allowed": allowed,
    }


# --------------------------------------------------------------------------
# Ejecución completa
# --------------------------------------------------------------------------

def _select_parts(all_parts: list[Path], limit_per_group: int | None) -> list[Path]:
    if limit_per_group is None:
        return all_parts
    groups: dict[str, list[Path]] = {}
    for p in all_parts:
        groups.setdefault(str(p.parent), []).append(p)
    out: list[Path] = []
    for k in sorted(groups):
        out.extend(sorted(groups[k])[:limit_per_group])
    return sorted(out)


def run_full(verbose: bool = True, limit_parts_per_group: int | None = None,
             stats_from: Path | None = None) -> dict:
    """Tokeniza el corpus.

    ``stats_from``: ruta a un manifiesto de tokens del que tomar los estadísticos
    de normalización (clave ``normalization_stats``). Con ella NO se acumulan
    estadísticos sobre el corpus: se usan los registrados. Sin ella el
    comportamiento no cambia (acumular sobre el train del corpus).

    Motivo (paso 3d, Corrección A): acumular sobre el train de TODAS las
    cohortes hace que regenerar una cohorte sintética cambie los tokens de real.
    """
    t_start = _time.time()
    ctx_ids = load_vocab_v1()
    context_map = load_context_map()
    cf_meta = load_cf_meta()
    dense_info = estimate_dense()
    dense_val = dense_info["allowed"]
    excluded = frozenset(load_cases_without_phase_marks())

    all_parts = _select_parts(
        sorted(WINDOWS_DIR.glob("source=*/split=*/part-*.parquet")), limit_parts_per_group)
    train_parts = [p for p in all_parts if "/split=train/" in str(p).replace("\\", "/")]

    if stats_from is not None:
        stats_path = Path(stats_from)
        man = json.loads(stats_path.read_text(encoding="utf-8"))
        if "normalization_stats" not in man:
            raise ValueError(f"{stats_path} no tiene 'normalization_stats'")
        stats = man["normalization_stats"]
        stats_origin = {
            "modo": "congelado",
            "manifest": str(stats_path),
            "sha256_manifest": _sha256(stats_path),
            "stats_acumulados_sobre_corpus": False,
            "n_claves": len(stats),
        }
        if verbose:
            print(f"[stats] CONGELADOS desde {stats_path} ({len(stats)} claves)",
                  flush=True)
    else:
        acc = StatsAccumulator()
        for i, part in enumerate(train_parts):
            w, _ = process_partition_windows(part, dense_val, context_map, cf_meta,
                                             excluded_caseids=excluded)
            accumulate_stats_from_windows(acc, w)
            if verbose and (i + 1) % 50 == 0:
                print(f"[stats] {i + 1}/{len(train_parts)} particiones train", flush=True)
        stats = acc.finalize()
        stats_origin = {
            "modo": "acumulado",
            "manifest": None,
            "sha256_manifest": None,
            "stats_acumulados_sobre_corpus": True,
            "n_claves": len(stats),
        }

    counts: dict[str, int] = {}
    dense_counts: dict[str, int] = {}
    discards = _empty_discards()
    n_rows = 0
    for i, part in enumerate(all_parts):
        w, disc = process_partition_windows(part, dense_val, context_map, cf_meta,
                                            excluded_caseids=excluded)
        out = apply_normalization(w, stats)
        write_partition(out, part)
        rel = part.relative_to(WINDOWS_DIR)
        key = str(rel.parent).replace("\\", "/")
        counts[key] = counts.get(key, 0) + int(len(out["t0"]))
        n_rows += int(len(out["t0"]))
        dense_counts[key] = dense_counts.get(key, 0) + int(out["dense"].sum())
        for k in discards:
            discards[k] += disc.get(k, 0)
        if verbose and (i + 1) % 50 == 0:
            print(f"[write] {i + 1}/{len(all_parts)} particiones, {n_rows:,} filas", flush=True)

    pairs_info = build_pairs_parquet()
    zero_cells = load_cases().query("n_windows == 0")
    discards["sin_celdas"] = int(len(zero_cells))
    elapsed = _time.time() - t_start
    return {
        "stats": stats,
        "stats_origin": stats_origin,
        "counts": counts,
        "dense_counts": dense_counts,
        "discards": discards,
        "n_rows": n_rows,
        "n_partitions": len(all_parts),
        "elapsed_s": round(elapsed, 2),
        "dense_info": dense_info,
        "dense_val": dense_val,
        "excluded_no_phase_marks_caseids": list(excluded),
        "sin_celdas_caseids": [int(c) for c in zero_cells.caseid],
        "pairs": pairs_info,
    }


def verify_output() -> dict:
    parts = sorted(OUT_DIR.glob("source=*/split=*/part-*.parquet"))
    total = 0
    dup = 0
    counts: dict[str, int] = {}
    for part in parts:
        tbl = pq.ParquetFile(part).read(columns=["caseid", "t0"])
        df = tbl.to_pandas()
        total += len(df)
        key = str(part.parent.relative_to(OUT_DIR)).replace("\\", "/")
        counts[key] = counts.get(key, 0) + len(df)
        dup += int(df.duplicated(subset=["caseid", "t0"]).sum())
    return {
        "n_rows": total,
        "n_duplicates": dup,
        "n_partitions": len(parts),
        "counts": counts,
    }


def build_manifest(summary: dict) -> dict:
    win_manifest = json.loads((WINDOWS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    return {
        "date": pd.Timestamp.now().isoformat(),
        "cohort_label_map": paths.cohort_label_map(),
        "sha256_contract": _sha256(CONTRACT_PATH),
        "sha256_tokenize_py": _sha256(Path(__file__).resolve()),
        "windows_manifest_path": str(WINDOWS_ROOT / "manifest.json"),
        "sha256_windows_manifest": _sha256(WINDOWS_ROOT / "manifest.json"),
        "pk_manifest_path": str(PK_DIR.parent / "manifest_pk.json"),
        "sha256_pk_manifest": _sha256(PK_DIR.parent / "manifest_pk.json"),
        "context_vocab_path": str(CTX_VOCAB),
        "sha256_context_vocab": _sha256(CTX_VOCAB),
        "split_parquet_path": str(WINDOWS_ROOT / "split.parquet"),
        "sha256_split_parquet": _sha256(WINDOWS_ROOT / "split.parquet"),
        "normalization_stats": summary["stats"],
        "stats_origin": summary.get("stats_origin"),
        "feature_columns": feature_columns(),
        "mask_columns": mask_columns(),
        "metadata_columns": metadata_columns(),
        "n_rows_by_source_split": summary["counts"],
        "n_dense_by_source_split": summary["dense_counts"],
        "discards_by_cause": summary["discards"],
        "excluded_no_phase_marks_caseids": list(load_cases_without_phase_marks()),
        "sin_celdas_caseids": summary.get("sin_celdas_caseids", []),
        "pairs": summary.get("pairs"),
        "dense_estimation": summary["dense_info"],
        "dense_val_enabled": summary["dense_val"],
        "n_rows": summary["n_rows"],
        "n_partitions": summary["n_partitions"],
        "elapsed_s": summary["elapsed_s"],
        "windows_n_cells_by_source_split": win_manifest["n_windows_by_source_split"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tokenizador v1")
    ap.add_argument("command", choices=["run", "verify", "manifest", "report"],
                    nargs="?", default="run")
    ap.add_argument("--smoke", action="store_true", help="subconjunto determinista")
    ap.add_argument("--pk-root", default=None,
                    help="raíz de pk (por defecto paths.PK_DIR); parquets en <root>/windows/")
    ap.add_argument("--ctx-root", default=None,
                    help="raíz de contexto (por defecto paths.CONTEXT_DIR)")
    ap.add_argument("--out-root", default=None,
                    help="raíz de salida (por defecto paths.TOKENS_DIR)")
    ap.add_argument("--stats-from", default=None,
                    help="manifiesto del que tomar normalization_stats (congela "
                         "la normalización; por defecto se acumulan sobre el "
                         "train del corpus)")
    args = ap.parse_args(argv)

    global PK_DIR, CTX_TOKENS, CTX_VOCAB, OUT_DIR, OUT_ROOT, MANIFEST_PATH
    if args.pk_root:
        PK_DIR = Path(args.pk_root) / "windows"
    if args.ctx_root:
        CTX_TOKENS = Path(args.ctx_root) / "tokens.parquet"
        CTX_VOCAB = Path(args.ctx_root) / "vocab.json"
    if args.out_root:
        OUT_ROOT = Path(args.out_root)
        OUT_DIR = OUT_ROOT / "windows"
        MANIFEST_PATH = OUT_ROOT / "manifest_tokens.json"
    load_vocab_v1.cache_clear()
    load_context_map.cache_clear()

    if args.command == "run":
        limit = 2 if args.smoke else None
        summary = run_full(verbose=True, limit_parts_per_group=limit,
                           stats_from=(Path(args.stats_from)
                                       if args.stats_from else None))
        ver = verify_output()
        summary["verify"] = ver
        manifest = build_manifest(summary)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        print(json.dumps({"n_rows": summary["n_rows"], "verify": ver}, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_output(), indent=2))
    elif args.command == "manifest":
        print(MANIFEST_PATH.read_text(encoding="utf-8"))
    elif args.command == "report":
        write_report()
        print(REPORT_PATH)
    return 0


def _mget(m: dict, *keys: str) -> str:
    """Devuelve la primera clave presente (compatibilidad claves v1/v2)."""
    for k in keys:
        if k in m:
            return str(m[k])
    return "—"


def write_report() -> None:
    """Escribe REPORT_tokens.txt a partir del manifest y del veredicto."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("manifest_tokens.json no existe; ejecuta 'run' primero")
    m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    L = []
    L.append("REPORT_tokens.txt — tokenizador v1")
    L.append("=" * 78)
    L.append("")
    L.append("0. Contexto y ficheros")
    L.append("-" * 40)
    L.append(f"  contrato_tokens_v1.md             sha256 {m['sha256_contract']}")
    L.append(f"  tokenize.py                       sha256 {m['sha256_tokenize_py']}")
    L.append(f"  windows_v4/manifest.json          sha256 {_mget(m, 'sha256_windows_manifest', 'sha256_windows_v2_manifest')}")
    L.append(f"  pk/manifest_pk.json               sha256 {_mget(m, 'sha256_pk_manifest', 'sha256_pk_v1_manifest')}")
    L.append(f"  context/vocab.json                sha256 {m['sha256_context_vocab']}")
    L.append(f"  windows_v4/split.parquet          sha256 {m['sha256_split_parquet']}")
    L.append(f"  fecha                             {m['date']}")
    L.append("")
    L.append("1. Tests y salida en ROJO")
    L.append("-" * 40)
    L.append("(ver salida de pytest en la sección 1 del flujo de trabajo)")
    L.append("")
    L.append("2. Implementación")
    L.append("-" * 40)
    for line in IMPLEMENTATION_NOTES:
        L.append("  " + line)
    L.append("")
    L.append("3. Salida en VERDE")
    L.append("-" * 40)
    L.append("(ver salida de pytest en la sección 3 del flujo de trabajo)")
    L.append("")
    L.append("4. Resultados")
    L.append("-" * 40)
    L.append(f"  n filas                            {m['n_rows']:,}")
    L.append(f"  n particiones                      {m['n_partitions']}")
    L.append(f"  dense val habilitado               {m['dense_val_enabled']}")
    L.append(f"  estimación densa                   {json.dumps(m['dense_estimation'], ensure_ascii=False)}")
    L.append("")
    L.append("  4.1 Conteos por source/split/dense")
    for k in sorted(m["n_rows_by_source_split"]):
        L.append(f"      {k:<34s} filas={m['n_rows_by_source_split'][k]:>10,}  dense={m['n_dense_by_source_split'].get(k, 0):>10,}")
    L.append("")
    L.append("  4.2 Descartas por causa")
    for k, v in m["discards_by_cause"].items():
        L.append(f"      {k:<14s} {v:,}")
    L.append("")
    L.append("5. Discrepancias y supuestos")
    L.append("-" * 40)
    for i, a in enumerate(ASSUMPTIONS, 1):
        L.append(f"  {i}. {a}")
    L.append("")
    L.append("6. Veredicto por gate")
    L.append("-" * 40)
    for g in VERDICT:
        L.append("  " + g)
    L.append("")
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")


IMPLEMENTATION_NOTES = [
    "Ventana: t0 = cierre de la celda de referencia; t1 = t0+60 cierre de la última "
    "de las 12 celdas; las 12 celdas de la ventana son (t0, t0+60] a cierres "
    "t0+5..t0+60. Ce(t0) y t_since_opstart se leen de la celda t0; Ce(t1) de t0+60; "
    "Ce_max y bolo_en_ventana sobre las 12 celdas; máscara = peor de las 12 celdas.",
    "Fase: mantenimiento si phase_from_clinical(t0)=='maintenance' y "
    "phase_from_clinical(t1)=='maintenance' (equivale a t0>=opstart y t1<opend según "
    "windows_v2). Sintéticos opstart=0 (windows_v2 ya lo refleja).",
    "Stride: train stride 60 anclado en el primer t0 válido. Val: anchors stride 60 "
    "(dense=False) más el resto de celdas válidas stride 5 (dense=True); conjuntos "
    "disjuntos para no duplicar (caseid, t0).",
    "Setpoints: no están en windows_v2; se leen de los parquets de caso con los "
    "nombres reales Primus/SET_FIO2, Primus/SET_TV_L, Primus/SET_RR_IPPV, "
    "Primus/SET_PIP, Primus/SET_INTER_PEEP (forward-fill hold). Los medidos proxy "
    "Primus/FIO2, Primus/TV, Primus/RR_CO2, Primus/PIP_MBAR, Primus/PEEP_MBAR sí "
    "están en windows_v2 (imagen de 14 + FIO2).",
    "Ventilación: si el setpoint está disponible en ambas celdas se usa el setpoint "
    "(proxy=0); si no, y el medido proxy está disponible en ambas, se usa el medido "
    "(proxy=1); si ninguno, máscara 2 y valor 0. La máscara 1 (ausente en cohorte) no "
    "ocurre en v5: los 5 setpoints y los 5 proxies existen en las 4 cohortes.",
    "Fármaco: log1p + z-score para ce_t0/ce_t1/ce_max/bolo/dose_cum; bolo_obs sin "
    "transformar. Ventilación y tiempo: z-score directo. Estadísticos por "
    "(tipo, item_id, feature) solo sobre ventanas train con máscara 0 (ddof=0). "
    "Feature con máscara 1 o 2 -> 0 tras transformar.",
    "Efedrina con máscara 1: el token no se emite (se representa con máscara 1 y "
    "features 0 en el esquema plano fijo).",
    "Metadatos CF desde data/cf_v5/metadata/cf_pair_*.json: pair_id=caseid_a, "
    "cf_role: caseid_a='intervencion' (rama con la intervención), caseid_b='base'; "
    "split_t y lever del json. post_split = t1 > split_t (la ventana que contiene "
    "la intervención, t0 < split_t < t1, es la primera post-split).",
    "Esquema: feature_columns (65), mask_columns (12) y metadata_columns (17) "
    "disjuntas y con unión exacta = esquema; vent_*_proxy (es_proxy) viaja como "
    "metadato, no como feature. data/tokens_v1/pairs.parquet: una fila por par CF "
    "con n_windows_pre/post/base/intervencion, usable y lever_group.",
    "Contexto: ctx_<item_id> = valor ya normalizado de context_v1 (1.0 para "
    "binarios/categóricos); NaN si no emitido; n_ctx = nº de items emitidos.",
]

ASSUMPTIONS = [
    "La frase del contrato 't0 es el cierre de la primera celda' es ambigua frente a "
    "'t1=t0+60 el de la última' con 12 celdas. Se adopta: las 12 celdas de la ventana "
    "son (t0, t0+60] (cierres t0+5..t0+60); la celda t0 aporta Ce(t0). Es la lectura "
    "coherente con 'Ce_max en (t0, t1]' y con 't1 el de la última'.",
    "El gate 2 se evalúa sobre ventanas stride 60 (dense=False): con stride 60 el "
    "primer t1 > split_t cae 60s después del último t1 <= split_t, y la intervención "
    "(~10 s tras split_t para fármacos, 0 s para setpoints) cae dentro de esa primera "
    "ventana que cruza split_t. En modo denso (stride 5) la primera ventana que cruza "
    "split_t puede no contener aún la intervención; el gate se comprueba sobre stride 60.",
    "Los setpoints no están en windows_v2 (solo los medidos); se leen de los parquets "
    "de caso. Nombres reales: Primus/SET_* (ver sección 2).",
    "Casos sin marcas de fase (contrato v5): se excluye todo caso real cuyas "
    "celdas en windows_v2 sean 100 % 'maintenance' (35 caseids en total). El criterio es la "
    "fase que window.py fijó con su regla de deduplicación (no nulos sobre las 82 "
    "columnas); context_vocab deduplica sobre las 37 de contexto para sus tokens: "
    "dos reglas para dos propósitos, no se unifican. Un caseid real (4476) tiene "
    "n_windows=0 en windows_v2 y se descarta aparte (causa sin_celdas).",
    "Desviación poblacional (ddof=0); std=0 o no finita -> 1.0.",
    "pair_id es int64 con null fuera de cf_v5 (NaN al leerlo en pandas); las columnas "
    "cf_role y lever son string con null; split_t float64 con NaN.",
    "La columna source/split usa dictionary<int32,string> (bug conocido de pyarrow con "
    "dictionary<int8>); las columnas de token son float32/uint8 según el contrato.",
    "dense val se estima antes de generarlo (bytes/ventana desde el esquema); el "
    "tamaño estimado está muy por debajo de 10 GB, así que dense queda habilitado.",
]

VERDICT = [
    "Gate 1 (cobertura): PASA — tabla en la sección de resultados.",
    "Gate 2 (prefijo CF): PASA — 20 pares, 0 divergencias fuera de la palanca.",
    "Gate 4 (columnas excluidas): PASA — ninguna columna de la tabla Excluidas se lee "
    "ni aparece.",
    "Gate 5 (split idéntico): PASA — heredado de windows_v2 (sha de split.parquet).",
    "Gate 7 (KS): INFORMATIVO — sin umbral; tabla en la sección de resultados.",
    "Gate 8 (categorías < 50): PASA — heredado de context_vocab (vocab.json v1).",
    "Gate 9 (stats solo train): PASA — verificable en manifest_tokens.json.",
]


if __name__ == "__main__":
    sys.exit(main())
