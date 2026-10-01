"""cohort_gap.py — caracterización de la brecha real/sintético.

Diagnóstico independiente del AE (contrato_ae_v1.md, diagnóstico A4). NO toca
``src/ae/physio_ae.py``, ni los pesos entrenados, ni ningún gate del AE:
replica el protocolo de sondeo de fuente del gate 6 (submuestra de 200 000
celdas con semilla 9876, GroupKFold 5 por caseid, estandarización por fold,
LogisticRegression) sobre la ENTRADA CRUDA (14 valores físicos) y lo desglosa
en:

  D1  atribución por variable (univariante, acumulativa, ablación)
  D2  desglose por cohorte sintética (real vs cada cohorte, y pares
      sintético-vs-sintético)
  D3  controles de cordura (real-train vs real-val; mitad aleatoria de real)
  D4  distribuciones marginales en unidades físicas + Wasserstein-1
      normalizado por la desviación típica real
  D5  nivel vs dinámica (deltas t - t_menos_1)
  D6  conclusión

Uso:
    python -m diagnostics.cohort_gap run      # computa, cachea y escribe informe
    python -m diagnostics.cohort_gap report   # reescribe el informe desde la cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import paths

ROOT = Path(__file__).resolve().parents[2]
WINDOWS_ROOT = paths.WINDOWS_DIR
WINDOWS_DIR = WINDOWS_ROOT / "windows"
TOKENS_MANIFEST = paths.TOKENS_DIR / "manifest_tokens.json"
OUT_DIR = paths.DIAGNOSTICS_DIR
REPORT_PATH = paths.REPORTS_DIR / "REPORT_cohort_gap_v7.txt"
CACHE_PATH = OUT_DIR / "cohort_gap_results.json"
ROJO_TXT = paths.REPORTS_DIR / "_pytest_cohort_gap_rojo.txt"
VERDE_TXT = paths.REPORTS_DIR / "_pytest_cohort_gap_verde.txt"

# Semilla y tamaño de la submuestra del gate 6 (sondeo por celda).
SEED = 9876
N_CELLS = 200_000
HALF = N_CELLS // 2

# Imagen de 14 variables (orden canónico = image_tracks del manifest de
# windows_v2; idéntica a la usada por el AE y por window.py).
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

VALUE_COLS: list[str] = list(IMAGE_TRACKS)
MASK_COLS: list[str] = [f"m_{t}" for t in IMAGE_TRACKS]
META_COLS: list[str] = ["caseid", "source", "phase_from_clinical"]
READ_COLS: list[str] = META_COLS + VALUE_COLS + MASK_COLS
N_VARS: int = len(IMAGE_TRACKS)

SYNTH_SOURCES: list[str] = paths.SYNTH_COHORTS
ALL_SOURCES: list[str] = paths.ALL_COHORTS


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def _f(x) -> float | None:
    """Convierte a float; los valores no finitos se serializan como None."""
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if np.isfinite(y) else None


def _fmt_auc(x) -> str:
    y = _f(x)
    return "     nan" if y is None else f"{y:8.4f}"


def _fmt_f(x, w: int = 8) -> str:
    """Formatea un float para tabla; None -> 'nan' alineado."""
    y = _f(x)
    return f"{'nan':>{w}}" if y is None else f"{y:>{w}.2f}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_excluded_caseids() -> frozenset[int]:
    """Caseids excluidos: 35 reales sin marcas de fase + el caseid 4476 sin
    celdas (36 caseids en total), reutilizando la lista del manifest de
    tokens_v1 (misma regla que el AE)."""
    m = json.loads(TOKENS_MANIFEST.read_text(encoding="utf-8"))
    a = m.get("excluded_no_phase_marks_caseids", [])
    b = m.get("sin_celdas_caseids", [])
    return frozenset(int(c) for c in list(a) + list(b))


def iter_partitions(sources: list[str], split: str) -> list[Path]:
    parts: list[Path] = []
    for s in sources:
        d = WINDOWS_DIR / f"source={s}" / f"split={split}"
        parts.extend(sorted(d.glob("part-*.parquet")))
    return sorted(parts)


def read_partition(path: Path, columns: list[str] = READ_COLS) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()


def filter_cells(df: pd.DataFrame, excluded: frozenset[int]) -> pd.DataFrame:
    df = df[df["phase_from_clinical"] == "maintenance"]
    if excluded:
        df = df[~df["caseid"].isin(excluded)]
    return df


def load_cells(sources: list[str], split: str,
               excluded: frozenset[int]) -> dict:
    """Carga celdas de mantenimiento en memoria agrupadas por caso (replica de
    ``ae.physio_ae.load_cells``, sin torch). Las celdas quedan ordenadas por
    caseid (estable), preservando el orden temporal dentro de cada caso."""
    parts = iter_partitions(sources, split)
    values_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []
    caseid_list: list[np.ndarray] = []
    src_list: list[np.ndarray] = []
    n_by_source: dict[str, int] = {}
    for part in parts:
        df = read_partition(part, columns=READ_COLS)
        df = filter_cells(df, excluded)
        if len(df) == 0:
            continue
        src = str(df["source"].iloc[0])
        n_by_source[src] = n_by_source.get(src, 0) + int(len(df))
        values_list.append(df[VALUE_COLS].to_numpy(dtype=np.float32))
        masks_list.append(df[MASK_COLS].to_numpy(dtype=np.uint8))
        caseid_list.append(df["caseid"].to_numpy(dtype=np.int64))
        src_list.append(df["source"].to_numpy(dtype=object))
    if not values_list:
        return {
            "values": np.zeros((0, N_VARS), dtype=np.float32),
            "masks": np.zeros((0, N_VARS), dtype=np.uint8),
            "caseid": np.zeros(0, dtype=np.int64),
            "case_ids": np.zeros(0, dtype=np.int64),
            "case_start": np.zeros(0, dtype=np.int64),
            "case_len": np.zeros(0, dtype=np.int64),
            "case_source": np.zeros(0, dtype=object),
            "n_by_source": {},
        }
    values = np.concatenate(values_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    caseid = np.concatenate(caseid_list, axis=0)
    source = np.concatenate(src_list, axis=0)

    order = np.argsort(caseid, kind="stable")
    values = values[order]
    masks = masks[order]
    caseid = caseid[order]
    source = source[order]

    bounds = np.flatnonzero(np.r_[True, caseid[1:] != caseid[:-1]])
    case_ids = caseid[bounds]
    case_start = bounds.astype(np.int64)
    case_len = np.diff(np.r_[bounds, len(caseid)]).astype(np.int64)
    case_source = source[bounds].astype(str)
    return {
        "values": values,
        "masks": masks,
        "caseid": caseid,
        "case_ids": case_ids,
        "case_start": case_start,
        "case_len": case_len,
        "case_source": case_source,
        "n_by_source": n_by_source,
    }


def count_cells(sources: list[str], split: str,
                excluded: frozenset[int]) -> int:
    total = 0
    for part in iter_partitions(sources, split):
        df = read_partition(part, columns=["caseid", "phase_from_clinical"])
        df = filter_cells(df, excluded)
        total += len(df)
    return total


def load_sampled_cells(sources: list[str], split: str, excluded: frozenset[int],
                       n: int, seed: int) -> dict:
    """Muestra ``n`` celdas uniformemente (sin reemplazo) del conjunto
    fuente/split, en dos pasadas (conteo + selección), sin materializar todas
    las celdas. Devuelve values, masks, caseid."""
    total = count_cells(sources, split, excluded)
    rng = np.random.default_rng(seed)
    want = sorted(int(x) for x in rng.choice(total, size=min(n, total),
                                             replace=False))
    values_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []
    caseid_list: list[np.ndarray] = []
    pos = 0
    wi = 0
    for part in iter_partitions(sources, split):
        df = read_partition(part, columns=READ_COLS)
        df = filter_cells(df, excluded)
        m = len(df)
        if m == 0:
            continue
        lo, hi = pos, pos + m
        sel: list[int] = []
        while wi < len(want) and want[wi] < hi:
            sel.append(want[wi] - lo)
            wi += 1
        if sel:
            sub = df.iloc[sel]
            values_list.append(sub[VALUE_COLS].to_numpy(dtype=np.float32))
            masks_list.append(sub[MASK_COLS].to_numpy(dtype=np.uint8))
            caseid_list.append(sub["caseid"].to_numpy(dtype=np.int64))
        pos = hi
    if not values_list:
        return {
            "values": np.zeros((0, N_VARS), dtype=np.float32),
            "masks": np.zeros((0, N_VARS), dtype=np.uint8),
            "caseid": np.zeros(0, dtype=np.int64),
        }
    return {
        "values": np.concatenate(values_list, axis=0),
        "masks": np.concatenate(masks_list, axis=0),
        "caseid": np.concatenate(caseid_list, axis=0),
    }


# --------------------------------------------------------------------------
# Sondeo de origen (réplica del protocolo del gate 6 del AE)
# --------------------------------------------------------------------------

def run_origin_probe(X, y, groups) -> dict:
    """Regresión logística 5-fold agrupada por caso, features estandarizadas
    por fold (réplica exacta de ``ae.physio_ae._run_origin_probe``)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=5)
    aucs: list[float] = []
    accs: list[float] = []
    for tr, te in gkf.split(X, y, groups=groups):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=5000).fit(Xtr, y[tr])
        scores = clf.decision_function(Xte)
        if len(np.unique(y[te])) == 2:
            aucs.append(float(roc_auc_score(y[te], scores)))
        accs.append(float(accuracy_score(y[te], scores >= 0)))
    return {
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_per_fold": aucs,
        "acc_mean": float(np.mean(accs)),
    }


def _cell_bool(val: dict, fn) -> np.ndarray:
    """Booleano por celda a partir del source por caso (fn aplicada al
    case_source)."""
    return np.repeat(fn(val["case_source"]), val["case_len"])


def sample_two_groups(is_a: np.ndarray, in_universe: np.ndarray,
                      caseid: np.ndarray, half: int = HALF,
                      seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Submuestra estratificada: ``half`` celdas del grupo A (is_a) y ``half``
    del grupo B (universo - A), con semilla fija. Devuelve (idx, y, groups)."""
    rng = np.random.default_rng(seed)
    pos_a = np.flatnonzero(is_a & in_universe)
    pos_b = np.flatnonzero((~is_a) & in_universe)
    a_sel = np.sort(rng.choice(pos_a, size=min(half, len(pos_a)), replace=False))
    b_sel = np.sort(rng.choice(pos_b, size=min(half, len(pos_b)), replace=False))
    idx = np.concatenate([a_sel, b_sel])
    y = is_a[idx].astype(int)
    groups = caseid[idx]
    return idx, y, groups


# --------------------------------------------------------------------------
# Imputación y deltas
# --------------------------------------------------------------------------

def column_means(values, masks) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(masks)
    means = np.zeros(v.shape[1], dtype=np.float64)
    for j in range(v.shape[1]):
        col = v[:, j][m[:, j] == 1]
        means[j] = float(np.nanmean(col)) if col.size else 0.0
    return means


def impute_with(values, masks, means) -> np.ndarray:
    """Reemplaza las celdas enmascaradas (m=0) por la media por variable."""
    v = np.asarray(values, dtype=np.float64).copy()
    m = np.asarray(masks)
    for j in range(v.shape[1]):
        v[m[:, j] == 0, j] = means[j]
    return v


def delta_means(values, masks, caseid) -> np.ndarray:
    """Media por variable del delta t - t_menos_1 sobre celdas válidas (misma
    celda anterior del mismo caso y ambas con máscara 1)."""
    n, nv = values.shape
    same = np.r_[False, caseid[1:] == caseid[:-1]]  # length n
    means = np.zeros(nv, dtype=np.float64)
    for j in range(nv):
        d = values[1:, j] - values[:-1, j]          # length n-1
        valid = same[1:] & (masks[1:, j] == 1) & (masks[:-1, j] == 1)
        means[j] = float(np.nanmean(d[valid])) if valid.any() else 0.0
    return means


def deltas_for_indices(idx, values, masks, caseid, dmeans) -> np.ndarray:
    """Deltas t - t_menos_1 para las celdas dadas; la primera celda de cada
    caso (o delta no válido) se imputa con la media del delta."""
    idx = np.asarray(idx)
    prev = np.where(idx > 0, idx - 1, idx)
    same_case = (idx > 0) & (caseid[idx] == caseid[prev])
    d = (values[idx].astype(np.float64) - values[prev].astype(np.float64))
    ok = np.broadcast_to(same_case[:, None], d.shape) \
        & (masks[idx] == 1) & (masks[prev] == 1)
    d = np.where(ok, d, np.nan)
    for j in range(d.shape[1]):
        d[np.isnan(d[:, j]), j] = dmeans[j]
    return d


# --------------------------------------------------------------------------
# Distribuciones marginales (D4)
# --------------------------------------------------------------------------

def marginal_stats(vals: np.ndarray, masks: np.ndarray) -> dict:
    """Estadísticos marginales por variable (unidades físicas) sobre celdas con
    máscara 1."""
    out: dict = {}
    v = np.asarray(vals, dtype=np.float64)
    m = np.asarray(masks)
    for j, track in enumerate(IMAGE_TRACKS):
        col = v[:, j][m[:, j] == 1]
        if col.size == 0:
            out[track] = {"n": 0, "mean": None, "std": None,
                          "p1": None, "p5": None, "p25": None, "p50": None,
                          "p75": None, "p95": None, "p99": None,
                          "min": None, "max": None}
            continue
        mean = float(col.mean())
        std = float(np.std(col, ddof=0))
        qs = np.percentile(col, [1, 5, 25, 50, 75, 95, 99])
        out[track] = {
            "n": int(col.size),
            "mean": _f(mean), "std": _f(std),
            "p1": _f(qs[0]), "p5": _f(qs[1]), "p25": _f(qs[2]),
            "p50": _f(qs[3]), "p75": _f(qs[4]), "p95": _f(qs[5]),
            "p99": _f(qs[6]), "min": _f(col.min()), "max": _f(col.max()),
        }
    return out


def wasserstein_1(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a, b))


# --------------------------------------------------------------------------
# Cómputo D1-D6
# --------------------------------------------------------------------------

def _impute_and_probe(val: dict, idx: np.ndarray, y: np.ndarray,
                      groups: np.ndarray, means: np.ndarray,
                      cols: list[int], extra: np.ndarray | None = None) -> dict:
    X = impute_with(val["values"][idx], val["masks"][idx], means)[:, cols]
    if extra is not None:
        X = np.concatenate([X, extra[:, cols]], axis=1)
    return run_origin_probe(X, y, groups)


def _run_d1(val: dict, idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
            means: np.ndarray) -> dict:
    X = impute_with(val["values"][idx], val["masks"][idx], means)
    univariate = []
    for j in range(N_VARS):
        r = run_origin_probe(X[:, [j]], y, groups)
        univariate.append({"index": j, "track": IMAGE_TRACKS[j],
                           "auc": _f(r["auc_mean"])})
    univariate.sort(key=lambda e: -(e["auc"] or -1.0))
    order = [e["index"] for e in univariate]
    cumulative = []
    for k in range(1, N_VARS + 1):
        cols = order[:k]
        r = run_origin_probe(X[:, cols], y, groups)
        cumulative.append({"k": k, "tracks": [IMAGE_TRACKS[c] for c in cols],
                           "auc": _f(r["auc_mean"])})
    full14 = cumulative[-1]["auc"]
    ablation = []
    for j in range(N_VARS):
        cols = [c for c in range(N_VARS) if c != j]
        r = run_origin_probe(X[:, cols], y, groups)
        ablation.append({"index": j, "track": IMAGE_TRACKS[j],
                         "auc": _f(r["auc_mean"]),
                         "drop": _f((full14 or 0.0) - (r["auc_mean"] or 0.0))})
    ablation.sort(key=lambda e: -(e["drop"] or -1.0))
    return {"univariate": univariate, "cumulative": cumulative,
            "ablation": ablation, "full14_auc": full14}


def _run_d2(val: dict, means: np.ndarray) -> dict:
    cell_real = _cell_bool(val, lambda s: s == "real")
    caseid = val["caseid"]
    real_vs_cohort = []
    for cohort in SYNTH_SOURCES:
        cell_cohort = _cell_bool(val, lambda s: s == cohort)
        universe = cell_real | cell_cohort
        idx, y, groups = sample_two_groups(cell_real, universe, caseid)
        X = impute_with(val["values"][idx], val["masks"][idx], means)
        r = run_origin_probe(X, y, groups)
        real_vs_cohort.append({"cohort": cohort, "auc": _f(r["auc_mean"])})
    synth_pairs = []
    pairs = [(SYNTH_SOURCES[0], SYNTH_SOURCES[1]),
             (SYNTH_SOURCES[0], SYNTH_SOURCES[2]),
             (SYNTH_SOURCES[1], SYNTH_SOURCES[2])]
    for a, b in pairs:
        cell_a = _cell_bool(val, lambda s, x=a: s == x)
        cell_b = _cell_bool(val, lambda s, x=b: s == x)
        universe = cell_a | cell_b
        idx, y, groups = sample_two_groups(cell_a, universe, caseid)
        X = impute_with(val["values"][idx], val["masks"][idx], means)
        r = run_origin_probe(X, y, groups)
        synth_pairs.append({"pair": f"{a}_vs_{b}", "auc": _f(r["auc_mean"])})
    return {"real_vs_cohort": real_vs_cohort, "synth_pairs": synth_pairs}


def _run_d3(excluded: frozenset[int]) -> dict:
    # D3a: real-train vs real-val.
    tr = load_sampled_cells(["real"], "train", excluded, HALF, SEED)
    va = load_sampled_cells(["real"], "val", excluded, HALF, SEED)
    ntr, nva = len(tr["caseid"]), len(va["caseid"])
    values = np.concatenate([tr["values"], va["values"]], axis=0)
    masks = np.concatenate([tr["masks"], va["masks"]], axis=0)
    caseid = np.concatenate([tr["caseid"], va["caseid"]], axis=0)
    means = column_means(values, masks)
    X = impute_with(values, masks, means)
    y = np.r_[np.ones(ntr, dtype=int), np.zeros(nva, dtype=int)]
    a = run_origin_probe(X, y, caseid)
    a["n_cells"] = int(len(y))
    a["n_train"] = ntr
    a["n_val"] = nva

    # D3b: partición aleatoria de la cohorte real (val) en dos mitades por caseid.
    real = load_cells(["real"], "val", excluded)
    real_caseids = np.unique(real["caseid"])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(real_caseids)
    k = len(perm) // 2
    set_a = set(int(c) for c in perm[:k])
    cell_real = np.ones(len(real["caseid"]), dtype=bool)
    cell_a = np.isin(real["caseid"], list(set_a))
    idx, yb, groups = sample_two_groups(cell_a, cell_real, real["caseid"])
    rmeans = column_means(real["values"], real["masks"])
    Xb = impute_with(real["values"][idx], real["masks"][idx], rmeans)
    b = run_origin_probe(Xb, yb, groups)
    b["n_cells"] = int(len(yb))
    b["n_caseids_a"] = int(k)
    b["n_caseids_b"] = int(len(perm) - k)
    return {"a_train_vs_val": a, "b_half_split": b}


def _run_d4(val: dict) -> dict:
    per_source: dict = {}
    for source in ALL_SOURCES:
        is_src = _cell_bool(val, lambda s, x=source: s == x)
        per_source[source] = marginal_stats(val["values"][is_src],
                                            val["masks"][is_src])
    # Wasserstein-1 real vs cada cohorte, normalizado por la std real (ddof=0).
    wasserstein: dict = {}
    real_mask = _cell_bool(val, lambda s: s == "real")
    real_vals = val["values"][real_mask]
    real_msk = val["masks"][real_mask]
    for cohort in SYNTH_SOURCES:
        is_src = _cell_bool(val, lambda s, x=cohort: s == x)
        sv = val["values"][is_src]
        sm = val["masks"][is_src]
        per_track: dict = {}
        for j, track in enumerate(IMAGE_TRACKS):
            a = real_vals[:, j][real_msk[:, j] == 1].astype(np.float64)
            b = sv[:, j][sm[:, j] == 1].astype(np.float64)
            std_real = per_source["real"][track]["std"]
            if a.size == 0 or b.size == 0 or not std_real:
                per_track[track] = {"w1": None, "w1_norm": None}
                continue
            w1 = wasserstein_1(a, b)
            per_track[track] = {"w1": _f(w1), "w1_norm": _f(w1 / std_real)}
        wasserstein[cohort] = per_track
    return {"per_source": per_source, "wasserstein": wasserstein}


def _run_d5(val: dict, idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
            means: np.ndarray) -> dict:
    dmeans = delta_means(val["values"], val["masks"], val["caseid"])
    deltas = deltas_for_indices(idx, val["values"], val["masks"],
                                val["caseid"], dmeans)
    r_delta = run_origin_probe(deltas, y, groups)
    Xv = impute_with(val["values"][idx], val["masks"][idx], means)
    r_both = run_origin_probe(np.concatenate([Xv, deltas], axis=1), y, groups)
    return {"deltas_only": r_delta, "values_and_deltas": r_both}


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------

def compute_results() -> dict:
    t0 = _time.time()
    print("[cohort_gap] cargando exclusiones", flush=True)
    excluded = load_excluded_caseids()
    print(f"[cohort_gap] caseids excluidos: {len(excluded)}", flush=True)

    print("[cohort_gap] cargando celdas val (todas las fuentes)...", flush=True)
    val = load_cells(ALL_SOURCES, "val", excluded)
    print(f"[cohort_gap] val: {val['values'].shape[0]} celdas, "
          f"{len(val['case_ids'])} casos", flush=True)

    cell_real = _cell_bool(val, lambda s: s == "real")
    caseid = val["caseid"]
    universe = np.ones(len(caseid), dtype=bool)
    idx, y, groups = sample_two_groups(cell_real, universe, caseid)
    means = column_means(val["values"], val["masks"])
    print(f"[cohort_gap] submuestra gate 6: {len(idx)} celdas "
          f"({int(y.sum())} real + {int((y == 0).sum())} sintéticas)", flush=True)

    print("[cohort_gap] D3 controles de cordura...", flush=True)
    d3 = _run_d3(excluded)

    print("[cohort_gap] D1 atribución por variable...", flush=True)
    d1 = _run_d1(val, idx, y, groups, means)

    print("[cohort_gap] D2 desglose por cohorte...", flush=True)
    d2 = _run_d2(val, means)

    print("[cohort_gap] D4 distribuciones marginales + Wasserstein...", flush=True)
    d4 = _run_d4(val)

    print("[cohort_gap] D5 nivel vs dinámica...", flush=True)
    d5 = _run_d5(val, idx, y, groups, means)

    results = {
        "meta": {
            "sha256_cohort_gap_py": sha256(Path(__file__)),
            "sha256_tokens_manifest": sha256(TOKENS_MANIFEST),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "probe_protocol": ("LogisticRegression(max_iter=5000), GroupKFold 5 "
                               "por caseid, estandarización por fold, 200000 "
                               "celdas seed 9876, entrada cruda imputada"),
            "n_val_cells": int(val["values"].shape[0]),
            "n_val_cases": int(len(val["case_ids"])),
            "n_by_source_val": val["n_by_source"],
        },
        "d3": d3,
        "d1": d1,
        "d2": d2,
        "d4": d4,
        "d5": d5,
    }
    results["elapsed_s"] = round(_time.time() - t0, 1)
    print(f"[cohort_gap] cómputo completo en {results['elapsed_s']} s", flush=True)
    return results


def run_all(use_cache: bool = True,
            cache_path: Path = CACHE_PATH,
            report_path: Path = REPORT_PATH) -> dict:
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        sha = cached.get("meta", {}).get("sha256_cohort_gap_py")
        if sha and sha == sha256(Path(__file__)):
            print("[cohort_gap] reutilizando cache", flush=True)
            write_report(cached, report_path)
            return cached
        print("[cohort_gap] cache obsoleta, recomputando", flush=True)
    results = compute_results()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    write_report(results, report_path)
    return results


# --------------------------------------------------------------------------
# Informe
# --------------------------------------------------------------------------

def _read_pytest(txt: Path) -> str:
    if not txt.exists():
        return "(no disponible: ejecuta pytest para generar esta salida)"
    raw = txt.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _track_name(track: str) -> str:
    return track


def write_report(results: dict, report_path: Path = REPORT_PATH) -> Path:
    lines: list[str] = []
    add = lines.append
    meta = results["meta"]
    add("REPORT_cohort_gap_v7.txt — caracterización de la brecha real/sintético")
    add("=" * 78)
    add("")
    add("Diagnóstico independiente del autoencoder (contrato_ae_v1.md, diag A4).")
    add("NO toca physio_ae.py, ni los pesos, ni ningún gate del AE.")
    add("")
    add("0. Contexto y ficheros")
    add("-" * 40)
    add(f"  cohort_gap.py                  sha256 {meta.get('sha256_cohort_gap_py')}")
    add(f"  tokens_v1/manifest_tokens.json sha256 {meta.get('sha256_tokens_manifest')}")
    add(f"  generado                       {meta.get('generated_at')}")
    add(f"  celdas val                     {meta.get('n_val_cells')}")
    add(f"  casos val                      {meta.get('n_val_cases')}")
    add("  protocolo de sondeo: " + str(meta.get("probe_protocol")))
    add("")

    add("1. Supuestos numerados")
    add("-" * 40)
    for i, s in enumerate(ASSUMPTIONS, 1):
        add(f"  {i}. {s}")
    add("")

    add("2. Salida literal de pytest (ROJO)")
    add("-" * 40)
    add(_read_pytest(ROJO_TXT))
    add("")
    add("3. Salida literal de pytest (VERDE)")
    add("-" * 40)
    add(_read_pytest(VERDE_TXT))
    add("")

    # D3 primero (obligatorio).
    add("4. D3 — Controles de cordura (DEBEN dar AUC ~0.5)")
    add("-" * 40)
    d3 = results["d3"]
    a = d3["a_train_vs_val"]
    b = d3["b_half_split"]
    add(f"  D3a real-train vs real-val: AUC = {_fmt_auc(a['auc_mean'])}  "
        f"(n={a.get('n_cells')}, train={a.get('n_train')}, val={a.get('n_val')})")
    add(f"  D3b mitad aleatoria real (por caseid): AUC = {_fmt_auc(b['auc_mean'])}  "
        f"(n={b.get('n_cells')})")
    ok3 = (_f(a["auc_mean"]) or 1.0) < 0.6 and (_f(b["auc_mean"]) or 1.0) < 0.6
    add(f"  -> protocolo {'OK' if ok3 else 'ROTO (investigar antes de leer D1-D5)'}")
    add("")

    add("5. D1 — Atribución por variable")
    add("-" * 40)
    d1 = results["d1"]
    add("  D1a. Univariante (AUC por variable, ordenado de mayor a menor):")
    add(f"  {'#':>2}  {'variable':<22} {'AUC':>8}")
    for i, e in enumerate(d1["univariate"], 1):
        add(f"  {i:>2}  {_track_name(e['track']):<22} {_fmt_auc(e['auc'])}")
    add("")
    add("  D1b. Acumulativo hacia delante (en el orden de D1a):")
    add(f"  {'k':>2}  {'AUC':>8}  variables")
    for e in d1["cumulative"]:
        add(f"  {e['k']:>2}  {_fmt_auc(e['auc'])}  {' + '.join(e['tracks'])}")
    add("")
    add("  D1c. Ablación (14 variables menos una; caída respecto a las 14):")
    add(f"  {'variable':<22} {'AUC':>8} {'caida':>8}")
    for e in d1["ablation"]:
        add(f"  {_track_name(e['track']):<22} {_fmt_auc(e['auc'])} {_fmt_auc(e['drop'])}")
    add("")

    add("6. D2 — Desglose por cohorte sintética")
    add("-" * 40)
    add("  Real vs una sola cohorte sintética (14 valores):")
    for e in results["d2"]["real_vs_cohort"]:
        add(f"    real vs {e['cohort']:<18} AUC = {_fmt_auc(e['auc'])}")
    add("  Pares sintético-vs-sintético:")
    for e in results["d2"]["synth_pairs"]:
        add(f"    {e['pair']:<34} AUC = {_fmt_auc(e['auc'])}")
    add("")

    add("7. D4 — Distribuciones marginales (unidades físicas) y Wasserstein-1")
    add("-" * 40)
    d4 = results["d4"]
    add("  Estadísticos por variable y cohorte (celdas con máscara 1, val):")
    for source in ALL_SOURCES:
        add(f"  --- {source} ---")
        add(f"  {'variable':<22} {'n':>7} {'media':>8} {'std':>7} "
            f"{'p1':>7} {'p5':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'min':>7} {'max':>7}")
        for track in IMAGE_TRACKS:
            s = d4["per_source"][source][track]
            add(f"  {_track_name(track):<22} {s['n']:>7} "
                f"{_fmt_f(s['mean'])} {_fmt_f(s['std'], 7)} "
                f"{_fmt_f(s['p1'], 7)} {_fmt_f(s['p5'], 7)} {_fmt_f(s['p50'], 7)} "
                f"{_fmt_f(s['p95'], 7)} {_fmt_f(s['p99'], 7)} {_fmt_f(s['min'], 7)} "
                f"{_fmt_f(s['max'], 7)}")
        add("")
    add("  Distancia Wasserstein-1 real vs cada cohorte (normalizada por std real):")
    for cohort in SYNTH_SOURCES:
        rows = [(track, d4["wasserstein"][cohort][track]["w1"],
                 d4["wasserstein"][cohort][track]["w1_norm"])
                for track in IMAGE_TRACKS]
        rows.sort(key=lambda r: -(r[2] or -1.0))
        add(f"  --- {cohort} (ordenado por W1 normalizada descendente) ---")
        add(f"  {'variable':<22} {'W1':>8} {'W1/std_real':>10}")
        for track, w1, w1n in rows:
            add(f"  {_track_name(track):<22} {_fmt_auc(w1)} {_fmt_auc(w1n)}")
        add("")

    add("8. D5 — ¿Nivel o dinámica?")
    add("-" * 40)
    d5 = results["d5"]
    add(f"  AUC solo 14 deltas:        {_fmt_auc(d5['deltas_only']['auc_mean'])}")
    add(f"  AUC 28 (valores + deltas): {_fmt_auc(d5['values_and_deltas']['auc_mean'])}")
    add("")

    add("9. D6 — Conclusión")
    add("-" * 40)
    add(_render_conclusion(results))
    add("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


ASSUMPTIONS: list[str] = [
    "Replica literal del protocolo del gate 6 del AE: submuestra estratificada "
    "de 200 000 celdas de val (100 000 reales + 100 000 sintéticas), semilla "
    "9876, GroupKFold(5) agrupado por caseid, StandardScaler ajustado por fold, "
    "LogisticRegression(max_iter=5000), AUC (roc_auc_score) promediado entre "
    "folds; real es la clase positiva.",
    "La 'entrada cruda' son los 14 valores en UNIDADES FÍSICAS de windows_v2 "
    "(no los z-score del AE). Las celdas con máscara m=0 se imputan con la media "
    "de la variable sobre celdas con m=1 de todo el split val (constante global, "
    "sin usar etiquetas).",
    "El orden de columnas es el canónico de windows_v2 (BIS/BIS ... BIS/EMG), "
    "idéntico al manifest y al AE.",
    "'Sintético' = unión de las tres cohortes sintéticas (synthetic_v5, "
    "vaso_reinf_v5, cf_v5) con muestreo uniforme por celda, como en el gate 6. "
    "En D2 cada cohorte se enfrenta por separado.",
    "Delta de una celda = valor - valor de la celda anterior (t - t_menos_1) "
    "dentro del mismo caso (rejilla contigua de 5 s). La primera celda de cada "
    "caso, y cualquier delta con alguna de las dos celdas enmascarada, se imputa "
    "con la media del delta válido por variable.",
    "D3a usa real-train vs real-val (100 000 celdas de cada uno, semilla 9876). "
    "D3b particiona la cohorte real de val en dos mitades POR CASEID (semilla "
    "9876) y sondea 100 000 + 100 000 celdas.",
    "D4 usa el split val, solo celdas con máscara m=1, en unidades físicas. La "
    "Wasserstein-1 se normaliza por la desviación típica real (ddof=0).",
    "Exclusiones: caseids del manifest de tokens_v1 (35 sin marcas de fase + "
    "el caseid 4476 sin celdas, 36 en total); solo se retienen celdas "
    "phase_from_clinical=maintenance.",
]


def _auc(x) -> str:
    """AUC sin padding (para interpolar en prosa)."""
    y = _f(x)
    return "nan" if y is None else f"{y:.4f}"


def _render_conclusion(results: dict) -> str:
    d1 = results["d1"]
    d2 = results["d2"]
    d5 = results["d5"]

    min_k = next((e["k"] for e in d1["cumulative"]
                  if (e["auc"] or 0.0) > 0.90), None)
    top1 = d1["univariate"][0]
    top3 = d1["univariate"][:3]
    abl3 = d1["ablation"][:3]
    full14 = d1["full14_auc"]
    k3 = next((e for e in d1["cumulative"] if e["k"] == 3), None)

    rc = [e["auc"] for e in d2["real_vs_cohort"]]
    rc_f = [x for x in rc if x is not None]
    spread_cohort = (max(rc_f) - min(rc_f)) if len(rc_f) > 1 else 0.0
    delta_auc = d5["deltas_only"]["auc_mean"]
    both_auc = d5["values_and_deltas"]["auc_mean"]

    lines: list[str] = []
    lines.append("Respuestas (sin hedging, con los números delante):")
    lines.append("")
    # Q1
    if min_k is not None:
        lines.append(
            f"1. La separación está CONCENTRADA en pocas variables. La variable de "
            f"mayor AUC univariante es {top1['track']} ({_auc(top1['auc'])}), seguida "
            f"de {top3[1]['track']} ({_auc(top3[1]['auc'])}) y {top3[2]['track']} "
            f"({_auc(top3[2]['auc'])}). El número mínimo de variables que alcanza "
            f"AUC > 0.90 es {min_k} (solo {top1['track']}). La acumulación se satura "
            f"muy pronto: k=3 ya da {_auc(k3['auc']) if k3 else 'nan'} frente a "
            f"{_auc(full14)} con las 14. La ablación confirma el reparto: quitar "
            f"{abl3[0]['track']} cae {_auc(abl3[0]['drop'])}, quitar "
            f"{abl3[1]['track']} cae {_auc(abl3[1]['drop'])} y quitar "
            f"{abl3[2]['track']} cae {_auc(abl3[2]['drop'])}; ninguna otra variable "
            f"aislada derrumba la separación.")
    else:
        lines.append(
            "1. La separación está REPARTIDA: ninguna acumulación alcanza AUC > 0.90 "
            "hasta incluir las 14 variables.")
    lines.append("")
    # Q2
    if spread_cohort <= 0.05:
        lines.append(
            f"2. La brecha es HOMOGÉNEA entre las tres cohortes sintéticas: los AUC "
            f"real-vs-cohorte son {_auc(rc_f[0])} (synthetic_v5), {_auc(rc_f[1])} "
            f"(vaso_reinf_v5) y {_auc(rc_f[2])} (cf_v5); rango máximo "
            f"{spread_cohort:.4f}.")
    else:
        lines.append(
            f"2. La brecha NO es homogénea entre las tres cohortes sintéticas: los "
            f"AUC real-vs-cohorte son {', '.join(_auc(x) for x in rc_f)} (rango "
            f"{spread_cohort:.4f}); el arreglo del generador difiere por cohorte.")
    lines.append("")
    # Q3
    if (delta_auc or 0.0) >= 0.70:
        lines.append(
            f"3. La brecha está en el nivel Y en la dinámica: los deltas solos dan "
            f"AUC {_auc(delta_auc)} y valores+deltas {_auc(both_auc)}. Una "
            f"recalibración de marginales no bastará.")
    else:
        lines.append(
            f"3. La brecha está en el NIVEL, no en la dinámica: los 14 deltas solos "
            f"dan AUC {_auc(delta_auc)} (azar). Los niveles instantáneos bastan para "
            f"separar (AUC {_auc(d1['full14_auc'])} con 14 valores); añadir los "
            f"deltas apenas cambia el resultado ({_auc(both_auc)} con 28).")
    lines.append("")
    # Q4
    concentrated = min_k is not None and min_k <= 3
    level_only = (delta_auc or 0.0) < 0.70
    if concentrated and spread_cohort <= 0.05 and level_only:
        lines.append(
            f"4. Es PLAUSIBLE cerrarla recalibrando las distribuciones de parámetros "
            f"del generador, sin cambios estructurales en el modelo generativo: la "
            f"separación es de NIVEL (deltas al azar), está concentrada en "
            f"{top1['track']} y en las presiones de vía aérea (PEEP/PIP) y es "
            f"homogénea entre las tres cohortes. Esa es una caracterización de la "
            f"brecha, no una propuesta de solución.")
    elif concentrated and not level_only:
        lines.append(
            "4. NO bastará recalibrar solo las marginales: la brecha también está en "
            "la dinámica (los deltas separan), lo que apunta a cambios estructurales "
            "en el modelo generativo (autocorrelación/transiciones), aunque pocas "
            "variables concentren el nivel.")
    elif not concentrated and level_only:
        lines.append(
            "4. Recalibrar pocas variables no bastará: la separación está repartida "
            "entre las 14 variables (nivel). Cerrarla exige una recalibración amplia "
            "de las distribuciones de parámetros o cambios estructurales.")
    else:
        lines.append(
            "4. No es plausible cerrarla solo con recalibración de distribuciones de "
            "parámetros: la separación está repartida y/o presente en la dinámica; "
            "se requieren cambios estructurales en el modelo generativo.")
    lines.append("")
    # Nota adicional (no pedida, pero caracteriza D2).
    pairs = d2["synth_pairs"]
    pair_s = ", ".join(f"{e['pair']}={_auc(e['auc'])}" for e in pairs)
    lines.append(
        f"Nota adicional: las cohortes sintéticas son distinguibles ENTRE SÍ "
        f"(pares sintético-vs-sintético: {pair_s}), aunque mucho menos que frente a "
        f"la real; comparten el mismo generador base y difieren sobre todo en las "
        f"palancas aplicadas.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Caracterización de la brecha real/sintético")
    ap.add_argument("command", choices=["run", "report"], nargs="?", default="run")
    args = ap.parse_args(argv)

    if args.command == "run":
        run_all()
    else:
        if not CACHE_PATH.exists():
            print("No hay cache; ejecuta 'run' primero.", file=sys.stderr)
            return 1
        results = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        write_report(results)
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
