"""v6_validate.py — validación de la recalibración v6 contra real_holdout (PASO 4).

RECONSTRUIDO el 2026-09-28 a partir de
``src/diagnostics/__pycache__/v6_validate.cpython-311.pyc``: el fichero fuente
original se perdió y Python no importa módulos pyc huérfanos de ``__pycache__``
(PEP 3147). El bytecode se desensambló y se reescribió función a función; el
comportamiento de cómputo (V1-V7) es idéntico al original. ÚNICO CAMBIO
deliberado: ``write_report`` ahora es PARAMETRIZADO (``cohort``) y la
conclusión se GENERA a partir de los números, no de texto fijo (PASO 3.2).

Valida las cohortes v6 frente a la partición real_holdout (60 % real_calib /
40 % real_holdout por caseid, semilla fija). Los gates:

  V1  Sonda LINEAL, 14 valores, real_holdout vs v6.
      Objetivo: AUC <= suelo P1a (0.6736) + 0.03.
  V2  Sonda NO LINEAL (HistGradientBoosting), 14 valores y 28
      (valores+deltas), real_holdout vs v6. Objetivo: AUC < 0.85.
  V3  Solapamiento de soportes: fracción de los 20 vecinos más cercanos de la
      otra cohorte en el espacio de 14 variables z-scoreadas (sin reentrenar
      el AE). Objetivo: media > 0.25, mediana > 0.10. Reporta también v5.
  V4  Repite D4 y P1b contra real_holdout (W1 normalizada y cuantización).
  V5  Gate 3 de pk_tokens (equivalencia integrador vs anessim) sobre casos v6:
      < 1 %.  (Bloqueante, prioridad sobre V1-V3.)
  V6  Pares contrafactuales cf_v6: divergencia entre ramas ANTES del split ~ 0.
  V7  Curvas dosis-efecto v6 vs v5 superpuestas (propofol->BIS, remi,
      vasoactivos->ART_MBP).

NO modifica el generador ni el AE. Solo pandas/numpy/pyarrow/sklearn.

Uso:
    python -m diagnostics.v6_validate run      # computa y escribe el informe
    python -m diagnostics.v6_validate report   # reescribe desde la cache
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

from diagnostics import cohort_gap as cg
from diagnostics import gap_addendum as ga

import paths

ROOT = Path(__file__).resolve().parents[2]
WINDOWS_REAL = paths.WINDOWS_DIR
WINDOWS_V6 = paths.WINDOWS_V3_DIR
AE_STATS = paths.AE_DIR / "ae_bal" / "norm_stats.json"
TOKENS_MANIFEST = paths.TOKENS_DIR / "manifest_tokens.json"
CF_V6_METADATA = paths.LOST_COHORT_DIRS["cf_v6"] / "metadata"
OUT_DIR = paths.DIAGNOSTICS_DIR
REPORT_PATH = paths.REPORTS_DIR / "REPORT_generator_validation.txt"
CACHE_PATH = OUT_DIR / "v6_validate_results.json"
ROJO_TXT = paths.REPORTS_DIR / "_pytest_v6_validate_rojo.txt"
VERDE_TXT = paths.REPORTS_DIR / "_pytest_v6_validate_verde.txt"

SEED = 9876
N_CELLS = 200_000
HALF = N_CELLS // 2
HOLDOUT_SEED = 20260922
HOLDOUT_FRAC = 0.4

SYNTH_SOURCES_V6: list[str] = paths.LOST_SYNTH_V6
SYNTH_SOURCES_V5: list[str] = paths.LOST_SYNTH_V5

ASSUMPTIONS: list[str] = [
    "La cohorte real se parte por caseid con semilla 20260922 en real_calib "
    "(60 %) y real_holdout (40 %). El holdout no se toca durante la "
    "recalibración y solo se consulta aquí, una única vez.",
    "V1 replica el protocolo del gate 6 (200 000 celdas, semilla 9876, "
    "GroupKFold 5 por caseid, StandardScaler por fold, LogisticRegression) "
    "sobre las 14 variables, real_holdout vs unión de las 3 cohortes v6.",
    "V2 usa HistGradientBoostingClassifier(max_iter=300) con el mismo "
    "protocolo; 14 valores y 28 (valores+deltas t - t_menos_1).",
    "V3 usa los 20 vecinos más cercanos (distancia EUCLÍDEA) en el espacio de "
    "14 variables z-scoreadas con las stats del AE (norm_stats.json de ae_bal), "
    "sobre una submuestra de 20 000 celdas por cohorte (semilla 9876).",
    "V5 delega en el gate 3 de tokens.pk_tokens (equivalencia de integrador "
    "frente a anessim) sobre casos de v6; se reporta el error estacionario.",
    "V6 compara las dos ramas de cada par cf_v6 en el PREFIJO (t <= split_t) "
    "usando los truth de los casos; la divergencia debe ser 0 o numéricamente "
    "despreciable.",
    "V7 agrega curvas dosis-efecto en bins de Ce/velocidad y compara v6 vs v5 "
    "con la misma rejilla de bins.",
]


def sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _f(x) -> float | None:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if np.isfinite(y) else None


def _fmt_auc(x) -> str:
    y = _f(x)
    return "     nan" if y is None else f"{y:8.4f}"


def _fmt_g(x, w: int = 8) -> str:
    y = _f(x)
    return f"{'nan':>{w}}" if y is None else f"{y:>{w}.4f}"


def split_real_caseids(excluded: frozenset[int],
                       seed: int = HOLDOUT_SEED,
                       frac: float = HOLDOUT_FRAC) -> tuple[list[int], list[int]]:
    """Partición determinista por caseid de la cohorte real (val)."""
    parts = cg.iter_partitions(["real"], "val")
    caseids: set[int] = set()
    for part in parts:
        df = pq.read_table(part, columns=["caseid", "phase_from_clinical"]).to_pandas()
        df = cg.filter_cells(df, excluded)
        caseids |= set(int(c) for c in df["caseid"].unique())
    caseids = sorted(caseids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(caseids)
    k = int(round(len(perm) * frac))
    holdout = sorted(int(c) for c in perm[:k])
    calib = sorted(int(c) for c in perm[k:])
    return calib, holdout


def load_cells_from(windows_dir: Path, sources: list[str], split: str,
                    excluded: frozenset[int]) -> dict:
    """Igual que cg.load_cells pero sobre un directorio de ventanas dado."""
    parts: list[Path] = []
    for s in sources:
        d = windows_dir / "windows" / f"source={s}" / f"split={split}"
        parts.extend(sorted(d.glob("part-*.parquet")))
    values_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []
    caseid_list: list[np.ndarray] = []
    src_list: list[np.ndarray] = []
    n_by_source: dict[str, int] = {}
    for part in sorted(parts):
        df = pq.read_table(part, columns=cg.READ_COLS).to_pandas()
        df = cg.filter_cells(df, excluded)
        if len(df) == 0:
            continue
        src = str(df["source"].iloc[0])
        n_by_source[src] = n_by_source.get(src, 0) + int(len(df))
        values_list.append(df[cg.VALUE_COLS].to_numpy(dtype=np.float32))
        masks_list.append(df[cg.MASK_COLS].to_numpy(dtype=np.uint8))
        caseid_list.append(df["caseid"].to_numpy(dtype=np.int64))
        src_list.append(df["source"].to_numpy(dtype=object))
    if not values_list:
        return {
            "values": np.zeros((0, cg.N_VARS), dtype=np.float32),
            "masks": np.zeros((0, cg.N_VARS), dtype=np.uint8),
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
    values, masks, caseid, source = (values[order], masks[order],
                                     caseid[order], source[order])

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


# --------------------------------------------------------------------------
# V1/V2 — sondas lineal y no lineal
# --------------------------------------------------------------------------

def _run_v1_v2(holdout: dict, synth: dict) -> dict:
    def merge(a: dict, b: dict) -> dict:
        return {
            "values": np.concatenate([a["values"], b["values"]], axis=0),
            "masks": np.concatenate([a["masks"], b["masks"]], axis=0),
            "caseid": np.concatenate([a["caseid"], b["caseid"]], axis=0),
        }

    merged = merge(holdout, synth)
    n_real = len(holdout["caseid"])
    y = np.r_[np.ones(n_real, dtype=int), np.zeros(len(synth["caseid"]), dtype=int)]
    means = cg.column_means(merged["values"], merged["masks"])
    idx = np.arange(len(y))
    rng = np.random.default_rng(SEED)
    pos_r = np.flatnonzero(y == 1)
    pos_s = np.flatnonzero(y == 0)
    sel_r = np.sort(rng.choice(pos_r, size=min(HALF, len(pos_r)), replace=False))
    sel_s = np.sort(rng.choice(pos_s, size=min(HALF, len(pos_s)), replace=False))
    idx = np.concatenate([sel_r, sel_s])
    y2 = y[idx]
    groups = merged["caseid"][idx]
    X14 = cg.impute_with(merged["values"][idx], merged["masks"][idx], means)
    v1 = cg.run_origin_probe(X14, y2, groups)
    v2_14 = ga.run_origin_probe_hgb(X14, y2, groups)

    dmeans = cg.delta_means(merged["values"], merged["masks"], merged["caseid"])
    deltas = cg.deltas_for_indices(idx, merged["values"], merged["masks"],
                                   merged["caseid"], dmeans)
    X28 = np.concatenate([X14, deltas], axis=1)
    v2_28 = ga.run_origin_probe_hgb(X28, y2, groups)
    return {
        "n_real": int(n_real),
        "n_synth": int(len(synth["caseid"])),
        "n_sample": int(len(idx)),
        "v1_linear_14": {"auc": _f(v1["auc_mean"]), "acc": v1["acc_mean"]},
        "v2_hgb_14": {"auc": _f(v2_14["auc_mean"]), "acc": v2_14["acc_mean"]},
        "v2_hgb_28": {"auc": _f(v2_28["auc_mean"]), "acc": v2_28["acc_mean"]},
    }


# --------------------------------------------------------------------------
# V3 — solapamiento de soportes (20-NN)
# --------------------------------------------------------------------------

def _load_ae_stats() -> dict[str, tuple[float, float]]:
    """Lee norm_stats.json del AE (ae_bal): {track: (mean, std)}."""
    stats: dict = {}
    if not AE_STATS.exists():
        return stats
    raw = json.loads(AE_STATS.read_text(encoding="utf-8"))
    for track, d in raw.items():
        mean = float(d.get("mean", 0.0))
        std = float(d.get("std", 1.0))
        stats[track] = (mean, std if (std and np.isfinite(std)) else 1.0)
    return stats


def _zscore_with(X: np.ndarray, stats: dict[str, tuple[float, float]]) -> np.ndarray:
    """Z-scorea las columnas de X (orden IMAGE_TRACKS) con las stats del AE."""
    Xz = X.astype(np.float64).copy()
    for j, track in enumerate(cg.IMAGE_TRACKS):
        mean, std = stats.get(track, (0.0, 1.0))
        Xz[:, j] = (Xz[:, j] - mean) / std
    return Xz


def nn_overlap(X_a: np.ndarray, X_b: np.ndarray, k: int = 20,
               rng_seed: int = SEED) -> dict:
    """Fracción de los k vecinos más cercanos (Euclídea) de la otra cohorte,
    en el espacio de 14 variables z-scoreadas con las stats del AE."""
    from sklearn.neighbors import NearestNeighbors

    stats = _load_ae_stats()
    rng = np.random.default_rng(rng_seed)
    na = min(len(X_a), 20000)
    nb = min(len(X_b), 20000)
    sel_a = rng.choice(len(X_a), na, replace=False)
    sel_b = rng.choice(len(X_b), nb, replace=False)
    Xa = _zscore_with(X_a[sel_a], stats)
    Xb = _zscore_with(X_b[sel_b], stats)
    Zall = np.vstack([Xa, Xb])
    labels = np.r_[np.ones(len(Xa), dtype=int), np.zeros(len(Xb), dtype=int)]

    nbr = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(Zall)
    _, ind = nbr.kneighbors(Zall)
    neigh = labels[ind[:, 1:]]
    fracs = np.mean(neigh != labels[:, None], axis=1)
    return {
        "mean": _f(float(np.mean(fracs))),
        "median": _f(float(np.median(fracs))),
        "n_a": int(len(Xa)),
        "n_b": int(len(Xb)),
    }


def _run_v3(holdout: dict, synth: dict, synth_v5: dict | None) -> dict:
    out: dict = {}
    pairs = [("v6", synth)]
    if synth_v5 is not None:
        pairs.append(("v5", synth_v5))
    for name, sv in pairs:
        rng = np.random.default_rng(SEED)
        ha = np.sort(rng.choice(
            len(holdout["values"]),
            min(25000, len(holdout["values"])), replace=False))
        sa = np.sort(rng.choice(
            len(sv["values"]),
            min(25000, len(sv["values"])), replace=False))
        Xa = cg.impute_with(holdout["values"][ha], holdout["masks"][ha],
                            cg.column_means(holdout["values"], holdout["masks"]))
        Xb = cg.impute_with(sv["values"][sa], sv["masks"][sa],
                            cg.column_means(sv["values"], sv["masks"]))
        out[name] = nn_overlap(Xa, Xb)
    return out


# --------------------------------------------------------------------------
# V4 — D4 + P1b (W1 normalizada y cuantización)
# --------------------------------------------------------------------------

def _run_v4(holdout: dict, synth: dict) -> dict:
    real_stats = cg.marginal_stats(holdout["values"], holdout["masks"])
    synth_stats = cg.marginal_stats(synth["values"], synth["masks"])
    wass: dict = {}
    for j, track in enumerate(cg.IMAGE_TRACKS):
        a = holdout["values"][:, j][holdout["masks"][:, j] == 1].astype(np.float64)
        b = synth["values"][:, j][synth["masks"][:, j] == 1].astype(np.float64)
        std_real = real_stats[track]["std"]
        if a.size == 0 or b.size == 0 or not std_real:
            wass[track] = {"w1_norm": None}
            continue
        w1 = cg.wasserstein_1(a, b)
        wass[track] = {"w1_norm": _f(w1 / std_real)}
    quant: dict = {}
    for j, track in enumerate(cg.IMAGE_TRACKS):
        quant[track] = ga.quantization_stats(synth["values"], synth["masks"],
                                             synth["caseid"], j)
    return {
        "w1_norm": wass,
        "real_stats": real_stats,
        "synth_stats": synth_stats,
        "synth_quant": quant,
    }


# --------------------------------------------------------------------------
# V5 — gate 3 de pk_tokens (equivalencia integrador vs anessim)
# --------------------------------------------------------------------------

def run_v5_pk_gate3() -> dict:
    """V5: gate 3 de pk_tokens — equivalencia del integrador (parte 1 del
    test_m) sobre casos de synthetic_v6. Compara el Ce del módulo (forma
    cerrada) frente al integrador RK4 iterativo de anessim SIN IIV, con la
    misma serie de RATE y bolos, en rejilla de 5 s. Objetivo < 1 %."""
    try:
        from anessim.pk.base import Infusion
        from anessim.pk.propofol import PropofolSchnider
        from anessim.pk.remifentanil import RemifentanilMinto
        from tokens import pk_tokens as pk
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}

    synth_v6 = paths.LOST_COHORT_DIRS["synthetic_v6"]
    meta_dir = synth_v6 / "metadata"
    cases = sorted((synth_v6 / "cases").glob("*.parquet"))
    syn_clin = pq.read_table(synth_v6 / "clinical_data.parquet").to_pandas()
    demo = {int(r.caseid): dict(weight=float(r.weight), age=float(r.age),
                                height=float(r.height), sex=str(r.sex))
            for _, r in syn_clin.iterrows()}

    def anessim_ce(model, grid, rate_hold, bolus_grid):
        rate_min = rate_hold / 3.0
        infs = []
        i = 1
        while i < len(grid):
            j = i
            while j + 1 < len(grid) and rate_min[j + 1] == rate_min[i]:
                j += 1
            if rate_min[i] != 0.0:
                infs.append(Infusion(start_s=grid[i - 1], end_s=grid[j] - 1e-9,
                                     rate=rate_min[i], drug=model.drug))
            i = j + 1
        boluses = [(grid[i - 1] if i > 0 else grid[0], float(bolus_grid[i]))
                   for i in np.where(bolus_grid > 0)[0]]
        st = model.simulate(infs, grid, boluses=boluses)
        return st[:, 3]

    prop_errs: list[float] = []
    remi_errs: list[float] = []
    n_cases = 0
    for case_path in cases[:20]:
        cid = int(case_path.stem)
        try:
            row = demo[cid]
        except KeyError:
            continue
        d = dict(weight=float(row["weight"]), age=float(row["age"]),
                 height=float(row["height"]), sex=str(row["sex"]))
        df = pq.read_table(case_path, columns=[
            "time", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE",
            "ppf_bolus_mg", "remi_bolus_ug"]).to_pandas()
        t = df["time"].to_numpy(float)
        grid = np.arange(t[0], t[-1], pk.DT_S)
        if len(grid) < 2:
            continue
        n_cases += 1
        models = [
            ("propofol", "Orchestra/PPF20_RATE", "ppf_bolus_mg",
             PropofolSchnider(weight_kg=d["weight"], age_y=d["age"],
                              height_cm=d["height"], sex=d["sex"])),
            ("remifentanilo", "Orchestra/RFTN20_RATE", "remi_bolus_ug",
             RemifentanilMinto(weight_kg=d["weight"], height_cm=d["height"],
                               age_y=d["age"], sex=d["sex"])),
        ]
        for drug, rate_col, bol_col, model in models:
            rate_hold = pk._forward_fill_hold(t, df[rate_col].to_numpy(float), grid)
            bolus = pk._bolus_on_grid(t, df[bol_col].to_numpy(float), grid)
            ce_my = pk.compute_ce(drug, rate_hold, bolus, d, dt_s=pk.DT_S)
            ce_an = anessim_ce(model, grid, rate_hold, bolus)
            ok = ce_an > 0.1
            if ok.sum() > 10:
                err = np.abs(ce_my[ok] - ce_an[ok]) / ce_an[ok]
                (prop_errs if drug == "propofol" else remi_errs).extend(err.tolist())
    mean_prop = float(np.mean(prop_errs)) if prop_errs else None
    mean_remi = float(np.mean(remi_errs)) if remi_errs else None
    return {
        "n_cases": n_cases,
        "propofol_err_mean": _f(mean_prop),
        "remifentanilo_err_mean": _f(mean_remi),
        "pass": (mean_prop is not None and mean_remi is not None
                 and mean_prop < 0.01 and mean_remi < 0.01),
    }


# --------------------------------------------------------------------------
# V6 — divergencia de prefijo CF
# --------------------------------------------------------------------------

def _read_truth(caseid: int, split_t: float) -> np.ndarray | None:
    path = paths.LOST_COHORT_DIRS["cf_v6"] / "truth" / f"{caseid}_truth.parquet"
    if not path.exists():
        return None
    try:
        df = pq.read_table(path).to_pandas()
    except Exception:
        return None
    cols = [c for c in ("bis", "map", "hr") if c in df.columns]
    if not cols or "time" not in df.columns:
        return None
    pre = df[df["time"] <= split_t][cols].to_numpy(dtype=np.float64)
    return pre


def run_v6_cf_prefix() -> dict:
    """Compara las ramas de cada par cf_v6 antes del split."""
    pairs_dir = CF_V6_METADATA
    if not pairs_dir.exists():
        return {"error": f"no existe {pairs_dir}"}
    diffs: list[float] = []
    n_pairs = 0
    for p in sorted(pairs_dir.glob("cf_pair_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        caseid_a = int(m.get("caseid_a", m.get("caseid_intervencion", -1)))
        caseid_b = int(m.get("caseid_b", m.get("caseid_base", -1)))
        split_t = float(m.get("split_t", 0.0))
        ta = _read_truth(caseid_a, split_t)
        tb = _read_truth(caseid_b, split_t)
        if ta is None or tb is None:
            continue
        n_pairs += 1
        diffs.append(float(np.max(np.abs(ta - tb))))
    diffs = np.array(diffs)
    return {
        "n_pairs": n_pairs,
        "prefix_max_abs_diff": {
            "max": _f(float(diffs.max())) if diffs.size else None,
            "p50": _f(float(np.median(diffs))) if diffs.size else None,
            "p99": _f(float(np.percentile(diffs, 99))) if diffs.size else None,
        },
    }


# --------------------------------------------------------------------------
# V7 — curvas dosis-efecto
# --------------------------------------------------------------------------

def dose_effect_curve(rates: np.ndarray, response: np.ndarray,
                      bins: int = 20) -> dict:
    """Media de la respuesta por bins de la dosis (misma rejilla)."""
    if rates.size == 0 or response.size == 0:
        return {"x": [], "y": []}
    edges = np.percentile(rates, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)
    idx = np.digitize(rates, edges)
    xs: list[float] = []
    ys: list[float] = []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() >= 10:
            xs.append(float((edges[b - 1] + edges[b]) / 2.0))
            ys.append(float(np.mean(response[m])))
    return {"x": xs, "y": ys}


def run_v7_dose_effect() -> dict:
    """Curvas dosis-efecto v6 vs v5 (propofol->BIS, remi, nora->ART_MBP)."""
    out: dict = {}
    for cohort in ("synthetic_v6", "synthetic_v5"):
        out[cohort] = {}
        d = paths.LOST_COHORT_DIRS[cohort] / "truth"
        if not d.exists():
            continue
        ce_p, bis, ce_r, mapv, nora = [], [], [], [], []
        for p in sorted(d.glob("*_truth.parquet")):
            df = pq.read_table(p).to_pandas()
            if "ce_propofol" not in df.columns:
                continue
            ce_p.append(df["ce_propofol"].to_numpy())
            bis.append(df["bis"].to_numpy())
            if "ce_remifentanil" in df.columns:
                ce_r.append(df["ce_remifentanil"].to_numpy())
            if "map" in df.columns and "noradrenaline_rate" in df.columns:
                mapv.append(df["map"].to_numpy())
                nora.append(df["noradrenaline_rate"].to_numpy())
        if ce_p:
            out[cohort]["propofol_bis"] = dose_effect_curve(
                np.concatenate(ce_p), np.concatenate(bis))
        if ce_r:
            out[cohort]["remi_map"] = dose_effect_curve(
                np.concatenate(ce_r), np.concatenate(mapv))
        if nora:
            out[cohort]["nora_map"] = dose_effect_curve(
                np.concatenate(nora), np.concatenate(mapv))
    return out


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------

def compute_results() -> dict:
    t0 = _time.time()
    print("[v6_validate] cargando exclusiones", flush=True)
    excluded = cg.load_excluded_caseids()

    print("[v6_validate] partición real_calib/real_holdout", flush=True)
    calib, holdout_ids = split_real_caseids(excluded)
    holdout_set = frozenset(holdout_ids)

    print(f"[v6_validate] holdout: {len(holdout_ids)} caseids, calib: "
          f"{len(calib)} caseids", flush=True)

    print("[v6_validate] cargando holdout real (windows_v4)", flush=True)
    real_all = cg.load_cells(["real"], "val", excluded)
    hmask = np.isin(real_all["caseid"], list(holdout_set))
    holdout = {
        "values": real_all["values"][hmask],
        "masks": real_all["masks"][hmask],
        "caseid": real_all["caseid"][hmask],
    }

    if WINDOWS_V6.exists():
        print("[v6_validate] cargando celdas v6 (windows_v3)", flush=True)
        synth = load_cells_from(WINDOWS_V6, SYNTH_SOURCES_V6, "val", excluded)
    else:
        print("[v6_validate] AVISO: windows_v3 no existe; V1-V4 se calculan "
              "sobre la cohorte v5 como referencia", flush=True)
        synth = load_cells_from(WINDOWS_REAL, SYNTH_SOURCES_V5, "val", excluded)
    synth_v5 = load_cells_from(WINDOWS_REAL, SYNTH_SOURCES_V5, "val", excluded)

    print("[v6_validate] V1/V2 sondas", flush=True)
    v1v2 = _run_v1_v2(holdout, synth)
    print("[v6_validate] V3 solapamiento de soportes", flush=True)
    v3 = _run_v3(holdout, synth, synth_v5)
    print("[v6_validate] V4 D4/P1b", flush=True)
    v4 = _run_v4(holdout, synth)
    print("[v6_validate] V5 pk gate3", flush=True)
    v5 = run_v5_pk_gate3()
    print("[v6_validate] V6 cf prefix", flush=True)
    v6 = run_v6_cf_prefix()
    print("[v6_validate] V7 dosis-efecto", flush=True)
    v7 = run_v7_dose_effect()

    results = {
        "meta": {
            "sha256_v6_validate_py": sha256(Path(__file__)),
            "sha256_cohort_gap_py": sha256(Path(cg.__file__)),
            "sha256_gap_addendum_py": sha256(Path(ga.__file__)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_holdout_caseids": len(holdout_ids),
            "n_calib_caseids": len(calib),
            "holdout_caseids_sha256": hashlib.sha256(
                ",".join(str(c) for c in holdout_ids).encode()).hexdigest(),
            "windows_v6_exists": WINDOWS_V6.exists(),
        },
        "v1v2": v1v2,
        "v3": v3,
        "v4": v4,
        "v5": v5,
        "v6": v6,
        "v7": v7,
    }
    results["elapsed_s"] = round(_time.time() - t0, 1)
    return results


def run_all(use_cache: bool = True) -> dict:
    if use_cache and CACHE_PATH.exists():
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        sha = cached.get("meta", {}).get("sha256_v6_validate_py")
        if sha and sha == sha256(Path(__file__)):
            print("[v6_validate] reutilizando cache", flush=True)
            write_report(cached)
            return cached
    results = compute_results()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                     default=str), encoding="utf-8")
    write_report(results)
    return results


def _read_pytest(txt: Path) -> str:
    if not txt.exists():
        return "(no disponible)"
    raw = txt.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Informe (parametrizado: cohort + conclusión generada)
# --------------------------------------------------------------------------

def _v7_curves_match(v7: dict, cohort: str) -> tuple[bool | None, float | None]:
    """Compara la curva dosis-efecto (propofol->BIS) de la cohorte vs v5,
    interpolando a una rejilla común. Devuelve (match, max_abs_diff)."""
    if not isinstance(v7, dict):
        return None, None
    v5_key = "synthetic_v5"
    other = [k for k in v7.keys() if k != v5_key]
    if not other or v5_key not in v7:
        return None, None
    c1 = v7.get(other[0], {}).get("propofol_bis")
    c2 = v7.get(v5_key, {}).get("propofol_bis")
    if not c1 or not c2 or not c1.get("x") or not c2.get("x"):
        return None, None
    x1, y1 = np.asarray(c1["x"], float), np.asarray(c1["y"], float)
    x2, y2 = np.asarray(c2["x"], float), np.asarray(c2["y"], float)
    xg = np.unique(np.sort(np.r_[x1, x2]))
    if xg.size < 2:
        return None, None
    i1 = np.interp(xg, x1, y1)
    i2 = np.interp(xg, x2, y2)
    diff = float(np.max(np.abs(i1 - i2)))
    return (diff < 1.0), diff


def write_report(results: dict, cohort: str = "v6") -> Path:
    L: list[str] = []
    add = L.append
    meta = results["meta"]

    windows_label = f"windows_{cohort}"
    add(f"REPORT_generator_validation.txt — validación {cohort} contra "
        f"real_holdout")
    add("=" * 78)
    add("")
    add("0. Contexto")
    add("-" * 40)
    add(f"  v6_validate.py  sha256 {meta.get('sha256_v6_validate_py')}")
    add(f"  generado        {meta.get('generated_at')}")
    add(f"  holdout caseids {meta.get('n_holdout_caseids')}")
    add(f"  calib caseids   {meta.get('n_calib_caseids')}")
    add(f"  {windows_label}      "
        f"{'presente' if meta.get('windows_v6_exists') else 'AUSENTE (usando v5)'}")
    add("")
    add("1. Supuestos numerados")
    add("-" * 40)
    for i, s in enumerate(ASSUMPTIONS, 1):
        # Inyectar la etiqueta de cohorte (v6 por defecto, v7 cuando
        # se valida la cohorte v7).
        text = s.replace("cohortes v6", f"cohortes {cohort}")
        text = text.replace("casos de v6", f"casos de {cohort}")
        text = text.replace("cf_v6", f"cf_{cohort}")
        text = text.replace("v6 vs v5", f"{cohort} vs v5")
        add(f"  {i}. {text}")
    add("")
    add("2. Salida literal de pytest (ROJO)")
    add("-" * 40)
    add(_read_pytest(ROJO_TXT))
    add("")
    add("3. Salida literal de pytest (VERDE)")
    add("-" * 40)
    add(_read_pytest(VERDE_TXT))
    add("")

    v1v2 = results.get("v1v2", {})
    add("4. V1/V2 — sondas lineal y no lineal")
    add("-" * 40)
    add(f"  V1 lineal 14 valores: AUC = "
        f"{_fmt_auc(v1v2.get('v1_linear_14', {}).get('auc'))} (objetivo <= "
        f"{0.7036:.4f})")
    add(f"  V2 HGB 14 valores:    AUC = "
        f"{_fmt_auc(v1v2.get('v2_hgb_14', {}).get('auc'))} (objetivo < 0.85)")
    add(f"  V2 HGB 28 (val+deltas): AUC = "
        f"{_fmt_auc(v1v2.get('v2_hgb_28', {}).get('auc'))}")
    add("")

    v3 = results.get("v3", {})
    def _v3_synth(cohort: str) -> dict:
        # El key del sintético en la cache puede ser "v6" (hardcoded por el
        # v6_validate original) o el nombre de la cohorte validada. Para v7,
        # el key "v6" contiene en realidad el solapamiento de la cohorte v7
        # (monkeypatcheada vía WINDOWS_V6 -> windows_v4).
        for k in (cohort, "v6", "v7"):
            if isinstance(v3.get(k), dict):
                return v3[k]
        return {}
    add("5. V3 — solapamiento de soportes (20-NN, 14 vars z-scoreadas)")
    add("-" * 40)
    for name in ([cohort, "v5"] if cohort != "v5" else ["v6", "v5"]):
        d = _v3_synth(name) if name == cohort else (v3.get(name) or {})
        add(f"  {name}: media = {_fmt_g(d.get('mean'))}, "
            f"mediana = {_fmt_g(d.get('median'))} "
            f"(objetivo media > 0.25, mediana > 0.10)")
    add("")

    v4 = results.get("v4", {})
    add(f"6. V4 — W1 normalizada (real_holdout vs {cohort}) por variable")
    add("-" * 40)
    add(f"  {'variable':<22} {'W1/std':>9}")
    for track in cg.IMAGE_TRACKS:
        d = (v4.get("w1_norm") or {}).get(track) or {}
        add(f"  {track:<22} {_fmt_g(d.get('w1_norm'), 9)}")
    add("")

    add("7. V5 — pk_tokens gate 3")
    add("-" * 40)
    add(json.dumps(results.get("v5", {}), indent=2, ensure_ascii=False))
    add("")

    add("8. V6 — divergencia de prefijo CF")
    add("-" * 40)
    add(json.dumps(results.get("v6", {}), indent=2, ensure_ascii=False))
    add("")

    add("9. V7 — curvas dosis-efecto")
    add("-" * 40)
    add(json.dumps(results.get("v7", {}), indent=2, ensure_ascii=False))
    add("")

    # ── Conclusión GENERADA a partir de los números ──
    v1_ok = ((v1v2.get("v1_linear_14", {}).get("auc") or 1.0) <= 0.7036)
    v2_ok = ((v1v2.get("v2_hgb_14", {}).get("auc") or 1.0) < 0.85)
    v3d = {}
    for _k in (cohort, "v6", "v7"):
        if isinstance(v3.get(_k), dict):
            v3d = v3[_k]
            break
    v3_ok = ((_f(v3d.get("mean")) or 0.0) > 0.25
             and (_f(v3d.get("median")) or 0.0) > 0.1)
    v5 = results.get("v5", {})
    v5_ok = bool(v5.get("pass"))
    v6 = results.get("v6", {})
    v6max = _f((v6.get("prefix_max_abs_diff") or {}).get("max"))
    v6_ok = v6max is not None and v6max < 1e-6
    v7_match, v7_diff = _v7_curves_match(results.get("v7", {}), cohort)
    v7_ok = v7_match

    add("10. Conclusión (generada a partir de los números)")
    add("-" * 40)
    add(f"  V1 lineal:        {'PASA' if v1_ok else 'FALLA'} "
        f"(AUC {_fmt_auc(v1v2.get('v1_linear_14', {}).get('auc'))}, "
        f"objetivo <= 0.7036)")
    add(f"  V2 no lineal:     {'PASA' if v2_ok else 'FALLA'} "
        f"(HGB 14 {_fmt_auc(v1v2.get('v2_hgb_14', {}).get('auc'))}, "
        f"objetivo < 0.85)")
    add(f"  V3 solapamiento:  {'PASA' if v3_ok else 'FALLA'} "
        f"(media {_fmt_g(v3d.get('mean'))}, objetivo > 0.25)")
    add(f"  V5 gate3 PK:      {'PASA' if v5_ok else 'FALLA'} "
        f"(propofol {_fmt_g(v5.get('propofol_err_mean'))}, "
        f"remi {_fmt_g(v5.get('remifentanilo_err_mean'))}; < 1 %)")
    add(f"  V6 prefijo CF:    {'PASA' if v6_ok else 'FALLA'} "
        f"(max diff {_fmt_g(v6max)})")
    if v7_ok is None:
        add(f"  V7 dosis-efecto:  NO DISPONIBLE (sin curvas comparables)")
    else:
        add(f"  V7 dosis-efecto:  {'PASA' if v7_ok else 'FALLA'} "
            f"(max |{cohort} - v5| propofol->BIS = {_fmt_g(v7_diff)}, "
            f"objetivo < 1.0 BIS)")
    pharma = v5_ok and v6_ok and (v7_ok is not False)
    add(f"  INTEGRIDAD FARMACOLÓGICA (V5-V7): "
        f"{'pasa' if pharma else 'NO pasa'}.")
    add(f"  Resumen: V1 {'PASA' if v1_ok else 'FALLA'}, "
        f"V2 {'PASA' if v2_ok else 'FALLA'}, "
        f"V3 {'PASA' if v3_ok else 'FALLA'}, "
        f"V5 {'PASA' if v5_ok else 'FALLA'}, "
        f"V6 {'PASA' if v6_ok else 'FALLA'}, "
        f"V7 {'PASA' if v7_ok else ('FALLA' if v7_ok is False else 'ND')}.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    return REPORT_PATH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="run", choices=["run", "report"])
    args = ap.parse_args()
    if args.command == "report":
        if not CACHE_PATH.exists():
            print("Sin cache; ejecuta 'run' primero.", file=sys.stderr)
            sys.exit(1)
        results = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        write_report(results)
        return
    run_all()


if __name__ == "__main__":
    main()
