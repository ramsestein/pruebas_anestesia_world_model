"""gap_addendum.py — mediciones adicionales de la brecha real/sintético (P1).

Complemento de ``diagnostics.cohort_gap`` (fase 0). Antes de tocar el
generador se miden cuatro cosas sobre las cohortes v5:

  P1a  SUELO ALCANZABLE: sonda LINEAL (mismo protocolo del gate 6: 200 000
       celdas, semilla 9876, GroupKFold 5 por caseid, LogisticRegression)
       usando SOLO las 11 variables que NO son BIS/EMG, PEEP ni PIP.
  P1b  CUANTIZACIÓN: por variable y por cohorte: número de valores distintos,
       fracción de enteros exactos, mínimo incremento no nulo entre valores
       consecutivos, y fracción de celdas cuyo valor no cambia respecto a la
       anterior.
  P1c  SONDA NO LINEAL DE REFERENCIA: HistGradientBoostingClassifier, mismo
       GroupKFold y mismas celdas, sobre las 14 variables, real vs sintético.
  P1d  RECORTES DE RANGO: mín/máx por variable entre real y cada cohorte, y
       fracción de celdas reales fuera del rango alcanzable del generador
       (los topes ``value_range`` de la capa de medida de simulate.py).

NO modifica el generador. NO toca el AE. Solo pandas/numpy/pyarrow/sklearn.

Uso:
    python -m diagnostics.gap_addendum run      # computa, cachea y escribe informe
    python -m diagnostics.gap_addendum report   # reescribe el informe desde la cache
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
import pyarrow.parquet as pq

import paths

from diagnostics import cohort_gap as cg

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = paths.DIAGNOSTICS_DIR
REPORT_PATH = paths.REPORTS_DIR / "REPORT_gap_addendum.txt"
CACHE_PATH = OUT_DIR / "gap_addendum_results.json"
ROJO_TXT = paths.REPORTS_DIR / "_pytest_gap_addendum_rojo.txt"
VERDE_TXT = paths.REPORTS_DIR / "_pytest_gap_addendum_verde.txt"

# Variables excluidas del suelo P1a (las tres que concentran la brecha).
EXCLUDED_FOR_FLOOR = {"BIS/EMG", "Primus/PEEP_MBAR", "Primus/PIP_MBAR"}
FLOOR_TRACKS: list[str] = [t for t in cg.IMAGE_TRACKS
                           if t not in EXCLUDED_FOR_FLOOR]

# Topes duros de la capa de medida del generador (value_range de cada
# ``self.sensor.observe`` / helper en src/anessim/simulate.py, v5).
# (lo, hi) inclusive: lo que el generador puede emitir como valor observado.
GEN_CLIP: dict[str, tuple[float, float]] = {
    "BIS/BIS": (0.0, 100.0),
    "Solar8000/HR": (30.0, 300.0),
    "Solar8000/PLETH_SPO2": (60.0, 100.0),
    "Primus/ETCO2": (22.0, 52.0),
    "Primus/PEEP_MBAR": (0.0, 15.0),
    "Primus/PIP_MBAR": (5.0, 40.0),
    "Primus/MV": (1.0, 20.0),
    "Primus/TV": (100.0, 1500.0),
    "Primus/RR_CO2": (4.0, 40.0),
    "Solar8000/BT": (15.0, 40.0),
    "Solar8000/ART_MBP": (-100.0, 350.0),
    "Solar8000/ART_SBP": (-100.0, 350.0),
    "Solar8000/ART_DBP": (-100.0, 350.0),
    "BIS/EMG": (0.0, 100.0),
}

ASSUMPTIONS: list[str] = [
    "P1a replica literalmente el protocolo del gate 6 del AE (200 000 celdas "
    "de val, 100 000 reales + 100 000 sintéticas, semilla 9876, GroupKFold(5) "
    "por caseid, StandardScaler por fold, LogisticRegression(max_iter=5000), "
    "AUC medio entre folds) pero restringido a las 11 variables que NO son "
    "BIS/EMG, Primus/PEEP_MBAR ni Primus/PIP_MBAR.",
    "Las celdas con máscara m=0 se imputan con la media de la variable sobre "
    "celdas m=1 del split val (constante global, sin usar etiquetas), igual "
    "que en cohort_gap.",
    "P1b se computa por cohorte (real, synthetic_v5, vaso_reinf_v5, cf_v5) y "
    "por variable sobre celdas de val con máscara 1 y valor finito. 'entero "
    "exacto' = |v - round(v)| <= 1e-6. 'incremento no nulo entre valores "
    "consecutivos' = diferencia temporal dentro del mismo caso (rejilla "
    "contigua de 5 s), ambas celdas con máscara 1. 'no cambia' = diferencia "
    "exactamente 0 respecto a la celda anterior válida del mismo caso.",
    "P1c usa HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, "
    "max_leaf_nodes=31, random_state=0) con el MISMO GroupKFold, las mismas "
    "celdas del gate 6 y la misma imputación que la sonda lineal. Se reporta "
    "el AUC real vs unión de las 3 cohortes sintéticas y, además, real vs cada "
    "cohorte por separado.",
    "P1d compara mín/máx por variable entre real y cada cohorte y cuantifica "
    "la fracción de celdas reales (máscara 1) que cae FUERA del rango "
    "alcanzable del generador: los topes value_range hardcodeados en "
    "simulate.py (_build_tracks, _obs_hr, _obs_art, _obs_bis, _obs_spo2), "
    "listados en GEN_CLIP. Un valor real dentro del rango nominal no implica "
    "que el generador lo produzca con probabilidad no nula, solo que el tope "
    "no lo excluye.",
    "Exclusiones: caseids del manifest de tokens_v1 (35 sin marcas de fase + "
    "el caseid 4476 sin celdas, 36 en total); solo se retienen celdas "
    "phase_from_clinical=maintenance (idéntico a cohort_gap).",
]


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def sha256(path: Path) -> str:
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


def _fmt_g(x) -> str:
    y = _f(x)
    return "    nan" if y is None else f"{y:8.4f}"


# --------------------------------------------------------------------------
# P1c: sonda no lineal
# --------------------------------------------------------------------------

def run_origin_probe_hgb(X, y, groups) -> dict:
    """HistGradientBoostingClassifier 5-fold agrupado por caso, con la MISMA
    estandarización por fold que la sonda lineal (inofensiva para HGB)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
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
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            random_state=0,
        ).fit(Xtr, y[tr])
        scores = clf.predict_proba(Xte)[:, 1]
        if len(np.unique(y[te])) == 2:
            aucs.append(float(roc_auc_score(y[te], scores)))
        accs.append(float(accuracy_score(y[te], scores >= 0.5)))
    return {
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_per_fold": aucs,
        "acc_mean": float(np.mean(accs)),
    }


# --------------------------------------------------------------------------
# P1a: suelo lineal con 11 variables
# --------------------------------------------------------------------------

def _run_p1a(val: dict, idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
             means: np.ndarray) -> dict:
    cols = [cg.IMAGE_TRACKS.index(t) for t in FLOOR_TRACKS]
    X = cg.impute_with(val["values"][idx], val["masks"][idx], means)[:, cols]
    r = cg.run_origin_probe(X, y, groups)
    return {
        "tracks": FLOOR_TRACKS,
        "n_tracks": len(FLOOR_TRACKS),
        "auc_mean": _f(r["auc_mean"]),
        "auc_per_fold": r["auc_per_fold"],
        "acc_mean": r["acc_mean"],
    }


# --------------------------------------------------------------------------
# P1c: sonda no lineal de referencia
# --------------------------------------------------------------------------

def _run_p1c(val: dict, idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
             means: np.ndarray) -> dict:
    X = cg.impute_with(val["values"][idx], val["masks"][idx], means)
    r_union = run_origin_probe_hgb(X, y, groups)

    cell_real = cg._cell_bool(val, lambda s: s == "real")
    caseid = val["caseid"]
    per_cohort = []
    for cohort in cg.SYNTH_SOURCES:
        cell_cohort = cg._cell_bool(val, lambda s: s == cohort)
        universe = cell_real | cell_cohort
        i2, y2, g2 = cg.sample_two_groups(cell_real, universe, caseid)
        X2 = cg.impute_with(val["values"][i2], val["masks"][i2], means)
        r = run_origin_probe_hgb(X2, y2, g2)
        per_cohort.append({"cohort": cohort, "auc": _f(r["auc_mean"])})
    return {"union_auc": _f(r_union["auc_mean"]),
            "union_acc": r_union["acc_mean"],
            "union_auc_per_fold": r_union["auc_per_fold"],
            "per_cohort": per_cohort}


# --------------------------------------------------------------------------
# P1b: cuantización
# --------------------------------------------------------------------------

def quantization_stats(values: np.ndarray, masks: np.ndarray,
                       caseid: np.ndarray, j: int) -> dict:
    """Métricas de cuantización de la variable j.

    Entrada: columna de valores (con NaN en celdas enmascaradas), columna de
    máscaras, y caseid ordenado (las celdas de cada caso son contiguas y en
    orden temporal).
    """
    col = values[:, j].astype(np.float64)
    m = masks[:, j] == 1
    valid = m & np.isfinite(col)
    if not valid.any():
        return {"n": 0, "n_distinct": 0, "frac_integer": None,
                "min_inc": None, "frac_nochange": None}
    cv = col[valid]
    n = int(cv.size)
    n_distinct = int(len(np.unique(cv)))
    rounded = np.abs(cv - np.round(cv)) <= 1e-6
    frac_integer = _f(float(np.mean(rounded)))

    # Consecutivos dentro del mismo caso (rejilla contigua).
    same = np.r_[False, caseid[1:] == caseid[:-1]]
    prev_valid = same[1:] & valid[1:] & valid[:-1]
    diff = np.abs(col[1:] - col[:-1])
    d_prev = diff[prev_valid]
    if d_prev.size:
        nonzero = d_prev[d_prev > 0.0]
        min_inc = _f(float(nonzero.min())) if nonzero.size else None
        frac_nochange = _f(float(np.mean(d_prev == 0.0)))
    else:
        min_inc = None
        frac_nochange = None
    return {"n": n, "n_distinct": n_distinct, "frac_integer": frac_integer,
            "min_inc": min_inc, "frac_nochange": frac_nochange}


def _run_p1b(val: dict) -> dict:
    out: dict = {}
    for source in cg.ALL_SOURCES:
        is_src = cg._cell_bool(val, lambda s, x=source: s == x)
        per_track: dict = {}
        for j, track in enumerate(cg.IMAGE_TRACKS):
            per_track[track] = quantization_stats(
                val["values"][is_src], val["masks"][is_src],
                val["caseid"][is_src], j)
        out[source] = per_track
    return out


# --------------------------------------------------------------------------
# P1d: recortes de rango
# --------------------------------------------------------------------------

def _run_p1d(val: dict) -> dict:
    per_source: dict = {}
    for source in cg.ALL_SOURCES:
        is_src = cg._cell_bool(val, lambda s, x=source: s == x)
        per_source[source] = cg.marginal_stats(val["values"][is_src],
                                               val["masks"][is_src])
    real_mask = cg._cell_bool(val, lambda s: s == "real")
    rv = val["values"][real_mask]
    rm = val["masks"][real_mask]
    outside: dict = {}
    for j, track in enumerate(cg.IMAGE_TRACKS):
        col = rv[:, j].astype(np.float64)
        m = (rm[:, j] == 1) & np.isfinite(col)
        if not m.any() or track not in GEN_CLIP:
            outside[track] = {"n": 0, "frac_outside": None,
                              "clip": None, "n_outside": 0}
            continue
        lo, hi = GEN_CLIP[track]
        cv = col[m]
        outside[track] = {
            "n": int(cv.size),
            "clip": [lo, hi],
            "n_outside": int(np.sum((cv < lo) | (cv > hi))),
            "frac_outside": _f(float(np.mean((cv < lo) | (cv > hi)))),
        }
    return {"per_source": per_source, "real_outside_generator_range": outside}


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------

def compute_results() -> dict:
    t0 = _time.time()
    print("[gap_addendum] cargando exclusiones", flush=True)
    excluded = cg.load_excluded_caseids()

    print("[gap_addendum] cargando celdas val (todas las fuentes)...", flush=True)
    val = cg.load_cells(cg.ALL_SOURCES, "val", excluded)
    print(f"[gap_addendum] val: {val['values'].shape[0]} celdas, "
          f"{len(val['case_ids'])} casos", flush=True)

    cell_real = cg._cell_bool(val, lambda s: s == "real")
    caseid = val["caseid"]
    universe = np.ones(len(caseid), dtype=bool)
    idx, y, groups = cg.sample_two_groups(cell_real, universe, caseid)
    means = cg.column_means(val["values"], val["masks"])

    print("[gap_addendum] P1a suelo lineal (11 variables)...", flush=True)
    p1a = _run_p1a(val, idx, y, groups, means)

    print("[gap_addendum] P1b cuantización...", flush=True)
    p1b = _run_p1b(val)

    print("[gap_addendum] P1c sonda no lineal...", flush=True)
    p1c = _run_p1c(val, idx, y, groups, means)

    print("[gap_addendum] P1d recortes de rango...", flush=True)
    p1d = _run_p1d(val)

    results = {
        "meta": {
            "sha256_gap_addendum_py": sha256(Path(__file__)),
            "sha256_cohort_gap_py": sha256(Path(cg.__file__)),
            "sha256_tokens_manifest": sha256(cg.TOKENS_MANIFEST),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "probe_protocol": ("LogisticRegression(max_iter=5000) y "
                               "HistGradientBoostingClassifier, GroupKFold 5 "
                               "por caseid, estandarización por fold, 200000 "
                               "celdas seed 9876, entrada cruda imputada"),
            "n_val_cells": int(val["values"].shape[0]),
            "n_val_cases": int(len(val["case_ids"])),
            "n_by_source_val": val["n_by_source"],
        },
        "p1a": p1a,
        "p1b": p1b,
        "p1c": p1c,
        "p1d": p1d,
    }
    results["elapsed_s"] = round(_time.time() - t0, 1)
    print(f"[gap_addendum] cómputo completo en {results['elapsed_s']} s",
          flush=True)
    return results


def run_all(use_cache: bool = True,
            cache_path: Path = CACHE_PATH,
            report_path: Path = REPORT_PATH) -> dict:
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        sha = cached.get("meta", {}).get("sha256_gap_addendum_py")
        if sha and sha == sha256(Path(__file__)):
            print("[gap_addendum] reutilizando cache", flush=True)
            write_report(cached, report_path)
            return cached
        print("[gap_addendum] cache obsoleta, recomputando", flush=True)
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


def write_report(results: dict, report_path: Path = REPORT_PATH) -> Path:
    lines: list[str] = []
    add = lines.append
    meta = results["meta"]
    add("REPORT_gap_addendum.txt — mediciones adicionales de la brecha (P1)")
    add("=" * 78)
    add("")
    add("Complemento de REPORT_cohort_gap.txt. Antes de tocar el generador:")
    add("P1a suelo lineal, P1b cuantización, P1c sonda no lineal, P1d recortes.")
    add("")
    add("0. Contexto y ficheros")
    add("-" * 40)
    add(f"  gap_addendum.py                sha256 {meta.get('sha256_gap_addendum_py')}")
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

    # P1a
    p1a = results["p1a"]
    add("4. P1a — Suelo alcanzable (sonda lineal, 11 variables sin EMG/PEEP/PIP)")
    add("-" * 40)
    add(f"  variables: {', '.join(p1a['tracks'])}")
    add(f"  n_variables = {p1a['n_tracks']}")
    add(f"  AUC (lineal, 11 variables) = {_fmt_auc(p1a['auc_mean'])}")
    add(f"  acc = {_fmt_g(p1a['acc_mean'])}")
    add("  (comparar con AUC 14 variables = 0.9734 de cohort_gap)")
    add("")

    # P1c
    p1c = results["p1c"]
    add("5. P1c — Sonda no lineal de referencia (HistGradientBoosting)")
    add("-" * 40)
    add(f"  real vs unión sintética (14 valores): AUC = {_fmt_auc(p1c['union_auc'])}")
    add(f"  acc = {_fmt_g(p1c['union_acc'])}")
    add(f"  folds = {p1c['union_auc_per_fold']}")
    add("  real vs cada cohorte sintética:")
    for e in p1c["per_cohort"]:
        add(f"    real vs {e['cohort']:<16} AUC = {_fmt_auc(e['auc'])}")
    add("")

    # P1b
    p1b = results["p1b"]
    add("6. P1b — Cuantización por variable y cohorte")
    add("-" * 40)
    add("  (n_distinct: nº de valores distintos; frac_int: fracción de enteros")
    add("   exactos; min_inc: mínimo incremento no nulo entre celdas consecutivas")
    add("   del mismo caso; frac_nochg: fracción de celdas que no cambian)")
    for source in cg.ALL_SOURCES:
        add(f"  --- {source} ---")
        add(f"  {'variable':<22} {'n':>9} {'n_distinct':>10} "
            f"{'frac_int':>9} {'min_inc':>9} {'frac_nochg':>11}")
        for track in cg.IMAGE_TRACKS:
            q = p1b[source][track]
            add(f"  {track:<22} {q['n']:>9} {q['n_distinct']:>10} "
                f"{_fmt_g(q['frac_integer'])} {_fmt_g(q['min_inc'])} "
                f"{_fmt_g(q['frac_nochange'])}")
        add("")

    # P1d
    p1d = results["p1d"]
    add("7. P1d — Recortes de rango (mín/máx por variable y cohorte)")
    add("-" * 40)
    add(f"  {'variable':<22} {'real_min':>9} {'real_max':>9} "
        f"{'synth_min':>9} {'synth_max':>9} {'vaso_min':>9} {'vaso_max':>9} "
        f"{'cf_min':>9} {'cf_max':>9}")
    synth_sources = cg.SYNTH_SOURCES
    for track in cg.IMAGE_TRACKS:
        r = p1d["per_source"]["real"][track]
        s = p1d["per_source"][synth_sources[0]][track]
        v = p1d["per_source"][synth_sources[1]][track]
        c = p1d["per_source"][synth_sources[2]][track]
        add(f"  {track:<22} {_fmt_g(r['min'])} {_fmt_g(r['max'])} "
            f"{_fmt_g(s['min'])} {_fmt_g(s['max'])} "
            f"{_fmt_g(v['min'])} {_fmt_g(v['max'])} "
            f"{_fmt_g(c['min'])} {_fmt_g(c['max'])}")
    add("")
    add("  Fracción de celdas REALES (máscara 1) fuera del rango alcanzable")
    add("  del generador (topes value_range de simulate.py):")
    add(f"  {'variable':<22} {'clip':>18} {'n_real':>9} "
        f"{'n_outside':>10} {'frac_outside':>13}")
    for track in cg.IMAGE_TRACKS:
        o = p1d["real_outside_generator_range"][track]
        clip = o.get("clip")
        cs = f"[{clip[0]:g}, {clip[1]:g}]" if clip else "      n/a"
        add(f"  {track:<22} {cs:>18} {o['n']:>9} {o['n_outside']:>10} "
            f"{_fmt_g(o['frac_outside'])}")
    add("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report"], default="run",
                        nargs="?")
    args = parser.parse_args()
    if args.command == "report":
        if not CACHE_PATH.exists():
            print("No hay cache; ejecuta 'run' primero.", file=sys.stderr)
            sys.exit(1)
        results = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        write_report(results)
        print(f"Informe reescrito en {REPORT_PATH}")
        return
    run_all()


if __name__ == "__main__":
    main()
