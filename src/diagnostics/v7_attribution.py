"""v7_attribution.py — atribución de la sonda NO LINEAL (PASO A de v7).

Replica D1 de cohort_gap pero con HistGradientBoostingClassifier, sobre las
mismas 200 000 celdas (100 000 real_holdout + 100 000 v6), GroupKFold 5 por
caseid, estandarización por fold, imputación máscara=0 por la media:

  A1  Univariante HGB, 14 valores, ordenado descendente.
  A2  Acumulativo hacia delante en el orden de A1.
  A3  Ablación: 14 sondas HGB con todas menos una (caída vs 14).
  A4  Vector 28 (valores+deltas): univariante HGB sobre los 14 deltas, y
      ablación del vector 28 (quitar cada valor y cada delta).
  A5  Cuantización lado a lado: n_distinct, frac_int, min_inc, frac_nochg de
      real_holdout y de v6 (el dato que _run_v4 calcula pero no imprime).
  A6  D7: matriz de correlación de Pearson 14x14 (real vs v6), norma de
      Frobenius de la diferencia, top 10 pares; e información mutua por pares
      (discretización en deciles) con top 10 discrepancias.

NO modifica el generador. Solo pandas/numpy/pyarrow/sklearn.

Uso:
    python -m diagnostics.v7_attribution run      # computa, cachea, escribe
    python -m diagnostics.v7_attribution report   # reescribe desde la cache
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

from diagnostics import cohort_gap as cg
from diagnostics import gap_addendum as ga
from diagnostics import v6_validate as vv

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "diagnostics"
REPORT_PATH = ROOT / "src" / "diagnostics" / "REPORT_v7_attribution.txt"
CACHE_PATH = OUT_DIR / "v7_attribution_results.json"
ROJO_TXT = ROOT / "reports" / "_pytest_v7_attribution_rojo.txt"
VERDE_TXT = ROOT / "reports" / "_pytest_v7_attribution_verde.txt"

SEED = 9876
HALF = 100_000

ASSUMPTIONS: list[str] = [
    "A1-A4 replican el protocolo del gate 6 con HistGradientBoostingClassifier("
    "max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=0), "
    "GroupKFold 5 por caseid, StandardScaler por fold, 200 000 celdas (100 000 "
    "real_holdout + 100 000 v6) semilla 9876, imputación máscara=0 por la media.",
    "real_holdout es la partición 40 % por caseid (semilla 20260922) del real "
    "de val; v6 es la unión de synthetic_v6, vaso_reinf_v6 y cf_v6 en windows_v3.",
    "Los deltas son t - t_menos_1 dentro del mismo caso (rejilla contigua 5 s); "
    "la primera celda y los pares con máscara 0 se imputan con la media del delta.",
    "A5 usa celdas con máscara 1 y valor finito, por cohorte (real_holdout vs "
    "unión v6). 'entero exacto' = |v - round(v)| <= 1e-6. 'frac_nochg' = "
    "fracción de celdas cuyo valor coincide con la celda anterior válida del "
    "mismo caso.",
    "A6 correlación de Pearson sobre valores imputados (máscara=0 -> media). "
    "La información mutua se estima por histograma 2D con discretización en "
    "deciles (10 cuantiles) por variable, sobre las mismas 200 000 celdas.",
    "Las exclusiones son las del manifest de tokens_v1 (35 sin marcas de "
    "fase + el caseid 4476 sin celdas, 36 en total); solo celdas "
    "phase_from_clinical=maintenance.",
]


def sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _f(x):
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


# --------------------------------------------------------------------------
# Utilidades de atribución (unitarias)
# --------------------------------------------------------------------------

def _top_pairs(pairs: list[tuple[str, float]], n: int) -> list[tuple[str, float]]:
    """Devuelve los n pares con mayor valor (orden descendente)."""
    return sorted(pairs, key=lambda e: -(abs(e[1]) or 0.0))[:n]


def pairwise_mi(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Información mutua por histograma 2D, discretización en deciles."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 50:
        return 0.0
    qx = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    qy = np.unique(np.quantile(y, np.linspace(0, 1, n_bins + 1)))
    xi = np.clip(np.searchsorted(qx, x, side="right") - 1, 0, len(qx) - 1)
    yi = np.clip(np.searchsorted(qy, y, side="right") - 1, 0, len(qy) - 1)
    h, _, _ = np.histogram2d(xi, yi, bins=[np.arange(len(qx) + 1) - 0.5,
                                           np.arange(len(qy) + 1) - 0.5])
    p = h / h.sum()
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p * np.log(p / (px * py))
    return float(np.nansum(term))


def corr_difference(A: np.ndarray, B: np.ndarray) -> dict:
    """Matriz de correlación de Pearson y discrepancia con top pares."""
    Ca = np.corrcoef(A.T)
    Cb = np.corrcoef(B.T)
    diff = Ca - Cb
    fro = float(np.linalg.norm(diff, ord="fro"))
    n = diff.shape[0]
    pairs: list[tuple[str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((f"{i}|{j}", float(diff[i, j])))
    return {"frobenius": fro, "top_pairs": _top_pairs(pairs, 10)}


def _quant_table_rows(real_q: dict, synth_q: dict, tracks: list[str]) -> list[dict]:
    """Filas de la tabla de cuantización lado a lado."""
    rows: list[dict] = []
    for t in tracks:
        r = real_q.get(t, {})
        s = synth_q.get(t, {})
        rows.append({
            "track": t,
            "real_n_distinct": r.get("n_distinct"),
            "real_frac_int": r.get("frac_integer"),
            "real_min_inc": r.get("min_inc"),
            "real_frac_nochg": r.get("frac_nochange"),
            "syn_n_distinct": s.get("n_distinct"),
            "syn_frac_int": s.get("frac_integer"),
            "syn_min_inc": s.get("min_inc"),
            "syn_frac_nochg": s.get("frac_nochange"),
        })
    return rows


def _ablation_from_univ(univ: list[dict], auc_without: dict[str, float],
                        full: float) -> list[dict]:
    """Ablación: drop = full - auc_sin_variable, ordenado descendente."""
    out = []
    for e in univ:
        t = e["track"]
        auc = auc_without.get(t)
        out.append({"track": t, "auc": _f(auc), "drop": _f((full or 0.0) - (auc or 0.0))})
    out.sort(key=lambda e: -(e["drop"] or -1.0))
    return out


# --------------------------------------------------------------------------
# Cómputo A1-A6
# --------------------------------------------------------------------------

def _hgb_probe(X, y, groups) -> dict:
    return ga.run_origin_probe_hgb(X, y, groups)


def _build_sample(holdout: dict, synth: dict):
    """Submuestra estratificada 100k+100k y matrices imputadas (14 y 28)."""
    merged = {
        "values": np.concatenate([holdout["values"], synth["values"]], axis=0),
        "masks": np.concatenate([holdout["masks"], synth["masks"]], axis=0),
        "caseid": np.concatenate([holdout["caseid"], synth["caseid"]], axis=0),
    }
    n_real = len(holdout["caseid"])
    y = np.r_[np.ones(n_real, dtype=int), np.zeros(len(synth["caseid"]), dtype=int)]
    means = cg.column_means(merged["values"], merged["masks"])
    rng = np.random.default_rng(SEED)
    pos_r = np.flatnonzero(y == 1)
    pos_s = np.flatnonzero(y == 0)
    sel_r = np.sort(rng.choice(pos_r, size=min(HALF, len(pos_r)), replace=False))
    sel_s = np.sort(rng.choice(pos_s, size=min(HALF, len(pos_s)), replace=False))
    idx = np.concatenate([sel_r, sel_s])
    y2 = y[idx]
    groups = merged["caseid"][idx]
    X14 = cg.impute_with(merged["values"][idx], merged["masks"][idx], means)
    dmeans = cg.delta_means(merged["values"], merged["masks"], merged["caseid"])
    deltas = cg.deltas_for_indices(idx, merged["values"], merged["masks"],
                                   merged["caseid"], dmeans)
    X28 = np.concatenate([X14, deltas], axis=1)
    return X14, X28, y2, groups


def _run_a1_a2_a3(X14, y, groups) -> dict:
    n = X14.shape[1]
    univariate = []
    for j in range(n):
        r = _hgb_probe(X14[:, [j]], y, groups)
        univariate.append({"track": cg.IMAGE_TRACKS[j], "auc": _f(r["auc_mean"])})
    univariate.sort(key=lambda e: -(e["auc"] or -1.0))
    order = [cg.IMAGE_TRACKS.index(e["track"]) for e in univariate]
    cumulative = []
    for k in range(1, n + 1):
        cols = order[:k]
        r = _hgb_probe(X14[:, cols], y, groups)
        cumulative.append({"k": k, "tracks": [cg.IMAGE_TRACKS[c] for c in cols],
                           "auc": _f(r["auc_mean"])})
    full14 = cumulative[-1]["auc"]
    auc_without = {}
    for j in range(n):
        cols = [c for c in range(n) if c != j]
        r = _hgb_probe(X14[:, cols], y, groups)
        auc_without[cg.IMAGE_TRACKS[j]] = r["auc_mean"]
    ablation = _ablation_from_univ(univariate, auc_without, full14)
    return {"univariate": univariate, "cumulative": cumulative,
            "ablation": ablation, "full14_auc": full14}


def _run_a4(X28, y, groups) -> dict:
    n_val = len(cg.IMAGE_TRACKS)
    # Univariante sobre los 14 deltas (columnas n_val .. 2*n_val-1).
    univ_deltas = []
    for j in range(n_val):
        r = _hgb_probe(X28[:, [n_val + j]], y, groups)
        univ_deltas.append({"track": cg.IMAGE_TRACKS[j], "auc": _f(r["auc_mean"])})
    univ_deltas.sort(key=lambda e: -(e["auc"] or -1.0))
    full28 = _f(_hgb_probe(X28, y, groups)["auc_mean"])
    # Ablación del vector 28: quitar cada valor y cada delta.
    auc_without = {}
    for j in range(2 * n_val):
        cols = [c for c in range(2 * n_val) if c != j]
        r = _hgb_probe(X28[:, cols], y, groups)
        key = f"{cg.IMAGE_TRACKS[j % n_val]}{'_delta' if j >= n_val else ''}"
        auc_without[key] = r["auc_mean"]
    ablation = []
    for key, auc in auc_without.items():
        ablation.append({"track": key, "auc": _f(auc),
                         "drop": _f((full28 or 0.0) - (auc or 0.0))})
    ablation.sort(key=lambda e: -(e["drop"] or -1.0))
    return {"univariate_deltas": univ_deltas, "full28_auc": full28,
            "ablation": ablation}


def _run_a5(holdout: dict, synth: dict) -> dict:
    rq: dict = {}
    sq: dict = {}
    for j, t in enumerate(cg.IMAGE_TRACKS):
        rq[t] = ga.quantization_stats(holdout["values"], holdout["masks"],
                                      holdout["caseid"], j)
        sq[t] = ga.quantization_stats(synth["values"], synth["masks"],
                                      synth["caseid"], j)
    return {"real": rq, "v6": sq,
            "rows": _quant_table_rows(rq, sq, cg.IMAGE_TRACKS)}


def _run_a6(synth: dict, holdout: dict) -> dict:
    # Correlación de Pearson sobre el FULL (no solo la submuestra): imputado.
    means = cg.column_means(
        np.concatenate([holdout["values"], synth["values"]], axis=0),
        np.concatenate([holdout["masks"], synth["masks"]], axis=0))
    Xr = cg.impute_with(holdout["values"], holdout["masks"], means)
    Xs = cg.impute_with(synth["values"], synth["masks"], means)
    # Submuestreo para que el cómputo sea manejable (mismo rng).
    rng = np.random.default_rng(SEED)
    rr = np.sort(rng.choice(len(Xr), size=min(100_000, len(Xr)), replace=False))
    rs = np.sort(rng.choice(len(Xs), size=min(100_000, len(Xs)), replace=False))
    corr = corr_difference(Xr[rr], Xs[rs])
    corr["top_pairs_named"] = [
        (f"{cg.IMAGE_TRACKS[i]}|{cg.IMAGE_TRACKS[j]}", v)
        for (i, j), v in _top_pairs_named(corr["top_pairs"])
    ]
    # Información mutua por pares.
    mi_real: dict = {}
    mi_synth: dict = {}
    for i in range(len(cg.IMAGE_TRACKS)):
        for j in range(i + 1, len(cg.IMAGE_TRACKS)):
            key = f"{cg.IMAGE_TRACKS[i]}|{cg.IMAGE_TRACKS[j]}"
            mi_real[key] = pairwise_mi(Xr[rr, i], Xr[rr, j])
            mi_synth[key] = pairwise_mi(Xs[rs, i], Xs[rs, j])
    mi_diff = [(k, (mi_synth.get(k, 0.0) - mi_real.get(k, 0.0)))
               for k in mi_real]
    return {"corr": corr, "mi": {"real": mi_real, "v6": mi_synth},
            "mi_top_pairs": _top_pairs(mi_diff, 10)}


def _top_pairs_named(pairs: list[tuple[str, float]]) -> list[tuple[tuple[int, int], float]]:
    return [(tuple(int(x) for x in k.split("|")), v) for k, v in pairs]


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------

def compute_results() -> dict:
    t0 = _time.time()
    print("[v7_attribution] exclusiones y partición", flush=True)
    excluded = cg.load_excluded_caseids()
    _, holdout_ids = vv.split_real_caseids(excluded)
    holdout_set = frozenset(holdout_ids)

    print("[v7_attribution] cargando real_holdout (windows_v2)", flush=True)
    real_all = cg.load_cells(["real"], "val", excluded)
    hmask = np.isin(real_all["caseid"], list(holdout_set))
    holdout = {"values": real_all["values"][hmask],
               "masks": real_all["masks"][hmask],
               "caseid": real_all["caseid"][hmask]}

    print("[v7_attribution] cargando v6 (windows_v3)", flush=True)
    synth = vv.load_cells_from(vv.WINDOWS_V6, vv.SYNTH_SOURCES_V6, "val", excluded)

    print("[v7_attribution] submuestra e imputación", flush=True)
    X14, X28, y, groups = _build_sample(holdout, synth)

    print("[v7_attribution] A1/A2/A3 (HGB 14 valores)", flush=True)
    a123 = _run_a1_a2_a3(X14, y, groups)
    print("[v7_attribution] A4 (HGB 28 valores+deltas)", flush=True)
    a4 = _run_a4(X28, y, groups)
    print("[v7_attribution] A5 cuantización lado a lado", flush=True)
    a5 = _run_a5(holdout, synth)
    print("[v7_attribution] A6 correlación + MI", flush=True)
    a6 = _run_a6(synth, holdout)

    results = {
        "meta": {
            "sha256_v7_attribution_py": sha256(Path(__file__)),
            "sha256_cohort_gap_py": sha256(Path(cg.__file__)),
            "sha256_gap_addendum_py": sha256(Path(ga.__file__)),
            "sha256_v6_validate_py": sha256(Path(vv.__file__)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_holdout_caseids": len(holdout_ids),
            "n_real_cells": int(holdout["values"].shape[0]),
            "n_v6_cells": int(synth["values"].shape[0]),
        },
        "a1": a123["univariate"],
        "a2": a123["cumulative"],
        "a3": a123["ablation"],
        "a4": a4,
        "a5": a5,
        "a6": a6,
        "full14_auc": a123["full14_auc"],
    }
    results["elapsed_s"] = round(_time.time() - t0, 1)
    print(f"[v7_attribution] completo en {results['elapsed_s']} s", flush=True)
    return results


def run_all(use_cache: bool = True) -> dict:
    if use_cache and CACHE_PATH.exists():
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        sha = cached.get("meta", {}).get("sha256_v7_attribution_py")
        if sha and sha == sha256(Path(__file__)):
            print("[v7_attribution] reutilizando cache", flush=True)
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


def write_report(results: dict) -> Path:
    L: list[str] = []
    add = L.append
    meta = results["meta"]
    add("REPORT_v7_attribution.txt — atribución de la sonda no lineal (PASO A)")
    add("=" * 78)
    add("")
    add("0. Contexto")
    add("-" * 40)
    add(f"  v7_attribution.py sha256 {meta.get('sha256_v7_attribution_py')}")
    add(f"  generado             {meta.get('generated_at')}")
    add(f"  holdout caseids      {meta.get('n_holdout_caseids')}")
    add(f"  celdas real / v6     {meta.get('n_real_cells')} / {meta.get('n_v6_cells')}")
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

    add("4. A1 — Univariante HGB (14 valores)")
    add("-" * 40)
    add(f"  {'#':>2}  {'variable':<22} {'AUC':>8}")
    for i, e in enumerate(results["a1"], 1):
        add(f"  {i:>2}  {e['track']:<22} {_fmt_auc(e['auc'])}")
    add("")

    add("5. A2 — Acumulativo HGB (en el orden de A1)")
    add("-" * 40)
    add(f"  {'k':>2}  {'AUC':>8}  variables")
    for e in results["a2"]:
        add(f"  {e['k']:>2}  {_fmt_auc(e['auc'])}  {' + '.join(e['tracks'])}")
    add("")

    add("6. A3 — Ablación HGB (14 menos una)")
    add("-" * 40)
    add(f"  {'variable':<22} {'AUC':>8} {'caida':>8}")
    for e in results["a3"]:
        add(f"  {e['track']:<22} {_fmt_auc(e['auc'])} {_fmt_g(e['drop'])}")
    add("")

    add("7. A4 — HGB 28 (valores+deltas)")
    add("-" * 40)
    add(f"  full 28 AUC = {_fmt_auc(results['a4']['full28_auc'])}  "
        f"(full 14 = {_fmt_auc(results['full14_auc'])})")
    add("  Univariante HGB sobre los 14 deltas:")
    for e in results["a4"]["univariate_deltas"]:
        add(f"    {e['track']:<22} {_fmt_auc(e['auc'])}")
    add("  Ablación del vector 28 (top caídas):")
    for e in results["a4"]["ablation"][:14]:
        add(f"    {e['track']:<26} {_fmt_auc(e['auc'])} {_fmt_g(e['drop'])}")
    add("")

    add("8. A5 — Cuantización lado a lado (real_holdout vs v6)")
    add("-" * 40)
    add(f"  {'variable':<22} {'r_ndist':>8} {'r_int':>6} {'r_mininc':>9} "
        f"{'r_nochg':>8} | {'s_ndist':>8} {'s_int':>6} {'s_mininc':>9} "
        f"{'s_nochg':>8}")
    for row in results["a5"]["rows"]:
        add(f"  {row['track']:<22} {str(row['real_n_distinct']):>8} "
            f"{_fmt_g(row['real_frac_int'], 6)} {_fmt_g(row['real_min_inc'], 9)} "
            f"{_fmt_g(row['real_frac_nochg'])} | {str(row['syn_n_distinct']):>8} "
            f"{_fmt_g(row['syn_frac_int'], 6)} {_fmt_g(row['syn_min_inc'], 9)} "
            f"{_fmt_g(row['syn_frac_nochg'])}")
    add("")

    add("9. A6 — D7: correlación e información mutua")
    add("-" * 40)
    c = results["a6"]["corr"]
    add(f"  Frobenius ||corr_real - corr_v6|| = {_fmt_g(c['frobenius'])}")
    add("  Top 10 pares con mayor discrepancia de correlación:")
    for name, v in c.get("top_pairs_named", []):
        add(f"    {name:<42} {v:+.4f}")
    add("  Top 10 pares con mayor discrepancia de información mutua:")
    for name, v in results["a6"]["mi_top_pairs"]:
        add(f"    {name:<42} {v:+.4f}")
    add("")

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
