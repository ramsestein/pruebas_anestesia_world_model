"""diag_source_probe_physio.py — sondeo de fuente sobre la fisiología.

Diagnóstico (sin aserción, sin salida a data/): ¿la fisiología de las ventanas
de mantenimiento delata la cohorte (real vs sintético)?

  - Muestra: 2 000 casos de val (estratificados por source, semilla 0), 20
    ventanas de mantenimiento (t > 600 s) de 12 celdas consecutivas por caso.
  - Features por ventana sobre las 14 variables de la imagen: media, std, min,
    max y fracción de celdas con máscara = 1 (ausente; en windows_v2 la columna
    m_<track> es 1=presente, así que fracción ausente = 1 - media(m_)).
  - Tres sondeos (regresión logística estandarizada, GroupKFold 5-fold por
    caso, balanced accuracy real vs sintético):
      (a) solo fracción de máscara; (b) solo media/std/min/max de variables con
      máscara completa en la ventana; (c) todo.
  - (b) además: ranking de los 15 coeficientes de mayor |coef| y ablación
    quitando la variable top-1 completa y luego las top-3.
  - (c) se repite por pares de cohortes.

Uso: python scripts/diag_source_probe_physio.py
"""

from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import paths

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_DIR = paths.WINDOWS_DIR / "windows"
CASES_PATH = paths.WINDOWS_DIR / "cases.parquet"
MANIFEST = json.loads((paths.WINDOWS_DIR / "manifest.json").read_text(encoding="utf-8"))
IMAGE_TRACKS = list(MANIFEST["image_tracks"])

SEED = 0
N_CASES = 2000
N_WINDOWS_PER_CASE = 20
WINDOW_CELLS = 12
T_MIN_MAINT = 600.0
STATS = ["mean", "std", "min", "max"]

FEAT_MASK = [f"{tr}__mask_frac" for tr in IMAGE_TRACKS]
FEAT_VAL = [f"{tr}__{s}" for tr in IMAGE_TRACKS for s in STATS]
FEAT_ALL = FEAT_MASK + FEAT_VAL


def _stratified_sample(cases: pd.DataFrame, n_total: int) -> pd.DataFrame:
    counts = cases.groupby("source", sort=False).size()
    per = {src: max(1, int(round(n_total * n / len(cases)))) for src, n in counts.items()}
    # ajustar la suma a n_total de forma determinista
    while sum(per.values()) != n_total:
        if sum(per.values()) > n_total:
            src = max(per, key=per.get)
            per[src] -= 1
        else:
            src = min(per, key=per.get)
            per[src] += 1
    frames = []
    for src in sorted(counts.index):
        g = cases[cases["source"] == src]
        k = min(per[src], len(g))
        frames.append(g.sample(n=k, random_state=SEED))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("caseid", kind="stable").reset_index(drop=True)


def _build_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (X, y, groups) con shape (n_windows, n_features), y=1 real."""
    cases = pq.read_table(CASES_PATH, columns=["caseid", "source", "split"]).to_pandas()
    cases["source"] = cases["source"].astype(str)
    cases["split"] = cases["split"].astype(str)
    val = cases[cases["split"] == "val"].reset_index(drop=True)
    sample = _stratified_sample(val, N_CASES)
    sampled = dict(zip(sample["caseid"], sample["source"]))
    sample_set = set(sample["caseid"])

    need_cols = ["caseid", "t", "source", "split", "phase_from_clinical"]
    for tr in IMAGE_TRACKS:
        need_cols += [tr, f"m_{tr}"]

    rng = np.random.default_rng(SEED)
    rows = []
    parts = sorted(WINDOWS_DIR.glob("source=*/split=val/part-*.parquet"))
    n_parts = len(parts)
    for pi, part in enumerate(parts):
        tbl = pq.read_table(part, columns=need_cols).to_pandas()
        tbl = tbl[tbl["caseid"].isin(sample_set)]
        if len(tbl) == 0:
            continue
        for cid, sub in tbl.groupby("caseid", sort=False):
            sub = sub.sort_values("t", kind="stable").reset_index(drop=True)
            maint = sub[(sub["phase_from_clinical"] == "maintenance")
                        & (sub["t"] > T_MIN_MAINT)].reset_index(drop=True)
            if len(maint) < WINDOW_CELLS:
                continue
            t = maint["t"].to_numpy(dtype=np.float64)
            starts = np.where(t[WINDOW_CELLS - 1:] - t[:-WINDOW_CELLS + 1]
                              == (WINDOW_CELLS - 1) * 5.0)[0]
            if len(starts) == 0:
                continue
            pick = rng.choice(starts, size=N_WINDOWS_PER_CASE, replace=True)
            src = sampled[int(cid)]
            for w in pick:
                win = maint.iloc[int(w):int(w) + WINDOW_CELLS]
                feats = []
                for tr in IMAGE_TRACKS:
                    m = win[f"m_{tr}"].to_numpy(dtype=np.float64)
                    feats.append(1.0 - float(m.mean()))  # fracción ausente (máscara=1)
                for tr in IMAGE_TRACKS:
                    m = win[f"m_{tr}"].to_numpy(dtype=np.float64)
                    v = win[tr].to_numpy(dtype=np.float64)
                    if m.min() == 1.0:
                        feats.extend([float(v.mean()), float(v.std(ddof=0)),
                                      float(v.min()), float(v.max())])
                    else:
                        feats.extend([0.0, 0.0, 0.0, 0.0])
                rows.append((int(cid), src, feats))
        print(f"[physio] {pi + 1}/{n_parts} particiones, {len(rows)} ventanas",
              file=sys.stderr, flush=True)

    cids = np.array([r[0] for r in rows], dtype=np.int32)
    srcs = np.array([r[1] for r in rows], dtype=object)
    X = np.array([r[2] for r in rows], dtype=np.float64)
    y = (srcs == "real").astype(np.float64)
    return X, y, cids, srcs


def _probe(X, y, groups, cols=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xk = X if cols is None else X[:, cols]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    yp = cross_val_predict(model, Xk, y, cv=GroupKFold(5), groups=groups,
                           method="predict")
    return float(balanced_accuracy_score(y, yp))


def _fmt(header, rows):
    widths = [len(h) for h in header]
    sr = [[str(c) for c in r] for r in rows]
    for r in sr:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([line, sep] + [" | ".join(c.ljust(widths[i]) for i, c in enumerate(r))
                                    for r in sr])


def main() -> int:
    t0 = _time.time()
    X, y, groups, srcs = _build_features()
    n_mask = len(FEAT_MASK)
    print("=" * 78)
    print("DIAGNÓSTICO 3 — sondeo de fuente sobre la fisiología (sin aserción)")
    print("=" * 78)
    print(f"ventanas totales: {len(y)} (de {len(np.unique(groups))} casos)")
    print(f"n_real={int(y.sum())}  n_synth={int(len(y) - y.sum())}")
    print(f"features: {len(FEAT_ALL)} = {n_mask} máscara + {len(FEAT_VAL)} valores")

    acc_a = _probe(X, y, groups, cols=list(range(n_mask)))
    acc_b = _probe(X, y, groups, cols=list(range(n_mask, X.shape[1])))
    acc_c = _probe(X, y, groups)
    print()
    print("balanced accuracy real vs sintético (GroupKFold por caso, 5-fold):")
    print(_fmt(["sondeo", "accuracy"],
               [["(a) solo fracción de máscara", f"{acc_a:.4f}"],
                ["(b) solo media/std/min/max (máscara completa)", f"{acc_b:.4f}"],
                ["(c) todo", f"{acc_c:.4f}"]]))

    # ranking de (b)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xb = X[:, n_mask:]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(Xb, y)
    coefs = m.named_steps["logisticregression"].coef_[0]
    order = np.argsort(-np.abs(coefs))
    print()
    print("(b) ranking de los 15 coeficientes de mayor |coef|:")
    top15 = [(FEAT_VAL[i], float(coefs[i]), FEAT_VAL[i].split("__")[0]) for i in order[:15]]
    print(_fmt(["#", "feature", "coef", "variable"],
               [[str(i + 1), f, f"{c:+.4f}", v] for i, (f, c, v) in enumerate(top15)]))

    # ablación de (b)
    def acc_b_without(vars_to_drop):
        keep = [i for i, fn in enumerate(FEAT_VAL) if fn.split("__")[0] not in vars_to_drop]
        cols = [n_mask + i for i in keep]
        return _probe(X, y, groups, cols=cols)

    top1_var = FEAT_VAL[order[0]].split("__")[0]
    distinct_vars = []
    for i in order:
        v = FEAT_VAL[i].split("__")[0]
        if v not in distinct_vars:
            distinct_vars.append(v)
        if len(distinct_vars) == 3:
            break
    top3_vars = distinct_vars
    print()
    print("(b) ablación (balanced accuracy):")
    print(_fmt(["ablación", "accuracy"],
               [[f"sin variable top-1 ({top1_var})", f"{acc_b_without({top1_var}):.4f}"],
                [f"sin top-3 ({', '.join(top3_vars)})", f"{acc_b_without(set(top3_vars)):.4f}"]]))

    # pares de cohortes sobre (c)
    pairs = [("real", "synthetic_v7"), ("real", "cf_v7"),
             ("real", "vaso_reinf_v7"), ("synthetic_v7", "cf_v7")]
    print()
    print("(c) por pares de cohortes (balanced accuracy):")
    from sklearn.linear_model import LogisticRegression as LR2
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline as mp2
    from sklearn.preprocessing import StandardScaler as SS2
    pair_rows = []
    for pos, neg in pairs:
        mask = (srcs == pos) | (srcs == neg)
        Xk = X[mask]
        yk = (srcs[mask] == pos).astype(np.float64)
        gk = groups[mask]
        model = mp2(SS2(), LR2(max_iter=2000))
        yp = cross_val_predict(model, Xk, yk, cv=GroupKFold(5), groups=gk, method="predict")
        acc = float(balanced_accuracy_score(yk, yp))
        pair_rows.append([f"{pos} vs {neg}", f"{acc:.4f}", str(int(yk.sum())),
                          str(int(len(yk) - yk.sum()))])
    print(_fmt(["par", "accuracy", "n_pos", "n_neg"], pair_rows))

    print()
    print(f"elapsed_s={_time.time() - t0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
