"""PASO 3 / C1 — reproducción a escala completa de la cohorte REAL.

Tokeniza la cohorte real completa (windows_v4) con los estadísticos de
normalización CONGELADOS de data/tokens_v1/manifest_tokens.json y compara con
la porción real de data/tokens_v1/.

Bloqueante C1:
  - mismo nº de filas por split y por flag dense (train 734 663; val 1 501 733,
    de las que 1 376 544 densas);
  - columnas drug_*, vent_*, time_*, máscaras y metadatos: idénticas
    (|diff| < 1e-6);
  - columnas ctx_*: las binarias (sex, asa, emop) idénticas; cada continua
    (age, height, weight, bmi) es una transformación afín exacta de la de
    tokens_v1 (una sola escala y un solo desplazamiento, residuo < 1e-6).

Escribe manifests/tokens_v2_reproduccion_real.json (recuentos y sumas de
control por columna) para que el test tests/test_paso3_c1.py lo fije.

Uso:
  python -m scripts.paso3_c1_reproduce   (requiere data/pk_v2 ya generado)
"""

from __future__ import annotations

import json
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import paths
from tokens import tokenize as tk

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = paths.MANIFESTS_DIR / "tokens_v2_reproduccion_real.json"

# Contexto continuo y binario/categórico del inventario v1.
CTX_CONTINUOUS = ("age", "height", "weight", "bmi")
CTX_BINARY = tuple(i for i in (
    "sex:F", "asa:1", "asa:2", "asa:3", "asa:4", "asa:otros",
    "emop:0", "emop:1"))

# Columnas de comparación exacta: todo el esquema salvo ctx_*.
EXACT_COLS = [
    c for c in tk.output_columns(tk.load_vocab_v1())
    if not c.startswith("ctx_")
]


def _read_tokens_v1_real(split: str) -> pd.DataFrame:
    """Lee TODAS las filas reales de data/tokens_v1 para un split."""
    base = paths.TOKENS_V1_DIR / "windows" / "source=real" / f"split={split}"
    ctx = [f"ctx_{i}" for i in tk.load_vocab_v1()]
    cols = list(dict.fromkeys(["caseid", "t0", "dense"] + EXACT_COLS + ctx))
    frames = [pq.read_table(p, columns=cols).to_pandas()
              for p in sorted(base.glob("part-*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["caseid", "t0"], kind="stable").reset_index(drop=True)


def _build_new_real(split: str, dense_val: bool, frozen_stats: dict,
                    context_map: dict, cf_meta: dict, excluded: frozenset) -> pd.DataFrame:
    """Reconstruye las ventanas reales de windows_v4 con stats congelados."""
    base = paths.WINDOWS_DIR / "windows" / "source=real" / f"split={split}"
    ctx_ids = tk.load_vocab_v1()
    frames = []
    for part in sorted(base.glob("part-*.parquet")):
        w, _ = tk.process_partition_windows(part, dense_val, context_map, cf_meta,
                                            excluded_caseids=excluded)
        if len(w["t0"]) == 0:
            continue
        norm = tk.apply_normalization(w, frozen_stats)
        tbl = tk.build_table(norm, ctx_ids).to_pandas()
        frames.append(tbl)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["caseid", "t0"], kind="stable").reset_index(drop=True)


def _max_abs_diff(a: pd.Series, b: pd.Series) -> float:
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        av = a.to_numpy(dtype=np.float64)
        bv = b.to_numpy(dtype=np.float64)
        both_nan = np.isnan(av) & np.isnan(bv)
        one_nan = np.isnan(av) ^ np.isnan(bv)
        if one_nan.any():
            return float("inf")
        d = np.abs(av - bv)
        d = d[~both_nan]
        return float(d.max()) if len(d) else 0.0
    # columnas no numéricas (source/split/cf_role/lever): igualdad exacta
    na = a.isna().to_numpy()
    nb = b.isna().to_numpy()
    if not np.array_equal(na, nb):
        return float("inf")
    if (~na).sum() == 0:
        return 0.0
    return 0.0 if (a.astype(str).to_numpy()[~na] == b.astype(str).to_numpy()[~nb]).all() else float("inf")


def _fit_affine(old: np.ndarray, new: np.ndarray) -> dict:
    """Ajusta new = scale*old + offset por mínimos cuadrados; residuo máximo."""
    ok = np.isfinite(old) & np.isfinite(new)
    o, n = old[ok], new[ok]
    if len(o) < 2:
        return {"scale": float("nan"), "offset": float("nan"),
                "max_residual": float("nan"), "n": int(len(o))}
    scale = float(np.cov(o, n, ddof=0)[0, 1] / np.var(o, ddof=0))
    offset = float(np.mean(n) - scale * np.mean(o))
    res = np.abs(n - (scale * o + offset))
    return {"scale": scale, "offset": offset,
            "max_residual": float(res.max()) if len(res) else 0.0, "n": int(len(o))}


def main() -> int:
    t0 = _time.time()
    v1_manifest = json.loads(
        (paths.TOKENS_V1_DIR / "manifest_tokens.json").read_text(encoding="utf-8"))
    frozen_stats = v1_manifest["normalization_stats"]
    dense_val = bool(v1_manifest["dense_val_enabled"])

    # Apuntar tokenize a pk_v2 (misma rejilla que windows_v4) y contexto v2
    # (mismo contexto que usará la ejecución de producción C2).
    tk.PK_DIR = paths.PK_V2_DIR / "windows"
    tk.CTX_TOKENS = paths.CONTEXT_V2_DIR / "tokens.parquet"
    tk.CTX_VOCAB = paths.CONTEXT_V2_DIR / "vocab.json"
    tk.load_vocab_v1.cache_clear()
    tk.load_context_map.cache_clear()

    context_map = tk.load_context_map()
    cf_meta = tk.load_cf_meta()
    excluded = frozenset(tk.load_cases_without_phase_marks())

    result: dict = {"date": pd.Timestamp.now().isoformat(), "splits": {}}
    for split in ("train", "val"):
        old = _read_tokens_v1_real(split)
        new = _build_new_real(split, dense_val, frozen_stats, context_map,
                              cf_meta, excluded)

        counts_match = (len(old) == len(new)
                        and int((old.dense == new.reset_index(drop=True).dense).all()))
        n_dense_old = int(old.dense.sum())
        n_dense_new = int(new.dense.sum())

        # comparación por columnas exactas (alineadas por (caseid, t0))
        m = old.merge(new, on=["caseid", "t0"], suffixes=("_o", "_n"), how="inner")
        assert len(m) == len(old) == len(new), \
            f"alineación rota en {split}: {len(m)} vs {len(old)}/{len(new)}"

        exact_diffs: dict[str, float] = {}
        checksums: dict[str, dict] = {}
        for c in EXACT_COLS:
            if c in ("caseid", "t0"):
                continue
            exact_diffs[c] = _max_abs_diff(m[f"{c}_o"], m[f"{c}_n"])
            if pd.api.types.is_numeric_dtype(m[f"{c}_n"]):
                arr = m[f"{c}_n"].to_numpy(dtype=np.float64)
                checksums[c] = {
                    "n": int(len(arr)), "sum": float(np.nansum(arr)),
                    "sumsq": float(np.nansum(arr * arr))}

        # ctx binarias: idénticas (patrón de emisión + valor); ctx continuas: afín
        ctx_binary_mismatch = 0
        for i in CTX_BINARY:
            c = f"ctx_{i}"
            a = m[f"{c}_o"].to_numpy(dtype=np.float64)
            b = m[f"{c}_n"].to_numpy(dtype=np.float64)
            a_nan = np.isnan(a)
            b_nan = np.isnan(b)
            ctx_binary_mismatch += int((a_nan != b_nan).sum())
            both = ~a_nan & ~b_nan
            if both.any():
                ctx_binary_mismatch += int((a[both] != b[both]).sum())
        ctx_affine: dict[str, dict] = {}
        for i in CTX_CONTINUOUS:
            c = f"ctx_{i}"
            ctx_affine[i] = _fit_affine(m[f"{c}_o"].to_numpy(dtype=np.float64),
                                        m[f"{c}_n"].to_numpy(dtype=np.float64))

        result["splits"][split] = {
            "n_rows_old": int(len(old)),
            "n_rows_new": int(len(new)),
            "n_dense_old": n_dense_old,
            "n_dense_new": n_dense_new,
            "counts_match": bool(counts_match and n_dense_old == n_dense_new),
            "max_abs_diff_exact": float(max(exact_diffs.values())) if exact_diffs else 0.0,
            "exact_diffs": {k: v for k, v in exact_diffs.items() if v != 0.0},
            "checksums": checksums,
            "ctx_binary_mismatch": ctx_binary_mismatch,
            "ctx_continuous_affine": ctx_affine,
        }

    exact_max = max(s["max_abs_diff_exact"] for s in result["splits"].values())
    affine_max = max(
        (a["max_residual"] for s in result["splits"].values()
         for a in s["ctx_continuous_affine"].values() if np.isfinite(a["max_residual"])),
        default=0.0)
    bin_mismatch = sum(s["ctx_binary_mismatch"] for s in result["splits"].values())
    result["verdict"] = {
        "counts_match": all(s["counts_match"] for s in result["splits"].values()),
        "exact_ok": exact_max < 1e-6,
        "ctx_affine_ok": affine_max < 1e-6,
        "ctx_binary_ok": bin_mismatch == 0,
    }
    result["elapsed_s"] = round(_time.time() - t0, 2)

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps(result["verdict"], indent=2))
    print(f"exact_max={exact_max:.3e} affine_max={affine_max:.3e} "
          f"bin_mismatch={bin_mismatch}")
    print("escrito:", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
