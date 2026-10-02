"""PASO 3 / C3 — verificación de pares contrafactuales en tokens_v2.

Bloqueantes:
  1. Ninguna palanca fuera de LEVER_GROUP.
  2. DIVERGENCIA DE PREFIJO: en todo par utilizable, las ventanas con
     post_split=False de las dos ramas son idénticas en TODAS las columnas de
     features y máscaras (salvo cf_role). 100 % de los pares.
  3. LA PALANCA TIENE EFECTO: en la primera ventana post-split (la que contiene
     split_t), las features del lever_group de su palanca difieren entre ramas.
     Umbral >= 99 % de los pares, POR PALANCA.

Escribe manifests/tokens_v2_cf_pairs.json con la tabla por palanca y los pares
no utilizables con su causa.

Uso:
  python -m scripts.paso3_c3_cf_pairs   (requiere data/tokens_v2/pairs.parquet)
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

OUT_JSON = paths.MANIFESTS_DIR / "tokens_v2_cf_pairs.json"

DRUG_FEATURES = tuple(tk.DRUG_FEATURES)
FEATURE_COLS = tuple(tk.feature_columns())
MASK_COLS = tuple(tk.mask_columns())
COMPARE_COLS = FEATURE_COLS + MASK_COLS

# features del lever_group de cada palanca (para el criterio "tiene efecto").
LEVER_FEATURES: dict[str, tuple[str, ...]] = {
    d: tuple(f"drug_{d}_{f}" for f in DRUG_FEATURES) for d in tk.DRUGS
}
LEVER_FEATURES["ventilacion"] = tuple(
    f"vent_{v}_{t}" for v in tk.VENT_ITEMS for t in ("t0", "t1"))


def _case_to_part(cf_parts: list[Path]) -> dict[int, Path]:
    case_to_part: dict[int, Path] = {}
    for part in cf_parts:
        df = pq.ParquetFile(part).read(columns=["caseid"]).to_pandas()
        for cid in df.caseid.unique():
            case_to_part[int(cid)] = part
    return case_to_part


class PartitionCache:
    def __init__(self, cols: list[str], max_entries: int = 3):
        self.cols = cols
        self.max_entries = max_entries
        self._cache: dict[Path, pd.DataFrame] = {}

    def get(self, part: Path, caseid: int) -> pd.DataFrame:
        df = self._cache.get(part)
        if df is None:
            df = pq.read_table(part, columns=self.cols).to_pandas()
            df = df[df.dense == False]
            if len(self._cache) >= self.max_entries:
                self._cache.clear()
            self._cache[part] = df
        return df[df.caseid == caseid].sort_values("t0", kind="stable").reset_index(drop=True)


def main() -> int:
    t0 = _time.time()
    pairs = pq.read_table(paths.TOKENS_V2_DIR / "pairs.parquet").to_pandas()
    pairs = pairs.sort_values("pair_id").reset_index(drop=True)

    # 1. palancas fuera de LEVER_GROUP (defensa; build_pairs_parquet ya aborta)
    unknown = set(pairs.lever.unique()) - set(tk.LEVER_GROUP)
    assert not unknown, f"palancas fuera de LEVER_GROUP: {sorted(unknown)}"

    cf_parts = sorted((paths.TOKENS_V2_DIR / "windows").glob(
        "source=cf_v7/split=*/part-*.parquet"))
    case_to_part = _case_to_part(cf_parts)
    cols = ["caseid", "t0", "dense", "post_split"] + list(COMPARE_COLS)
    cache = PartitionCache(cols)

    # pares no utilizables y su causa
    unusable = pairs[~pairs.usable]
    unusable_rows = []
    for _, r in unusable.iterrows():
        if int(r.n_windows_total) == 0:
            cause = "sin_ventanas"
        elif int(r.n_windows_base) != int(r.n_windows_intervencion):
            cause = "n_windows_desigual"
        else:
            cause = "t0_divergentes"
        unusable_rows.append({"pair_id": int(r.pair_id), "lever": str(r.lever),
                              "cause": cause})

    # por palanca: pares utilizables y con efecto
    per_lever: dict[str, dict] = {}
    prefix_failures: list[dict] = []
    lever_failures: list[dict] = []
    n_usable = 0

    for _, r in pairs.iterrows():
        if not bool(r.usable):
            continue
        n_usable += 1
        pid = int(r.pair_id)
        base = int(r.caseid_base)
        inter = int(r.caseid_intervencion)
        lever = str(r.lever)
        group = str(r.lever_group)
        db = cache.get(case_to_part[base], base)
        di = cache.get(case_to_part[inter], inter)
        m = db[["t0", "post_split"] + list(COMPARE_COLS)].merge(
            di[["t0"] + list(COMPARE_COLS)], on="t0", suffixes=("_b", "_i"))

        # 2. prefijo idéntico (NaN-aware: ctx_* pueden ser NaN en ambas ramas)
        pre = m[m.post_split == False]
        for c in COMPARE_COLS:
            if not np.array_equal(pre[f"{c}_b"].to_numpy(),
                                  pre[f"{c}_i"].to_numpy(), equal_nan=True):
                prefix_failures.append({"pair_id": pid, "lever": lever, "col": c})
                break

        # 3. efecto de la palanca
        post = m[m.post_split == True].sort_values("t0")
        effect_first = False
        effect_any = False
        if len(post):
            first = post.iloc[0]
            effect_first = any(
                not np.isclose(float(first[f"{c}_b"]), float(first[f"{c}_i"]),
                               rtol=0.0, atol=0.0, equal_nan=True)
                for c in LEVER_FEATURES[group])
            effect_any = any(
                not np.isclose(float(row[f"{c}_b"]), float(row[f"{c}_i"]),
                               rtol=0.0, atol=0.0, equal_nan=True)
                for _, row in post.iterrows() for c in LEVER_FEATURES[group])
        if not effect_any:
            lever_failures.append({"pair_id": pid, "lever": lever, "group": group})
        d = per_lever.setdefault(lever, {"usable": 0, "effect": 0, "effect_any": 0,
                                          "group": group})
        d["usable"] += 1
        d["effect"] += 1 if effect_first else 0
        d["effect_any"] += 1 if effect_any else 0

    table = []
    for lever in sorted(per_lever):
        d = per_lever[lever]
        pct = 100.0 * d["effect"] / d["usable"] if d["usable"] else 0.0
        pct_any = 100.0 * d["effect_any"] / d["usable"] if d["usable"] else 0.0
        table.append({"lever": lever, "lever_group": d["group"],
                      "usable": d["usable"], "effect": d["effect"],
                      "pct": round(pct, 4), "effect_any": d["effect_any"],
                      "pct_any": round(pct_any, 4)})

    prefix_ok = len(prefix_failures) == 0
    effect_ok = all(d["effect"] / d["usable"] >= 0.99
                    for d in per_lever.values() if d["usable"])

    result = {
        "date": pd.Timestamp.now().isoformat(),
        "n_pairs": int(len(pairs)),
        "n_usable": n_usable,
        "n_unusable": int(len(unusable)),
        "prefix_identical_all_pairs": prefix_ok,
        "n_prefix_failures": len(prefix_failures),
        "prefix_failures": prefix_failures[:20],
        "effect_ge_99pct_all_levers": effect_ok,
        "lever_failures": lever_failures[:50],
        "per_lever": table,
        "unusable": unusable_rows,
        "elapsed_s": round(_time.time() - t0, 2),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps({
        "n_pairs": result["n_pairs"], "n_usable": n_usable,
        "n_unusable": result["n_unusable"],
        "prefix_identical_all_pairs": prefix_ok,
        "n_prefix_failures": len(prefix_failures),
        "effect_ge_99pct_all_levers": effect_ok,
        "n_lever_failures": len(lever_failures),
    }, indent=2))
    print("tabla por palanca:")
    for row in table:
        print(f"  {row['lever']:<16s} usable={row['usable']:>4d} "
              f"effect={row['effect']:>4d} pct={row['pct']:.2f} "
              f"effect_any={row['effect_any']:>4d} pct_any={row['pct_any']:.2f}")
    print("escrito:", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
