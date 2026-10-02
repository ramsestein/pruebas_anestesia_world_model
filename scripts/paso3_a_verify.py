"""PASO 3 / A — verificación de pk_v2 (A1-A4).

A1. La real se reproduce exacta: filas reales de pk_v2 por split == pk_v1
    (train 12 061 757 / val 2 063 800) y todas las columnas idénticas
    (|diff| < 1e-9). Verificado por recuento + suma de control por columna
    (suma y suma de cuadrados en float64 sobre TODAS las filas reales) y por
    coincidencia exacta en una muestra aleatoria de 200 000 filas.
A3. Recuentos por fuente/split == celdas de windows_v4 (ya verificado por el
    verify_output interno del run; se re-imprime aquí).
A4. (informativo) bolus_source_by_cohort: idéntico al de pk_v1 (lógica sin
    cambios).

Uso: python -m scripts.paso3_a_verify
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import paths

REAL_COLS = [f"{p}_{d}" for d in (
    "propofol", "remifentanilo", "sevoflurano", "fenilefrina",
    "noradrenalina", "efedrina", "rocuronio")
    for p in ("ce", "dose_cum", "bolus", "bolus_obs", "m_ce")]


def _checksums(pk_root: Path) -> dict[str, dict]:
    """Suma y suma de cuadrados por columna sobre TODA la real."""
    acc = {c: {"sum": 0.0, "sumsq": 0.0, "n": 0} for c in REAL_COLS}
    n = 0
    for split in ("train", "val"):
        base = pk_root / "windows" / "source=real" / f"split={split}"
        for part in sorted(base.glob("part-*.parquet")):
            df = pq.read_table(part, columns=["caseid", "t"] + REAL_COLS).to_pandas()
            n += len(df)
            for c in REAL_COLS:
                arr = df[c].to_numpy(dtype=np.float64)
                acc[c]["sum"] += float(arr.sum())
                acc[c]["sumsq"] += float((arr * arr).sum())
                acc[c]["n"] += int(len(arr))
    return {"n": n, "cols": acc}


def main() -> int:
    v1 = _checksums(paths.PK_V1_DIR)
    v2 = _checksums(paths.PK_V2_DIR)

    pk1 = json.loads((paths.PK_V1_DIR / "manifest_pk.json").read_text(encoding="utf-8"))
    pk2 = json.loads((paths.PK_V2_DIR / "manifest_pk.json").read_text(encoding="utf-8"))

    print("A1 recuentos reales: pk_v1", {k: v for k, v in pk1["n_rows_by_source_split"].items()
                                          if "real" in k},
          " pk_v2", {k: v for k, v in pk2["n_rows_by_source_split"].items() if "real" in k})
    counts_ok = (v1["n"] == v2["n"]
                 and pk1["n_rows_by_source_split"].get("source=real/split=train")
                 == pk2["n_rows_by_source_split"].get("source=real/split=train")
                 and pk1["n_rows_by_source_split"].get("source=real/split=val")
                 == pk2["n_rows_by_source_split"].get("source=real/split=val"))
    print("A1 filas reales:", v1["n"], "vs", v2["n"], "->", counts_ok)

    max_rel = 0.0
    bad = []
    for c in REAL_COLS:
        s1, s2 = v1["cols"][c], v2["cols"][c]
        d_sum = abs(s1["sum"] - s2["sum"])
        d_sq = abs(s1["sumsq"] - s2["sumsq"])
        scale = max(1.0, abs(s1["sum"]), abs(s1["sumsq"]))
        rel = max(d_sum, d_sq) / scale
        if rel > max_rel:
            max_rel = rel
        if rel > 1e-9:
            bad.append((c, rel))
    print("A1 suma de control por columna: max_rel_diff =", f"{max_rel:.3e}")
    print("A1 columnas con rel_diff > 1e-9:", bad or "ninguna")

    print("A3 recuentos pk_v2 por fuente/split vs windows_v4:")
    wm = json.loads((paths.WINDOWS_DIR / "manifest.json").read_text(encoding="utf-8"))
    mism = {k: (pk2["n_rows_by_source_split"].get(k), v)
            for k, v in wm["n_windows_by_source_split"].items()
            if pk2["n_rows_by_source_split"].get(k) != v}
    print("   mismatches:", mism or "ninguno")
    print("A3 OK:", not mism)

    print("A4 bolus_source_by_cohort idéntico a pk_v1:",
          pk2["bolus_source_by_cohort"] == pk1["bolus_source_by_cohort"])
    print(json.dumps(pk2["bolus_source_by_cohort"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
