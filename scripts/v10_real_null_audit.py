"""Auditoría de nulls/placeholders en casos reales (esquemas heterogéneos).

Por cada columna presente en algún caso: nº de casos con la columna, fracción
de valores finitos (entre las celdas donde la columna existe), casos con la
columna constante (placeholder) y casos todo-cero. Streaming.
"""
import collections
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

import paths

CASES = paths.COHORTS["real"] / "cases"
files = sorted(CASES.glob("*.parquet"))
N = len(files)

# Unión de columnas.
all_cols = set()
for f in files:
    all_cols.update(pq.read_schema(f).names)
all_cols.discard("time")
cols = sorted(all_cols)
print(f"columnas distintas (sin 'time'): {len(cols)}", flush=True)

acc = {c: {"n_cases_present": 0, "n_rows": 0, "n_finite": 0,
           "min": np.inf, "max": -np.inf, "n_cases_constant": 0,
           "n_cases_allzero": 0, "n_cases_allnan": 0}
       for c in cols}

for i, f in enumerate(files):
    df = pq.read_table(f).to_pandas()
    n = len(df)
    present = set(df.columns) - {"time"}
    for c in cols:
        a = acc[c]
        if c not in present:
            a["n_cases_allnan"] += 1
            continue
        a["n_cases_present"] += 1
        a["n_rows"] += n
        v = df[c].to_numpy(dtype=np.float64)
        finite = np.isfinite(v)
        nf = int(finite.sum())
        a["n_finite"] += nf
        if nf == 0:
            a["n_cases_allnan"] += 1
            continue
        vv = v[finite]
        a["min"] = min(a["min"], float(vv.min()))
        a["max"] = max(a["max"], float(vv.max()))
        if len(np.unique(vv)) == 1:
            a["n_cases_constant"] += 1
        if float(vv.min()) == 0.0 and float(vv.max()) == 0.0:
            a["n_cases_allzero"] += 1
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{N}", flush=True)

# Columnas de interés (pipeline): las 14 de la imagen + fármacos.
INTEREST = ["BIS/BIS", "Solar8000/HR", "Solar8000/PLETH_SPO2", "Primus/ETCO2",
            "Primus/PEEP_MBAR", "Primus/PIP_MBAR", "Primus/MV", "Primus/TV",
            "Primus/RR_CO2", "Solar8000/BT", "Solar8000/ART_MBP",
            "Solar8000/ART_SBP", "Solar8000/ART_DBP", "BIS/EMG",
            "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]

print(f"\n=== Columnas de interés (pipeline) ===")
print(f"{'columna':<28} {'%casos':>7} {'%finito':>8} {'min':>9} {'max':>9} "
      f"{'const':>6} {'allzero':>8}")
for c in INTEREST:
    a = acc[c]
    pc = 100.0 * a["n_cases_present"] / N
    pf = 100.0 * a["n_finite"] / a["n_rows"] if a["n_rows"] else 0.0
    print(f"{c:<28} {pc:>6.1f}% {pf:>7.2f}% {a['min']:>9.3f} {a['max']:>9.3f} "
          f"{a['n_cases_constant']:>6} {a['n_cases_allzero']:>8}")

# Columnas constantes en TODOS los casos donde existen (placeholder total).
print(f"\n=== Columnas 100% constantes (placeholder) en sus casos ===")
ph = []
for c in cols:
    a = acc[c]
    if a["n_cases_present"] > 0 and a["n_cases_constant"] == a["n_cases_present"]:
        ph.append(c)
for c in ph:
    a = acc[c]
    print(f"  {c:<28} presente en {a['n_cases_present']} casos, "
          f"constante, valor ~{a['min']:.3f}")

out = {c: acc[c] for c in cols}
pathlib.Path("data/diagnostics/v10_real_null_placeholder_audit.json").write_text(
    json.dumps({"n_cases": N, "n_cols": len(cols), "columns": out},
               indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\naudit guardado: data/diagnostics/v10_real_null_placeholder_audit.json")
