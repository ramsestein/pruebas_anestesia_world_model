"""Inventario de columnas presentes por caso (esquemas heterogéneos)."""
import collections
import pathlib

import pyarrow.parquet as pq

import paths

CASES = paths.COHORTS["real"] / "cases"
files = sorted(CASES.glob("*.parquet"))

col_counts = collections.Counter()
col_to_ncases = collections.Counter()
for i, f in enumerate(files):
    sch = pq.read_schema(f)
    col_counts[len(sch.names)] += 1
    for n in sch.names:
        col_to_ncases[n] += 1
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(files)}", flush=True)

print("\ndistribución de nº de columnas por caso:")
for k in sorted(col_counts):
    print(f"  {k} columnas: {col_counts[k]} casos")

print(f"\ncolumnas presentes en menos de {len(files)} casos:")
for n, c in sorted(col_to_ncases.items(), key=lambda x: x[1]):
    if c < len(files):
        print(f"  {n:<28} {c} casos")

print(f"\ntotal columnas distintas: {len(col_to_ncases)}")
