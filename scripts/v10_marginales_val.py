"""v10 1a: marginales de la real SOLO split=val con exclusiones, comparados
uno a uno con la tabla D4 de REPORT_cohort_gap (cache cohort_gap_results.json)."""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diagnostics import cohort_gap as cg
from diagnostics import v6_validate as vv

excluded = cg.load_excluded_caseids()
cells = vv.load_cells_from(Path("data/windows_v4"), ["real"], "val", excluded)
stats = cg.marginal_stats(cells["values"], cells["masks"])

ref = json.load(open("data/diagnostics/cohort_gap_results.json", encoding="utf-8"))
ref_real = ref["d4"]["per_source"]["real"]

print(f"{'variable':<24} {'n_rec':>9} {'n_D4':>9} {'mean_rec':>10} {'mean_D4':>10} "
      f"{'std_rec':>9} {'std_D4':>9} {'d_mean':>9} {'d_std':>9}")
rows = []
for t in cg.IMAGE_TRACKS:
    r = stats[t]
    d = ref_real[t]
    dm = r["mean"] - d["mean"] if r["mean"] is not None and d["mean"] is not None else None
    ds = r["std"] - d["std"] if r["std"] is not None and d["std"] is not None else None
    rows.append({"track": t, "n_rec": r["n"], "n_D4": d["n"],
                 "mean_rec": r["mean"], "mean_D4": d["mean"],
                 "std_rec": r["std"], "std_D4": d["std"],
                 "d_mean": dm, "d_std": ds})
    print(f"{t:<24} {r['n']:>9} {d['n']:>9} "
          f"{r['mean']:>10.3f} {d['mean']:>10.3f} "
          f"{r['std']:>9.3f} {d['std']:>9.3f} "
          f"{('' if dm is None else f'{dm:+.4f}'):>9} "
          f"{('' if ds is None else f'{ds:+.4f}'):>9}")

# Máxima discrepancia relativa en std y mean.
max_dstd = max((abs(x["d_std"]) / x["std_D4"]) for x in rows if x["d_std"] is not None and x["std_D4"])
max_dmean = max(abs(x["d_mean"]) for x in rows if x["d_mean"] is not None)
print(f"\nmáx |Δstd|/std_D4 = {max_dstd:.4f}; máx |Δmean| = {max_dmean:.4f}")
out = {"rows": rows, "max_rel_dstd": max_dstd, "max_abs_dmean": max_dmean,
       "n_real_val_cells": int(cells['values'].shape[0])}
Path("data/diagnostics/v10_marginales_val.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
