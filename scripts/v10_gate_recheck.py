"""v10 1b: reejecuta los GATE 2 y GATE 3 del ae_bal CONGELADO sobre la
cohorte real de windows_v4, split val. Debe reproducir los valores del
manifest ae_bal (HR 0.394, ART_MBP 0.241, BIS 0.176, ETCO2 0.091; gate3
943 casos / 1 481 996 celdas / 0.3843 / 0.7161)."""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import paths  # noqa: E402

# IMPORTANTE (Windows): importar ae.physio_ae ANTES que torch; el módulo
# importa pyarrow antes que torch (orden seguro) para no romper pq.read_table.
import ae.physio_ae as pa  # noqa: E402
import torch  # noqa: E402  (ya importado por pa en el orden correcto)

# La cohorte real en windows_v4 es equivalente a windows_v2 (v9 1.1); el AE
# lee de WINDOWS_DIR que ahora apunta a windows_v4 vía paths.
pa.WINDOWS_DIR = paths.WINDOWS_DIR / "windows"

device = torch.device("cpu")
pa.OUT_ROOT = paths.AE_V1_DIR  # ae_bal es histórico (data/ae_v1)
model = pa.load_model("ae_bal", "cpu").to(device)
norm_stats = pa.load_norm_stats("ae_bal")
excluded = pa.excluded_caseids()

print("cargando real val (windows_v4) ...", flush=True)
val = pa.load_cells(["real"], "val", excluded)
print(f"celdas real val (mascara mant): {val['values'].shape[0]}",
      f"casos: {len(val['case_ids'])}", flush=True)

print("evaluando gates ...", flush=True)
gates = pa.evaluate(model, norm_stats, val, device)

g2 = gates["gate2"]["real"]
g3 = gates["gate3"]

out = {
    "gate2_real": {t: g2[t] for t in
                   ["Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS",
                    "Primus/ETCO2"]},
    "gate3": g3,
    "n_real_val_cells": int(val["values"].shape[0]),
    "n_real_val_cases": int(len(val["case_ids"])),
}
Path(paths.DIAGNOSTICS_DIR / "v10_gate_recheck.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

print("GATE 2 real (ae_bal congelado, windows_v4):")
for t, v in out["gate2_real"].items():
    print(f"  {t:<24} {v:.4f}")
print("GATE 3:", json.dumps(g3, ensure_ascii=False))
print("celdas:", out["n_real_val_cells"], "casos:", out["n_real_val_cases"])

# Comparación contra el manifest registrado.
m = json.load(open(paths.AE_V1_DIR / "ae_bal" / "manifest_ae.json", encoding="utf-8"))
ref2 = m["gates"]["gate2"]["real"]
ref3 = m["gates"]["gate3"]
print("\ncomparación:")
for t in ["Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS", "Primus/ETCO2"]:
    ok = abs(g2[t] - ref2[t]) <= 1e-3
    print(f"  {t:<24} recalc={g2[t]:.4f} manifest={ref2[t]:.4f} "
          f"{'OK' if ok else 'DIFERENTE'}")
for k in ["n_cases", "n_cells"]:
    ok = g3[k] == ref3[k]
    print(f"  gate3.{k} recalc={g3[k]} manifest={ref3[k]} {'OK' if ok else 'DIF'}")
for k in ["err_hr_with_art", "err_hr_without_art"]:
    ok = abs(g3[k] - ref3[k]) <= 1e-3
    print(f"  gate3.{k} recalc={g3[k]:.4f} manifest={ref3[k]:.4f} "
          f"{'OK' if ok else 'DIF'}")
