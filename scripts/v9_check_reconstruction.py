"""Validación numérica de la reconstrucción de v6_validate.py (PARTE 1.3).

Re-ejecuta V1, V2, V3(v6) y V4 del protocolo con la cohorte real de windows_v4
(equivalente a windows_v2) y las cohortes v7 de windows_v4, y compara contra
los valores de data/diagnostics/v7_validate_results.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diagnostics import v6_validate as vv
from diagnostics import cohort_gap as cg

CACHE = json.load(open("data/diagnostics/v7_validate_results.json", encoding="utf-8"))

results = {}

# 0. Excluidos (misma lista del manifest).
excluded = cg.load_excluded_caseids()
results["n_excluded"] = len(excluded)

# 1. Reconstruir la partición holdout (lógica de split_real_caseids) sobre
#    windows_v4 y verificar el sha de la cache.
real_ids = []
for split in ("val",):
    d = Path("data/windows_v4/windows/source=real") / f"split={split}"
    import pyarrow.parquet as pq
    for part in sorted(d.glob("part-*.parquet")):
        df = pq.read_table(part, columns=["caseid", "phase_from_clinical"]).to_pandas()
        df = cg.filter_cells(df, excluded)
        real_ids.extend(int(c) for c in df["caseid"].unique())
real_ids = sorted(set(real_ids))
rng = np.random.default_rng(20260922)
perm = rng.permutation(real_ids)
k = int(round(len(perm) * 0.40))
holdout_ids = sorted(int(c) for c in perm[:k])
calib_ids = sorted(int(c) for c in perm[k:])
holdout_sha = hashlib.sha256(",".join(str(c) for c in holdout_ids).encode()).hexdigest()
results["n_real_caseids"] = len(real_ids)
results["n_holdout"] = len(holdout_ids)
results["n_calib"] = len(calib_ids)
results["holdout_sha_match"] = (holdout_sha == CACHE["meta"]["holdout_caseids_sha256"])

# 2. Cargar real (val) de windows_v4 y filtrar al holdout.
print("cargando real val ...", flush=True)
real_all = vv.load_cells_from(Path("data/windows_v4"), ["real"], "val", excluded)
hmask = np.isin(real_all["caseid"], holdout_ids)
holdout = {k: real_all[k][hmask] for k in ("values", "masks", "caseid")}

# 3. Cargar synth v7 (val).
print("cargando synth v7 val ...", flush=True)
synth = vv.load_cells_from(Path("data/windows_v4"),
                           ["synthetic_v7", "vaso_reinf_v7", "cf_v7"], "val", excluded)

# 4. V1/V2.
print("V1/V2 ...", flush=True)
v1v2 = vv._run_v1_v2(holdout, synth)
results["v1_linear_14_auc"] = v1v2["v1_linear_14"]["auc"]
results["v2_hgb_14_auc"] = v1v2["v2_hgb_14"]["auc"]
results["v2_hgb_28_auc"] = v1v2["v2_hgb_28"]["auc"]

# 5. V3 (solo la rama synth v7; synth_v5 ya no existe).
print("V3 ...", flush=True)
v3 = vv._run_v3(holdout, synth, None)
results["v3_v6_mean"] = v3["v6"]["mean"]
results["v3_v6_median"] = v3["v6"]["median"]

# 6. V4.
print("V4 ...", flush=True)
v4 = vv._run_v4(holdout, synth)
results["v4_w1_norm_BT"] = v4["w1_norm"]["Solar8000/BT"]["w1_norm"]
results["v4_w1_norm_EMG"] = v4["w1_norm"]["BIS/EMG"]["w1_norm"]
results["v4_w1_norm_PIP"] = v4["w1_norm"]["Primus/PIP_MBAR"]["w1_norm"]

# 7. Comparación contra la cache.
def close(a, b, tol=1e-3):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol

comp = {
    "v1_linear_14_auc": (results["v1_linear_14_auc"],
                         CACHE["v1v2"]["v1_linear_14"]["auc"]),
    "v2_hgb_14_auc": (results["v2_hgb_14_auc"],
                      CACHE["v1v2"]["v2_hgb_14"]["auc"]),
    "v2_hgb_28_auc": (results["v2_hgb_28_auc"],
                      CACHE["v1v2"]["v2_hgb_28"]["auc"]),
    "v3_v6_mean": (results["v3_v6_mean"], CACHE["v3"]["v6"]["mean"]),
    "v3_v6_median": (results["v3_v6_median"], CACHE["v3"]["v6"]["median"]),
    "v4_w1_norm_BT": (results["v4_w1_norm_BT"],
                      CACHE["v4"]["w1_norm"]["Solar8000/BT"]["w1_norm"]),
    "v4_w1_norm_EMG": (results["v4_w1_norm_EMG"],
                       CACHE["v4"]["w1_norm"]["BIS/EMG"]["w1_norm"]),
    "v4_w1_norm_PIP": (results["v4_w1_norm_PIP"],
                       CACHE["v4"]["w1_norm"]["Primus/PIP_MBAR"]["w1_norm"]),
}
for k, (got, ref) in comp.items():
    ok = close(got, ref)
    print(f"  {k}: got={got} ref={ref} {'OK' if ok else 'DIFERENTE'}")
    results[f"match_{k}"] = ok

results["all_match"] = all(results[f"match_{k}"] for k in comp)
print("VEREDICTO:", "RECONSTRUCCIÓN VALIDADA" if results["all_match"] else "RECONSTRUCCIÓN NO VALIDADA")

Path("data/diagnostics/v9_reconstruction_check.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
