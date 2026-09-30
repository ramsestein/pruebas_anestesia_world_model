"""Verificación de equivalencia de la cohorte REAL entre windows_v2 y windows_v4.

PARTE 1.1 de la tarea v9. Compara contra los manifiestos conservados y calcula
estadísticos/máscaras de la cohorte real de windows_v4.
"""
import hashlib
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

ROOT = pathlib.Path("data")

def sha256(p):
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

out = {}

# 1. split.parquet actual
out["split_parquet_sha256_actual"] = sha256(ROOT / "windows_v4" / "split.parquet")

# 2. tokens manifest (referencia de windows_v2)
tm = json.load(open(ROOT / "tokens_v1" / "manifest_tokens.json", encoding="utf-8"))
out["split_parquet_sha256_ref"] = tm["sha256_split_parquet"]
out["windows_v2_n_cells_real"] = {
    k: v for k, v in tm["windows_v2_n_cells_by_source_split"].items()
    if k.startswith("source=real")
}
out["excluded_no_phase_marks_caseids"] = tm["excluded_no_phase_marks_caseids"]
out["sin_celdas_caseids"] = tm["sin_celdas_caseids"]

# 3. manifest windows_v4
wm = json.load(open(ROOT / "windows_v4" / "manifest.json", encoding="utf-8"))
out["windows_v4_n_windows_real"] = {
    k: v for k, v in wm["n_windows_by_source_split"].items()
    if k.startswith("source=real")
}
out["windows_v4_n_cases_real"] = {
    k: v for k, v in wm["n_cases_by_source_split"].items() if k.startswith("real|")
}

# 4. split.parquet: sujetos reales por split
sp = pq.read_table(ROOT / "windows_v4" / "split.parquet").to_pandas()
real_sp = sp[sp["source"] == "real"]
out["real_subjects_by_split"] = real_sp.groupby("split").size().to_dict()
out["real_subjects_total"] = int(len(real_sp))
out["split_parquet_cols"] = list(sp.columns)

# 5. cases.parquet: caseids reales
cp = pq.read_table(ROOT / "windows_v4" / "cases.parquet").to_pandas()
real_cp = cp[cp["source"] == "real"]
real_ids = sorted(real_cp["caseid"].tolist())
out["real_n_cases"] = int(len(real_ids))
out["real_caseid_min"] = int(real_ids[0]) if real_ids else None
out["real_caseid_max"] = int(real_ids[-1]) if real_ids else None
out["real_caseid_contiguous_1_to_6388"] = (
    real_ids == list(range(1, 6389))
)

# 6. norm_stats AE (ae_bal)
ns = json.load(open(ROOT / "ae_v1" / "ae_bal" / "norm_stats.json", encoding="utf-8"))
out["ae_bal_norm_stats"] = ns

# 7. Estadísticos y máscaras de la cohorte REAL de windows_v4.
IMAGE_TRACKS = list(ns.keys())
VALUE_COLS = list(IMAGE_TRACKS)
MASK_COLS = [f"m_{t}" for t in IMAGE_TRACKS]
cols = ["caseid", "source", "phase_from_clinical"] + VALUE_COLS + MASK_COLS

# Acumuladores por variable: n_total, n_mask1, sum, sumsq (sobre máscara 1).
n_total = np.zeros(len(IMAGE_TRACKS), dtype=np.int64)
n_mask1 = np.zeros(len(IMAGE_TRACKS), dtype=np.int64)
sumv = np.zeros(len(IMAGE_TRACKS), dtype=np.float64)
sumsq = np.zeros(len(IMAGE_TRACKS), dtype=np.float64)
n_cells = 0
n_maint = 0
for split in ("train", "val"):
    d = ROOT / "windows_v4" / "windows" / "source=real" / f"split={split}"
    for part in sorted(d.glob("part-*.parquet")):
        df = pq.read_table(part, columns=cols).to_pandas()
        n_cells += len(df)
        maint = df[df["phase_from_clinical"] == "maintenance"]
        n_maint += len(maint)
        for j, t in enumerate(IMAGE_TRACKS):
            v = maint[VALUE_COLS[j]].to_numpy(dtype=np.float64)
            m = maint[MASK_COLS[j]].to_numpy(dtype=np.uint8)
            n_total[j] += len(v)
            ok = (m == 1) & np.isfinite(v)
            n_mask1[j] += int(ok.sum())
            vv = v[ok]
            sumv[j] += float(vv.sum())
            sumsq[j] += float((vv * vv).sum())

out["real_n_cells_all"] = int(n_cells)
out["real_n_cells_maintenance"] = int(n_maint)
mask_frac = {IMAGE_TRACKS[j]: (float(n_mask1[j] / n_total[j]) if n_total[j] else None)
             for j in range(len(IMAGE_TRACKS))}
out["real_mask_frac_m1"] = mask_frac
real_stats = {}
for j, t in enumerate(IMAGE_TRACKS):
    n = n_mask1[j]
    if n == 0:
        real_stats[t] = {"n": 0, "mean": None, "std": None}
        continue
    mean = sumv[j] / n
    var = max(sumsq[j] / n - mean * mean, 0.0)
    real_stats[t] = {"n": int(n), "mean": float(mean), "std": float(np.sqrt(var))}
out["real_stats_maintenance_mask1"] = real_stats

pathlib.Path("data/diagnostics/v9_real_equivalence.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("OK. n_cells_all=", out["real_n_cells_all"],
      "n_maint=", out["real_n_cells_maintenance"])
print("contiguous:", out["real_caseid_contiguous_1_to_6388"])
