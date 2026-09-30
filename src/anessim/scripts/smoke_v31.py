"""Quick smoke test for generator v3.1 fixes — multi-seed."""
from pathlib import Path

import numpy as np

from anessim.simulate import CaseSimulator
from anessim.config import SimulatorConfig

PROJECT_DIR = Path(__file__).resolve().parents[3]

results = []
for seed in [42, 123, 456, 789, 101112]:
    c = SimulatorConfig()
    c.clinical_path = PROJECT_DIR / "data" / "real" / "clinical_data_enriched.parquet"
    c.labs_path = PROJECT_DIR / "data" / "real" / "lab_data.parquet"
    c.cases_dir = PROJECT_DIR / "data" / "real" / "cases"
    c.n_cases = 1
    c.dt_seconds = 0.5
    c.random_seed = seed

    s = CaseSimulator(c)
    result = s.run(caseid=30000 + seed, duration_min=45)

    truth = result["truth"]
    map_vals = truth["map"]
    nora_rate = truth["noradrenaline_rate"]
    bis_vals = truth["bis"]
    ppf_rate = truth["propofol_rate"]
    ce_prop = truth["ce_propofol"]

    hypo_no_nora = np.sum((map_vals < 60) & (nora_rate < 0.001))
    mask2 = ~np.isnan(map_vals) & ~np.isnan(nora_rate)
    corr_nm = np.corrcoef(map_vals[mask2], nora_rate[mask2])[0, 1]
    mask3 = ~np.isnan(bis_vals) & ~np.isnan(ppf_rate)
    corr_pb = np.corrcoef(bis_vals[mask3], ppf_rate[mask3])[0, 1]

    nora_actions = [a for a in result["actions"] if a.drug == "noradrenaline"]
    exo = sum(1 for a in nora_actions if a.metadata.get("exogenous"))
    react = len(nora_actions) - exo

    results.append({
        "seed": seed,
        "map_mean": np.nanmean(map_vals),
        "map_min": np.nanmin(map_vals),
        "map_max": np.nanmax(map_vals),
        "bis_mean": np.nanmean(bis_vals),
        "ce_prop_mean": np.nanmean(ce_prop),
        "hypo_no_nora_pct": 100 * hypo_no_nora / len(map_vals),
        "corr_nora_map": corr_nm,
        "corr_ppf_bis": corr_pb,
        "nora_exo": exo,
        "nora_react": react,
        "nora_nonzero_pct": 100 * np.mean(nora_rate > 0.001),
    })

print(f"{'Seed':>8} {'MAP':>8} {'MAPmin':>8} {'MAPmax':>8} {'BIS':>8} {'CePPF':>8} {'Hypo%':>8} {'N↔M':>8} {'P↔B':>8} {'Exo':>5} {'React':>6} {'Nora%':>7}")
for r in results:
    print(f"{r['seed']:>8} {r['map_mean']:>7.1f} {r['map_min']:>7.1f} {r['map_max']:>7.1f} "
          f"{r['bis_mean']:>7.1f} {r['ce_prop_mean']:>7.2f} {r['hypo_no_nora_pct']:>7.1f}% "
          f"{r['corr_nora_map']:>7.3f} {r['corr_ppf_bis']:>7.3f} {r['nora_exo']:>5} {r['nora_react']:>6} {r['nora_nonzero_pct']:>6.1f}%")

# Pool all and compute global
all_map = []; all_nora = []; all_bis = []; all_ppf = []
all_hypo = 0; all_total = 0
for seed in [42, 123, 456, 789, 101112]:
    c = SimulatorConfig()
    c.clinical_path = "data/real/clinical_data_enriched.parquet"
    c.labs_path = "data/real/lab_data.parquet"
    c.cases_dir = "data/real/cases"
    c.n_cases = 1
    c.dt_seconds = 0.5
    c.random_seed = seed
    s = CaseSimulator(c)
    result = s.run(caseid=30000 + seed, duration_min=45)
    t = result["truth"]
    all_map.append(t["map"]); all_nora.append(t["noradrenaline_rate"])
    all_bis.append(t["bis"]); all_ppf.append(t["propofol_rate"])
    all_hypo += np.sum((t["map"] < 60) & (t["noradrenaline_rate"] < 0.001))
    all_total += len(t["map"])

am = np.concatenate(all_map); an = np.concatenate(all_nora)
ab = np.concatenate(all_bis); ap = np.concatenate(all_ppf)
m2 = ~np.isnan(am) & ~np.isnan(an)
m3 = ~np.isnan(ab) & ~np.isnan(ap)
print(f"\nPOOLED (5 cases):")
print(f"  NORA-MAP corr: {np.corrcoef(am[m2], an[m2])[0,1]:.4f}")
print(f"  PPF-BIS corr: {np.corrcoef(ab[m3], ap[m3])[0,1]:.4f}")
print(f"  MAP<60 & NORA~0: {all_hypo}/{all_total} ({100*all_hypo/all_total:.1f}%)")
print(f"  MAP max: {np.nanmax(am):.1f}")

