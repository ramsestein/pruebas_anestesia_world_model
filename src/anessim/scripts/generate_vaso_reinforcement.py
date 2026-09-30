"""generate_vaso_reinforcement.py — Refuerzo vasoactivo (70% eph, 20% phen, 10% nora).

Uso: python scripts/generate_vaso_reinforcement.py --validate 30
"""

import argparse
import time
from pathlib import Path

import numpy as np

from anessim.simulate import CaseSimulator
from anessim.config import SimulatorConfig

PROJECT_DIR = Path(__file__).resolve().parents[3]

NORA_GAIN = 150.0; PHEN_GAIN = 0.012
# Ephedrine: Emax + tachyphylaxis (matching hemodynamics.py v3.2)
EPH_EMAX_MAP = 30.0; EPH_EMAX_HR = 25.0; EPH_ED50 = 8.0; EPH_TACHY_HALF = 25.0
DRUG_DIST = {"nora": 0.10, "phen": 0.20, "eph": 0.70}


def _decayed_effect(t_target, times, doses, hl_s):
    if hl_s <= 0: return 0.0
    tau = hl_s / np.log(2)
    return sum(d * np.exp(-(t_target - tb) / tau) for tb, d in zip(times, doses) if tb <= t_target)


def generate_one(sim, caseid):
    rng = np.random.default_rng(caseid * 99999)
    result = sim.run(caseid)
    truth = result["truth"]
    t = truth["time"].astype(np.float64); n = len(t)

    tl = result["timeline"]
    maint = None
    for p in tl.phases:
        if p.name == "maintenance": maint = p; break
    if maint is None or maint.end_s - maint.start_s < 360: return None

    ms, me = maint.start_s, maint.end_s
    weight = result["patient"].weight

    nora_rate = np.zeros(n, dtype=np.float64)
    phen_t, phen_d = [], []
    eph_t, eph_d = [], []

    budget = rng.integers(4, 10)
    drugs = rng.choice(["nora","phen","eph"], size=budget,
                       p=[DRUG_DIST["nora"], DRUG_DIST["phen"], DRUG_DIST["eph"]])

    nn, np_, ne = (drugs == "nora").sum(), (drugs == "phen").sum(), (drugs == "eph").sum()

    for _ in range(nn):
        t0 = rng.uniform(ms + 60, me - 600)
        mask = (t >= t0) & (t <= t0 + rng.uniform(180, 600))
        nora_rate[mask] = np.maximum(nora_rate[mask], rng.uniform(0.02, 0.35))

    for _ in range(np_):
        phen_t.append(rng.uniform(ms + 60, me - 300))
        phen_d.append(rng.uniform(50, 200))

    for _ in range(ne):
        eph_t.append(rng.uniform(ms + 60, me - 300))
        eph_d.append(rng.uniform(5, 15))

    map_bl = truth["map_baseline"].astype(np.float64)
    hr_stored = truth["hr"].astype(np.float64)
    map_new = np.zeros(n, dtype=np.float64)
    hr_new = np.zeros(n, dtype=np.float64)

    for i in range(n):
        ti = t[i]
        neff = NORA_GAIN * nora_rate[i]
        peff = PHEN_GAIN * _decayed_effect(ti, phen_t, phen_d, 300.0) / weight
        eactive = _decayed_effect(ti, eph_t, eph_d, 600.0)
        # Ephedrine: Emax + tachyphylaxis (matching hemodynamics.py v3.2)
        if eactive > 0:
            emax_f = 1.0 - np.exp(-eactive / EPH_ED50)
            # Tachyphylaxis from cumulative ORIGINAL dose up to but not including current
            cum_prior = sum(d for tb, d in zip(eph_t, eph_d) if tb < ti)
            tachy = np.exp(-cum_prior / EPH_TACHY_HALF)
            emeff = EPH_EMAX_MAP * emax_f * tachy * 70.0 / weight
            ehreff = EPH_EMAX_HR * emax_f * tachy * 70.0 / weight
        else:
            emeff = 0.0; ehreff = 0.0
        map_new[i] = float(np.clip(map_bl[i] + neff + peff + emeff, 30, 140))
        hr_new[i] = float(np.clip(hr_stored[i] + ehreff, 30, 180))

    truth["map"] = map_new.astype(np.float32)
    truth["hr"] = hr_new.astype(np.float32)
    truth["noradrenaline_rate"] = nora_rate.astype(np.float32)

    pa = np.zeros(n, dtype=np.float32)
    for tb, d in zip(phen_t, phen_d):
        idx = np.searchsorted(t, tb)
        if 0 <= idx < n: pa[idx] += float(d)
    ea = np.zeros(n, dtype=np.float32)
    for tb, d in zip(eph_t, eph_d):
        idx = np.searchsorted(t, tb)
        if 0 <= idx < n: ea[idx] += float(d)
    truth["phenylephrine_dose"] = pa
    truth["ephedrine_dose"] = ea

    active = nora_rate > 0.005
    map_at = map_bl[active] if active.any() else np.array([])
    nc = np.corrcoef(nora_rate, map_bl)[0,1] if n > 10 else 0.0
    ec = np.corrcoef(ea, map_bl)[0,1] if ea.sum() > 0 and n > 10 else 0.0

    return {"truth": truth, "tracks": result["tracks"], "caseid": caseid,
            "patient": result["patient"], "timeline": result["timeline"],
            "actions": result["actions"], "metadata": result.get("metadata"),
            "clinical_row": result["clinical_row"],
            "duration_s": t[-1], "n_anchors": max(0, (n-120)//60+1),
            "map_at_nora": map_at, "nora_max": float(nora_rate.max()),
            "map_min": float(map_new.min()), "map_max": float(map_new.max()),
            "nora_corr": float(nc), "eph_corr": float(ec),
            "nn": nn, "np_": np_, "ne": ne}


def run_validation(n_cases=30):
    print("=== T2 VALIDATION: {} vaso reinforcement cases ===\n".format(n_cases))
    print("Drug dist: 70% eph, 20% phen, 10% nora\n")

    config = SimulatorConfig()
    config.random_seed = 77777
    config.clinical_path = str(PROJECT_DIR / "data" / "real" / "clinical_data_enriched.parquet")
    sim = CaseSimulator(config)

    out_dir = PROJECT_DIR / "data" / "synthetic_vaso_reinf"
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    (out_dir / "truth").mkdir(parents=True, exist_ok=True)

    all_map = []; nc_all = []; ec_all = []
    tw = 0; ok = 0; dtot = {"nora": 0, "phen": 0, "eph": 0}

    for ci in range(1, n_cases + 1):
        r = generate_one(sim, 60000 + ci)
        if r is None:
            print("  Case {}: SKIP".format(60000 + ci)); continue
        ok += 1; tw += r["n_anchors"]
        if len(r["map_at_nora"]) > 0: all_map.extend(r["map_at_nora"].tolist())
        nc_all.append(r["nora_corr"]); ec_all.append(r["eph_corr"])
        for d, k in [("nora","nn"), ("phen","np_"), ("eph","ne")]:
            dtot[d] += r[k]

        import pandas as pd; import polars as pl
        from anessim.render import render_case
        render_case(
            caseid=60000 + ci, tracks=r["tracks"], clinical_row=r["clinical_row"],
            truth=r["truth"], output_dir=out_dir,
            patient=r["patient"], timeline=r["timeline"], actions=r["actions"],
            duration_min=r["duration_s"] / 60.0, metadata=r["metadata"],
        )

        print("  Case {}: {:.0f}s {}w n={} p={} e={} map=[{:.0f},{:.0f}] cn={:.3f} ce={:.3f}"
              .format(60000 + ci, r["duration_s"], r["n_anchors"],
                      r["nn"], r["np_"], r["ne"], r["map_min"], r["map_max"],
                      r["nora_corr"], r["eph_corr"]))

    if ok == 0: print("ERROR: No cases"); return

    print("\n--- Summary ---")
    print("  Generated: {}/{}".format(ok, n_cases))
    print("  Events: nora={} phen={} eph={}".format(dtot["nora"], dtot["phen"], dtot["eph"]))
    print("  Windows: ~{}".format(tw))
    nc_arr = [c for c in nc_all if not np.isnan(c)]
    ec_arr = [c for c in ec_all if not np.isnan(c)]
    print("  NORA-MAP corr: {:.4f} +- {:.4f}".format(np.mean(nc_arr), np.std(nc_arr)))
    print("  EPH-MAP corr:  {:.4f} +- {:.4f}".format(np.mean(ec_arr), np.std(ec_arr)))

    if all_map:
        m = np.array(all_map)
        h = (m < 70).sum(); n_ = ((m>=70)&(m<=100)).sum(); hp = (m>100).sum()
        print("\n  MAP at nora admin:")
        print("    Hypo  (<70):   {:5d} ({:.1f}%)".format(h, 100*h/len(m)))
        print("    Normo (70-100): {:5d} ({:.1f}%)".format(n_, 100*n_/len(m)))
        print("    Hyper (>100):  {:5d} ({:.1f}%)".format(hp, 100*hp/len(m)))
        print("    Non-hypo: {:.1f}%".format(100*(n_+hp)/len(m)))

    wpc = tw / ok
    needed = int(75000 / (wpc * 0.35))
    print("\n  Windows/case: ~{:.0f}".format(wpc))
    print("  Est. cases for ~75k vasoactive: ~{}".format(needed))
    print("  Files: {}".format(out_dir))


def _worker_generate(caseid, out_dir_str, clinical_path_str):
    """Module-level worker for multiprocessing."""
    import pandas as pd
    import polars as pl
    from anessim.render import render_case
    out_dir = Path(out_dir_str)

    config = SimulatorConfig()
    config.random_seed = caseid * 77777
    config.clinical_path = clinical_path_str
    sim = CaseSimulator(config)
    r = generate_one(sim, caseid)
    if r is None:
        return None

    render_case(
        caseid=caseid, tracks=r["tracks"], clinical_row=r["clinical_row"],
        truth=r["truth"], output_dir=out_dir,
        patient=r["patient"], timeline=r["timeline"], actions=r["actions"],
        duration_min=r["duration_s"] / 60.0, metadata=r["metadata"],
    )
    return {"caseid": caseid, "anchors": r["n_anchors"],
            "nn": r["nn"], "np_": r["np_"], "ne": r["ne"]}

def run_batch(n_cases, workers=4, start_caseid=60000, out_dir=None):
    """Generate N vaso reinforcement cases using multiprocessing."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = Path(out_dir) if out_dir else (PROJECT_DIR / "data" / "synthetic_vaso_reinf_v5")
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    (out_dir / "truth").mkdir(parents=True, exist_ok=True)
    clinical_path = str(PROJECT_DIR / "data" / "real" / "clinical_data_enriched.parquet")
    out_dir_str = str(out_dir)

    caseids = list(range(start_caseid, start_caseid + n_cases))
    done = 0; tw = 0; dtot = {"nora": 0, "phen": 0, "eph": 0}
    t0 = time.time()

    print("Generating {} cases with {} workers...".format(n_cases, workers))

    if workers == 1:
        for cid in caseids:
            r = _worker_generate(cid, out_dir_str, clinical_path)
            if r: done += 1; tw += r["anchors"]
            for d, k in [("nora","nn"), ("phen","np_"), ("eph","ne")]:
                dtot[d] += (r or {}).get(k, 0)
            if done % 50 == 0:
                print("  {}/{} cases ({:.0f}s)...".format(done, n_cases, time.time()-t0))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker_generate, cid, out_dir_str, clinical_path): cid
                    for cid in caseids}
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    done += 1; tw += r["anchors"]
                    for d, k in [("nora","nn"), ("phen","np_"), ("eph","ne")]:
                        dtot[d] += r[k]
                if done % 50 == 0:
                    print("  {}/{} cases ({:.0f}s)...".format(done, n_cases, time.time()-t0))

    print("\nDone: {}/{} cases in {:.0f}s".format(done, n_cases, time.time()-t0))
    print("Windows: ~{}".format(tw))
    print("Events: nora={}, phen={}, eph={}".format(dtot["nora"], dtot["phen"], dtot["eph"]))
    print("Files: {}".format(out_dir))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate", type=int, default=0)
    p.add_argument("--cases", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--start-caseid", type=int, default=60000)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    if args.validate > 0:
        run_validation(args.validate)
    elif args.cases > 0:
        run_batch(args.cases, args.workers, args.start_caseid, args.output_dir)
    else:
        print("Use --validate N or --cases N")


if __name__ == "__main__":
    main()
