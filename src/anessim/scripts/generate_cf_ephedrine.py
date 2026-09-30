"""generate_cf_ephedrine.py — Pares CF de efedrina (evaluacion, held-out).

Genera pares: rama A = bolo efedrina, rama B = nada.
Mismo paciente/seed/historia hasta t_split.

Uso:
  python scripts/generate_cf_ephedrine.py --n-pairs 120 --start-caseid 52000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from anessim.actions import Action, ActionType
from anessim.config import SimulatorConfig
from anessim.simulate import CaseSimulator


def save_case(caseid, result, out_dir):
    """Save tracks and truth as parquet."""
    import polars as pl
    import pandas as pd

    tracks = result["tracks"]
    if isinstance(tracks, dict):
        arrays = {k: v for k, v in tracks.items()
                  if isinstance(v, np.ndarray) and v.ndim == 1}
        pdf = pd.DataFrame(arrays)
        pl.from_pandas(pdf).write_parquet(
            str(out_dir / "cases" / "{}.parquet".format(caseid)))
    else:
        tracks.write_parquet(
            str(out_dir / "cases" / "{}.parquet".format(caseid)))

    truth = result["truth"]
    arrays = {k: v for k, v in truth.items()
              if isinstance(v, np.ndarray) and v.ndim == 1}
    pdf = pd.DataFrame(arrays)
    pl.from_pandas(pdf).write_parquet(
        str(out_dir / "truth" / "{}_truth.parquet".format(caseid)))


def generate_pairs(n_pairs, start_caseid, seed, out_dir):
    rng = np.random.default_rng(seed)
    pairs_meta = []
    success = 0

    for pair_idx in range(n_pairs):
        pair_seed = seed + pair_idx * 1000
        caseid_a = start_caseid + pair_idx * 2
        caseid_b = caseid_a + 1
        t0 = time.time()

        # Preview to find split_t
        cfg = SimulatorConfig()
        cfg.random_seed = pair_seed
        cfg.clinical_path = str(ROOT / "data" / "real" /
                                "clinical_data_enriched.parquet")
        sim = CaseSimulator(cfg)
        preview = sim.run(caseid_a)
        tl = preview["timeline"]

        maint = None
        for p in tl.phases:
            if p.name == "maintenance":
                maint = p
                break
        if maint is None or maint.end_s - maint.start_s < 600:
            print("  Pair {}: SKIP (no maintenance)".format(pair_idx))
            continue

        split_t = rng.uniform(maint.start_s + 300, maint.end_s - 300)
        dose_mg = rng.uniform(5.0, 15.0)

        eph_action = Action(
            t_s=split_t + 10.0,
            action_type=ActionType.VASOACTIVE_BOLUS,
            drug="ephedrine", value=dose_mg, unit="mg", route="IV",
        )

        # Branch A: ephedrine
        cfg_a = SimulatorConfig()
        cfg_a.random_seed = pair_seed
        cfg_a.clinical_path = str(ROOT / "data" / "real" /
                                  "clinical_data_enriched.parquet")
        sim_a = CaseSimulator(cfg_a)
        result_a = sim_a.run(caseid_a, counterfactual_split_t=split_t,
                             counterfactual_actions_override=[eph_action])

        # Branch B: nothing
        cfg_b = SimulatorConfig()
        cfg_b.random_seed = pair_seed
        cfg_b.clinical_path = str(ROOT / "data" / "real" /
                                  "clinical_data_enriched.parquet")
        sim_b = CaseSimulator(cfg_b)
        result_b = sim_b.run(caseid_b, counterfactual_split_t=split_t,
                             counterfactual_actions_override=[])

        # Save
        (out_dir / "cases").mkdir(parents=True, exist_ok=True)
        (out_dir / "truth").mkdir(parents=True, exist_ok=True)
        save_case(caseid_a, result_a, out_dir)
        save_case(caseid_b, result_b, out_dir)

        # Deltas
        t_arr = result_a["truth"]["time"]
        map_a = result_a["truth"]["map"]
        map_b = result_b["truth"]["map"]
        hr_a = result_a["truth"]["hr"]
        hr_b = result_b["truth"]["hr"]

        pre = t_arr <= split_t
        map_pre_diff = float(np.abs(map_a[pre] - map_b[pre]).max())

        post = (t_arr > split_t + 30) & (t_arr < split_t + 600)
        dMAP = float(np.mean(map_a[post] - map_b[post])) if post.any() else 0
        dHR = float(np.mean(hr_a[post] - hr_b[post])) if post.any() else 0
        dMAP_pk = float(np.max(map_a[post] - map_b[post])) if post.any() else 0
        dHR_pk = float(np.max(hr_a[post] - hr_b[post])) if post.any() else 0

        map_at_split = float(map_a[np.searchsorted(t_arr, split_t,
                                                    side="right") - 1])

        manifest = {
            "caseid_a": caseid_a, "caseid_b": caseid_b,
            "split_t_s": split_t, "seed": pair_seed,
            "intervention_type": "ephedrine",
            "ephedrine_dose_mg": dose_mg,
            "intervention_a": [eph_action.summary()],
            "intervention_b": [],
            "map_at_split": map_at_split,
            "delta_map_mean": dMAP, "delta_hr_mean": dHR,
            "delta_map_peak": dMAP_pk, "delta_hr_peak": dHR_pk,
            "pre_diff_map_max": map_pre_diff,
        }

        (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "metadata" / "cf_pair_{}.json".format(caseid_a)
        meta_path.write_text(json.dumps(manifest, indent=2))
        pairs_meta.append(manifest)
        success += 1

        dt = time.time() - t0
        print("  Pair {}: {}ms, cases {}/{}, split={:.0f}s, dose={:.1f}mg, "
              "MAP@split={:.0f}, dMAP={:.2f}, dHR={:.2f}, pre_diff={:.4f}"
              .format(success, int(dt*1000), caseid_a, caseid_b,
                      split_t, dose_mg, map_at_split, dMAP, dHR, map_pre_diff))

    if pairs_meta:
        summary = {
            "n_pairs": len(pairs_meta),
            "intervention_type": "ephedrine",
            "stats": {
                "delta_map_mean": float(np.mean(
                    [p["delta_map_mean"] for p in pairs_meta])),
                "delta_hr_mean": float(np.mean(
                    [p["delta_hr_mean"] for p in pairs_meta])),
                "delta_map_peak_mean": float(np.mean(
                    [p["delta_map_peak"] for p in pairs_meta])),
                "delta_hr_peak_mean": float(np.mean(
                    [p["delta_hr_peak"] for p in pairs_meta])),
                "pre_diff_max": float(np.max(
                    [p["pre_diff_map_max"] for p in pairs_meta])),
            }
        }
        (out_dir / "metadata" / "cf_ephedrine_summary.json").write_text(
            json.dumps(summary, indent=2))

    print("\nGenerated {}/{} ephedrine CF pairs".format(success, n_pairs))
    return pairs_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=120)
    parser.add_argument("--start-caseid", type=int, default=52000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data" / "counterfactual_eval")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_pairs(args.n_pairs, args.start_caseid, args.seed, out_dir)


if __name__ == "__main__":
    main()
