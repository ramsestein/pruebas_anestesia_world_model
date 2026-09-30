"""generate_counterfactuals.py — Paired bifurcated trajectories for evaluation.

Generates PAIRS: same patient (same seed, same IIV, same history up to T),
then bifurcates with different actions:
  - propofol bolus vs a_null
  - noradrenaline vs nothing (same hemodynamic state)
  - raise remi vs don't raise, under same stimulus

This is TRUE counterfactual ground truth — what no clinical data can ever give.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anessim.actions import Action, ActionType
from anessim.cli import _generate_one
from anessim.config import SimulatorConfig
from anessim.simulate import CaseSimulator


def generate_counterfactual_pair(
    config: SimulatorConfig,
    base_caseid: int,
    split_t: float,
    branch_a_actions: list[Action],
    branch_b_actions: list[Action],
    seed: int,
) -> tuple[int, int]:
    """Generate a counterfactual pair.

    Both branches share the same patient, timeline, and history up to split_t.
    After split_t, they diverge with different actions.

    Returns:
        (caseid_a, caseid_b) — the two generated case IDs.
    """
    caseid_a = base_caseid
    caseid_b = base_caseid + 1

    # Branch A: baseline
    cfg_a = SimulatorConfig.from_dict(config.to_dict())
    cfg_a.random_seed = seed
    sim_a = CaseSimulator(cfg_a)
    result_a = sim_a.run(
        caseid_a,
        counterfactual_split_t=split_t,
        counterfactual_actions_override=branch_a_actions,
    )

    # Branch B: same seed, bifurcated actions
    cfg_b = SimulatorConfig.from_dict(config.to_dict())
    cfg_b.random_seed = seed  # SAME seed → same patient, IIV, history
    sim_b = CaseSimulator(cfg_b)
    result_b = sim_b.run(
        caseid_b,
        counterfactual_split_t=split_t,
        counterfactual_actions_override=branch_b_actions,
    )

    # Render both (reuse _generate_one logic or render directly)
    from anessim.render import render_case
    render_case(
        caseid=caseid_a,
        tracks=result_a["tracks"],
        clinical_row=result_a["clinical_row"],
        truth=result_a["truth"],
        output_dir=config.output_dir,
        patient=result_a["patient"],
        timeline=result_a["timeline"],
        actions=result_a["actions"],
        duration_min=result_a["timeline"].total_duration_s / 60.0,
        metadata=result_a.get("metadata"),
    )
    render_case(
        caseid=caseid_b,
        tracks=result_b["tracks"],
        clinical_row=result_b["clinical_row"],
        truth=result_b["truth"],
        output_dir=config.output_dir,
        patient=result_b["patient"],
        timeline=result_b["timeline"],
        actions=result_b["actions"],
        duration_min=result_b["timeline"].total_duration_s / 60.0,
        metadata=result_b.get("metadata"),
    )

    # Write pair manifest
    manifest_path = config.output_dir / "metadata" / f"cf_pair_{base_caseid}.json"
    manifest = {
        "caseid_a": caseid_a,
        "caseid_b": caseid_b,
        "split_t_s": split_t,
        "seed": seed,
        "intervention_a": [a.summary() for a in branch_a_actions if a.t_s > split_t],
        "intervention_b": [a.summary() for a in branch_b_actions if a.t_s > split_t],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return caseid_a, caseid_b


def build_propofol_bolus_vs_null(
    timeline, patient, split_t: float, rng: np.random.Generator,
) -> tuple[list[Action], list[Action]]:
    """Branch A: propofol bolus. Branch B: nothing (a_null)."""
    bolus = Action(
        t_s=split_t + 10.0,
        action_type=ActionType.BOLUS,
        drug="propofol",
        value=1.0 * patient.weight,  # 1 mg/kg
        unit="mg",
        route="IV",
    )
    # Both branches: no changes after split except A gets bolus
    branch_a = [bolus]
    branch_b = []
    return branch_a, branch_b


def build_noradrenaline_vs_nothing(
    timeline, patient, split_t: float, rng: np.random.Generator,
) -> tuple[list[Action], list[Action]]:
    """Branch A: noradrenaline infusion. Branch B: nothing."""
    nora = Action(
        t_s=split_t + 10.0,
        action_type=ActionType.VASOACTIVE_INFUSION,
        drug="noradrenaline",
        value=0.08,  # mcg/kg/min
        unit="mcg/kg/min",
        route="IV",
    )
    branch_a = [nora]
    branch_b = []
    return branch_a, branch_b


def build_remi_up_vs_no_up(
    timeline, patient, split_t: float, rng: np.random.Generator,
) -> tuple[list[Action], list[Action]]:
    """Branch A: increase remi. Branch B: don't."""
    remi_up = Action(
        t_s=split_t + 10.0,
        action_type=ActionType.INFUSION_CHANGE,
        drug="remifentanil",
        value=0.2 * patient.weight,  # high remi
        unit="mcg/min",
        route="IV",
    )
    branch_a = [remi_up]
    branch_b = []
    return branch_a, branch_b


INTERVENTIONS = {
    "propofol_bolus": build_propofol_bolus_vs_null,
    "noradrenaline": build_noradrenaline_vs_nothing,
    "remi_up": build_remi_up_vs_no_up,
}


def main():
    parser = argparse.ArgumentParser(description="Generate counterfactual evaluation pairs")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "counterfactual_eval")
    parser.add_argument("--n-pairs", type=int, default=10,
                        help="Number of pairs PER intervention type")
    parser.add_argument("--intervention", type=str, nargs="*",
                        choices=list(INTERVENTIONS.keys()) + ["all"],
                        default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-caseid", type=int, default=20001)
    args = parser.parse_args()

    if args.config.exists():
        config = SimulatorConfig.from_yaml(args.config)
    else:
        config = SimulatorConfig()
    config.output_dir = args.output_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    interventions = (list(INTERVENTIONS.keys()) if "all" in args.intervention
                     else args.intervention)

    next_caseid = args.start_caseid
    all_pairs = []

    for int_name in interventions:
        build_fn = INTERVENTIONS[int_name]
        print(f"\nGenerating {args.n_pairs} pairs for: {int_name}")

        for i in range(args.n_pairs):
            seed = args.seed + i * 100
            cfg = SimulatorConfig.from_dict(config.to_dict())
            cfg.random_seed = seed
            sim = CaseSimulator(cfg)

            # Generate a baseline case first to get timeline and patient
            temp_result = sim.run(next_caseid)

            timeline = temp_result["timeline"]
            patient = temp_result["patient"]
            maintenance = next(p for p in timeline.phases if p.name == "maintenance")

            # Split time: middle of maintenance
            split_t = (maintenance.start_s + maintenance.end_s) / 2.0

            branch_a, branch_b = build_fn(timeline, patient, split_t, rng)

            cid_a, cid_b = generate_counterfactual_pair(
                config, next_caseid, split_t, branch_a, branch_b, seed,
            )
            all_pairs.append({
                "pair_id": i,
                "intervention": int_name,
                "caseid_a": cid_a,
                "caseid_b": cid_b,
                "split_t_s": split_t,
            })
            next_caseid = max(cid_a, cid_b) + 1
            print(f"  Pair {i+1}/{args.n_pairs}: cases {cid_a} vs {cid_b}")

    # Write master manifest
    manifest_path = config.output_dir / "metadata" / "counterfactual_manifest.json"
    manifest = {
        "n_pairs_per_intervention": args.n_pairs,
        "interventions": interventions,
        "pairs": all_pairs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nGenerated {len(all_pairs)} counterfactual pairs in {config.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
