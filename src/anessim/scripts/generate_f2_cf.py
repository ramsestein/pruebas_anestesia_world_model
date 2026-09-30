"""generate_f2_cf.py — F2 counterfactual generation (pharma / vent / learning).

Regenerates the reserved pharmacological pairs, the ventilation families, and
the randomised single-lever learning collection with the F0-fixed simulator.

Usage:
  python scripts/generate_f2_cf.py pharma    --n-pairs 250 --start-caseid 70001 \
      --output-dir D:/data/anestesia_world/cf_pharma_v5 --workers 6
  python scripts/generate_f2_cf.py vent      --n-pairs 150 --start-caseid 80001 \
      --output-dir D:/data/anestesia_world/cf_vent_v5 --workers 6
  python scripts/generate_f2_cf.py learning  --n-pairs 375 --start-caseid 90001 \
      --output-dir D:/data/anestesia_world/cf_learning_v5 --workers 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from anessim.actions import Action, ActionType
from anessim.config import SimulatorConfig
from anessim.render import render_case
from anessim.simulate import CaseSimulator

ROOT = Path(__file__).resolve().parents[3]


def bucket_from_group_key(group_key: str, n_buckets: int = 10000) -> int:
    """Deterministic bucket for a group key using SHA-256.

    This replaces the previous dependency on ``harness_v5.ids`` so that
    counterfactual generation remains self-contained inside the generator.
    """
    h = int(hashlib.sha256(group_key.encode("utf-8")).digest().hex(), 16)
    return h % n_buckets

# ── Intervention definitions ────────────────────────────────────────────────

PHARMA_LEVERS = ["propofol_bolus", "noradrenaline", "remi_up", "ephedrine"]
VENT_LEVERS = ["peep_up", "peep_down", "fio2_down", "sevo_up"]
LEARNING_LEVERS = [
    "ppf20_rate", "rftn20_rate", "ppf_bolus", "remi_bolus",
    "nepi_rate", "phen_rate", "phen_bolus", "eph_bolus",
    "set_fio2", "set_rr", "set_tv", "set_peep", "sevo_mac",
]

# Reserved pharmacological families: 870 pairs total (plan.md F2).
PHARMA_COUNTS = {"propofol_bolus": 250, "noradrenaline": 250, "remi_up": 250, "ephedrine": 120}
COLLECTION_SEED_OFFSETS = {"pharma": 0, "vent": 100_000_000, "learning": 200_000_000}
ROLE_BUCKETS = {
    "cf_train": (0, 6999),
    "cf_dev": (7000, 7999),
    "cf_calibration": (8000, 8999),
    "cf_test": (9000, 9999),
}


def _make_infusion(drug, t, value, unit, kind=ActionType.INFUSION_CHANGE) -> Action:
    return Action(t_s=t, action_type=kind, drug=drug, value=value, unit=unit, route="IV")


def _make_bolus(drug, t, value, unit) -> Action:
    return Action(t_s=t, action_type=ActionType.BOLUS, drug=drug, value=value, unit=unit, route="IV")


def _make_vaso_bolus(drug, t, value, unit) -> Action:
    return Action(t_s=t, action_type=ActionType.VASOACTIVE_BOLUS, drug=drug, value=value, unit=unit, route="IV")


def _make_vaso_infusion(drug, t, value, unit) -> Action:
    return Action(t_s=t, action_type=ActionType.VASOACTIVE_INFUSION, drug=drug, value=value, unit=unit, route="IV")


def _make_sevo(t, value) -> Action:
    return Action(t_s=t, action_type=ActionType.VENTILATOR_SETTING, drug="sevoflurane", value=value, unit="MAC")


def _lever_seed_offset(lever: str) -> int:
    """Deterministic per-lever seed offset (no Python hash())."""
    return int.from_bytes(hashlib.sha256(lever.encode()).digest()[:4], "big") % 7919


def _seed_for_split_role(
    initial_seed: int,
    split_role: str,
    donor_subjectids: np.ndarray,
    used_seeds: set[int],
) -> int:
    """Find a deterministic unique seed whose sampled donor belongs to split_role."""
    low, high = ROLE_BUCKETS[split_role]
    for attempt in range(1_000_000):
        candidate = initial_seed + attempt
        if candidate in used_seeds:
            continue
        donor_row = int(np.random.default_rng(candidate).integers(0, len(donor_subjectids)))
        subjectid = int(donor_subjectids[donor_row])
        bucket = bucket_from_group_key(f"real:subject:{subjectid}")
        if low <= bucket <= high:
            used_seeds.add(candidate)
            return candidate
    raise RuntimeError(f"could not find donor seed for {split_role}")


def build_intervention(
    lever: str,
    patient,
    split_t: float,
    rng: np.random.Generator,
    probe_truth: dict | None = None,
) -> tuple[list[Action], list[Action], dict | None, dict | None]:
    """Return (branch_a_actions, branch_b_actions, vent_override_a, vent_override_b).

    ``probe_truth`` is the truth dict from the probe run, used to compute
    deltas relative to the current state (e.g. sevoflurane MAC, remifentanil
    rate) so the intervention always produces a detectable change.
    """
    w = patient.weight
    t = split_t + 10.0

    if lever == "propofol_bolus":
        return [_make_bolus("propofol", t, 1.0 * w, "mg")], [], None, None
    if lever == "noradrenaline":
        return [_make_vaso_infusion("noradrenaline", t, 0.12, "mcg/kg/min")], [], None, None
    if lever == "remi_up":
        return [_make_infusion("remifentanil", t, 0.2 * w, "mcg/min")], [], None, None
    if lever == "ephedrine":
        dose = float(rng.uniform(5.0, 15.0))
        return [_make_vaso_bolus("ephedrine", t, dose, "mg")], [], None, None

    if lever == "peep_up":
        return [], [], {"peep_delta": 5.0}, None
    if lever == "peep_down":
        return [], [], {"peep_delta": -5.0}, None
    if lever == "fio2_down":
        return [], [], {"fio2_delta": -0.20}, None
    if lever == "sevo_up":
        mac = float(rng.uniform(0.6, 1.5))
        return [_make_sevo(t, mac)], [], None, None

    # Learning levers: single lever, randomised direction/magnitude.
    if lever == "ppf20_rate":
        return [_make_infusion("propofol", t, float(rng.uniform(2.0, 15.0)), "mg/min")], [], None, None
    if lever == "rftn20_rate":
        return [_make_infusion("remifentanil", t, float(rng.uniform(0.02, 0.30)) * w, "mcg/min")], [], None, None
    if lever == "ppf_bolus":
        return [_make_bolus("propofol", t, float(rng.uniform(0.5, 2.0)) * w, "mg")], [], None, None
    if lever == "remi_bolus":
        # Delta relative to baseline: ensure the bolus is large enough to
        # produce a detectable MAP effect even at high baseline rates.
        # Minimum 40 mcg, scaled up if the patient already has a high rate.
        baseline_rate = 0.0
        if probe_truth is not None and "remifentanil_rate" in probe_truth:
            idx = np.searchsorted(probe_truth["time"], split_t)
            if idx < len(probe_truth["remifentanil_rate"]):
                baseline_rate = float(probe_truth["remifentanil_rate"][idx])
        min_bolus = 40.0 + 0.5 * baseline_rate  # scale with baseline
        dose = float(rng.uniform(min_bolus, 120.0))
        return [_make_bolus("remifentanil", t, dose, "mcg")], [], None, None
    if lever == "nepi_rate":
        return [_make_vaso_infusion("noradrenaline", t, float(rng.uniform(0.06, 0.25)), "mcg/kg/min")], [], None, None
    if lever == "phen_rate":
        return [_make_vaso_infusion("phenylephrine", t, float(rng.uniform(40.0, 120.0)), "mcg/min")], [], None, None
    if lever == "phen_bolus":
        return [_make_vaso_bolus("phenylephrine", t, float(rng.uniform(80.0, 250.0)), "mcg")], [], None, None
    if lever == "eph_bolus":
        return [_make_vaso_bolus("ephedrine", t, float(rng.uniform(4.0, 20.0)), "mg")], [], None, None
    if lever == "set_fio2":
        # v5.1: bajar FiO2 a un valor BAJO absoluto (0.18). Antes 0.35 era
        # idéntico al baseline CF (0.35) → palanca sin efecto (no_divergence).
        # Con FiO2 0.18 la PaO2 cae lo bastante para desaturar SpO2 (~90%).
        return [], [], {"fio2_set": 0.18}, None
    if lever == "set_rr":
        return [], [], {"rr_delta": float(rng.uniform(-4.0, 4.0))}, None
    if lever == "set_tv":
        return [], [], {"tv_delta": float(rng.uniform(-150.0, 150.0))}, None
    if lever == "set_peep":
        # Set FiO2 low (0.35) AND apply PEEP delta, so PEEP has headroom.
        return [], [], {"fio2_set": 0.35, "peep_delta": float(rng.uniform(-5.0, 5.0))}, None
    if lever == "sevo_mac":
        # Delta relative to baseline: apply a minimum MAC change of ±0.5
        # relative to the current MAC (if any), so the effect is detectable
        # even when the patient already has sevoflurane.
        # v5.1: elegir la DIRECCIÓN según el baseline para garantizar un cambio
        # real (antes un delta negativo sobre baseline bajo se saturaba a 0 →
        # rama idéntica al control, no_divergence).
        baseline_mac = 0.0
        if probe_truth is not None and "sevoflurane_mac" in probe_truth:
            idx = np.searchsorted(probe_truth["time"], split_t)
            if idx < len(probe_truth["sevoflurane_mac"]):
                baseline_mac = float(probe_truth["sevoflurane_mac"][idx])
        if baseline_mac < 0.4:
            new_mac = float(rng.uniform(0.6, 1.4))
        elif baseline_mac > 1.1:
            new_mac = float(rng.uniform(0.1, max(0.1, baseline_mac - 0.6)))
        else:
            delta = float(rng.uniform(0.5, 1.0)) * rng.choice([-1, 1])
            new_mac = float(np.clip(baseline_mac + delta, 0.1, 1.6))
            if abs(new_mac - baseline_mac) < 0.3:
                new_mac = 1.4 if baseline_mac < 0.7 else 0.2
        return [_make_sevo(t, new_mac)], [], None, None

    raise ValueError(f"unknown lever {lever}")


def _render(result: dict, caseid: int, out_dir: Path) -> None:
    render_case(
        caseid=caseid,
        tracks=result["tracks"],
        clinical_row=result["clinical_row"],
        truth=result["truth"],
        output_dir=out_dir,
        patient=result["patient"],
        timeline=result["timeline"],
        actions=result["actions"],
        duration_min=result["timeline"].total_duration_s / 60.0,
        metadata=result.get("metadata"),
    )


def generate_pair(
    cfg: SimulatorConfig,
    out_dir: Path,
    base_caseid: int,
    lever: str,
    split_t: float,
    presence: list[str],
    branch_a: list[Action],
    branch_b: list[Action],
    vent_a: dict | None,
    vent_b: dict | None,
    seed: int,
    collection: str,
    physiology: dict | None = None,
) -> dict:
    caseid_a = base_caseid
    caseid_b = base_caseid + 1

    cfg_a = SimulatorConfig.from_dict(cfg.to_dict())
    cfg_a.random_seed = seed
    res_a = CaseSimulator(cfg_a).run(
        caseid_a, presence_override=presence,
        counterfactual_split_t=split_t, counterfactual_actions_override=branch_a,
        counterfactual_vent_override=vent_a,
        physiology_override=physiology,
    )

    cfg_b = SimulatorConfig.from_dict(cfg.to_dict())
    cfg_b.random_seed = seed
    res_b = CaseSimulator(cfg_b).run(
        caseid_b, presence_override=presence,
        counterfactual_split_t=split_t, counterfactual_actions_override=branch_b,
        counterfactual_vent_override=vent_b,
        physiology_override=physiology,
    )

    _render(res_a, caseid_a, out_dir)
    _render(res_b, caseid_b, out_dir)

    manifest = {
        "caseid_a": caseid_a,
        "caseid_b": caseid_b,
        "split_t_s": split_t,
        "split_t": split_t,
        "seed": seed,
        "lever": lever,
        "collection": collection,
        "intervention_a": [a.summary() for a in branch_a if a.t_s > split_t],
        "intervention_b": [a.summary() for a in branch_b if a.t_s > split_t],
        "vent_override_a": vent_a,
        "vent_override_b": vent_b,
        "physiology_override": physiology,
    }
    mp = out_dir / "metadata" / f"cf_pair_{caseid_a}.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2))
    return manifest


def _worker(args: dict) -> dict:
    out_dir = Path(args["out_dir"])
    base_caseid = int(args["caseid"])
    lever = args["lever"]
    seed = int(args["seed"])
    collection = args["collection"]

    case_a = out_dir / "cases" / f"{base_caseid}.parquet"
    case_b = out_dir / "cases" / f"{base_caseid + 1}.parquet"
    if case_a.exists() and case_b.exists() and not args.get("force", False):
        return {"caseid": base_caseid, "lever": lever, "skipped": True}

    cfg = SimulatorConfig.from_dict(args["config_dict"])

    # Probe run to get timeline/patient/split_t and a fixed track presence.
    probe_cfg = SimulatorConfig.from_dict(args["config_dict"])
    probe_cfg.random_seed = seed
    probe = CaseSimulator(probe_cfg).run(base_caseid)
    timeline = probe["timeline"]
    patient = probe["patient"]
    maintenance = next(p for p in timeline.phases if p.name == "maintenance")
    split_t = (maintenance.start_s + maintenance.end_s) / 2.0
    presence = [c for c in probe["tracks"] if c != "time"]

    rng = np.random.default_rng(seed)
    branch_a, branch_b, vent_a, vent_b = build_intervention(
        lever, patient, split_t, rng, probe_truth=probe["truth"]
    )

    # Ventilation levers need physiological headroom: force low FiO2 and high
    # shunt so PEEP/FiO2 changes produce a measurable SpO2 effect.
    physiology = None
    if lever in ("set_peep", "set_fio2", "peep_up", "peep_down", "fio2_down"):
        physiology = {"fio2_baseline": 0.35, "shunt_fraction": 0.15}

    try:
        m = generate_pair(
            cfg, out_dir, base_caseid, lever, split_t, presence,
            branch_a, branch_b, vent_a, vent_b, seed, collection,
            physiology=physiology,
        )
        m["skipped"] = False
        return m
    except Exception as exc:  # noqa: BLE001
        return {"caseid": base_caseid, "lever": lever, "error": repr(exc)}


def run_collection(
    levers: list[str],
    counts: dict[str, int],
    start_caseid: int,
    out_dir: Path,
    config_path: Path,
    seed: int,
    workers: int,
    collection: str,
) -> None:
    if config_path.exists():
        cfg = SimulatorConfig.from_yaml(config_path)
    else:
        cfg = SimulatorConfig()
    cfg.output_dir = out_dir
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    (out_dir / "truth").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (out_dir / "clinical").mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(out_dir / "config.yaml")
    config_dict = cfg.to_dict()
    donor_subjectids = pl.read_parquet(
        cfg.clinical_path, columns=["subjectid"]
    )["subjectid"].to_numpy()
    used_seeds: set[int] = set()

    tasks = []
    caseid = start_caseid
    for lever in levers:
        n = counts.get(lever, 0)
        for i in range(n):
            if collection == "learning":
                split_role = "cf_train" if i < 300 else "cf_calibration"
            else:
                split_role = "cf_dev" if i < int(round(0.20 * n)) else "cf_test"
            initial_seed = (
                seed
                + COLLECTION_SEED_OFFSETS[collection]
                + i * 1000
                + _lever_seed_offset(lever)
            )
            tasks.append({
                "out_dir": str(out_dir),
                "caseid": caseid,
                "lever": lever,
                "seed": _seed_for_split_role(
                    initial_seed, split_role, donor_subjectids, used_seeds
                ),
                "collection": collection,
                "config_dict": config_dict,
            })
            caseid += 2

    print(f"Collection '{collection}': {len(tasks)} pairs, {len(levers)} levers, "
          f"{workers} workers, out={out_dir}")

    t0 = time.time()
    done = 0
    ok = 0
    errors = 0
    all_manifests = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, task): task for task in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                done += 1
                if "error" in r:
                    errors += 1
                    print(f"  ERROR {r}", flush=True)
                elif not r.get("skipped"):
                    ok += 1
                    all_manifests.append(r)
                if done % 25 == 0:
                    print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s), ok={ok}, err={errors}", flush=True)
    else:
        for task in tasks:
            r = _worker(task)
            done += 1
            if "error" in r:
                errors += 1
                print(f"  ERROR {r}", flush=True)
            elif not r.get("skipped"):
                ok += 1
                all_manifests.append(r)
            if done % 25 == 0:
                print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s), ok={ok}, err={errors}", flush=True)

    master = {
        "collection": collection,
        "levers": levers,
        "counts": counts,
        "n_pairs": len(tasks),
        "n_ok": ok,
        "n_errors": errors,
        "pairs": all_manifests,
    }
    (out_dir / "metadata" / "counterfactual_manifest.json").write_text(
        json.dumps(master, indent=2)
    )
    print(f"Done '{collection}': ok={ok} err={errors} in {time.time()-t0:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("collection", choices=["pharma", "vent", "learning"])
    p.add_argument("--n-pairs", type=int, default=10)
    p.add_argument("--start-caseid", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    if args.collection == "pharma":
        levers = PHARMA_LEVERS
        counts = dict(PHARMA_COUNTS)
    elif args.collection == "vent":
        levers = VENT_LEVERS
        counts = {lv: args.n_pairs for lv in levers}
    else:
        levers = LEARNING_LEVERS
        counts = {lv: args.n_pairs for lv in levers}

    run_collection(
        levers, counts, args.start_caseid, args.output_dir,
        args.config, args.seed, args.workers, args.collection,
    )


if __name__ == "__main__":
    main()
