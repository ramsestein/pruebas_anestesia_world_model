"""End-to-end smoke: N base cases + CF pairs with varied seeds, then audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from anessim.config import SimulatorConfig
from anessim.render import render_case
from anessim.simulate import CaseSimulator
from anessim.scripts.generate_f2_cf import build_intervention, generate_pair

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "_smoke_test"


def make_cfg(seed):
    cfg = SimulatorConfig()
    cfg.random_seed = seed
    cfg.clinical_path = str(ROOT / "data" / "real" / "clinical_data_enriched.parquet")
    cfg.cases_dir = str(ROOT / "data" / "real" / "cases")
    cfg.output_dir = OUT
    return cfg


def gen_base(caseid, seed):
    cfg = make_cfg(seed)
    sim = CaseSimulator(cfg)
    res = sim.run(caseid)
    render_case(
        caseid=caseid, tracks=res["tracks"], clinical_row=res["clinical_row"],
        truth=res["truth"], output_dir=OUT, patient=res["patient"],
        timeline=res["timeline"], actions=res["actions"],
        duration_min=res["timeline"].total_duration_s / 60.0, metadata=res["metadata"],
    )
    return res


def gen_pair(lever, base_caseid, seed):
    cfg = make_cfg(seed)
    probe_cfg = make_cfg(seed)
    probe = CaseSimulator(probe_cfg).run(base_caseid)
    timeline = probe["timeline"]
    patient = probe["patient"]
    maint = next(p for p in timeline.phases if p.name == "maintenance")
    split_t = (maint.start_s + maint.end_s) / 2.0
    presence = [c for c in probe["tracks"] if c != "time"]
    rng = np.random.default_rng(seed)
    branch_a, branch_b, vent_a, vent_b = build_intervention(
        lever, patient, split_t, rng, probe_truth=probe["truth"])
    physiology = None
    if lever in ("set_peep", "set_fio2", "peep_up", "peep_down", "fio2_down"):
        physiology = {"fio2_baseline": 0.35, "shunt_fraction": 0.15}
    generate_pair(cfg, OUT, base_caseid, lever, split_t, presence,
                  branch_a, branch_b, vent_a, vent_b, seed, "pharma",
                  physiology=physiology)


if __name__ == "__main__":
    import sys
    n_base = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for i in range(n_base):
        gen_base(40001 + i, 1000 + i * 37)
    print(f"{n_base} base ok")
    levers = ["propofol_bolus", "noradrenaline", "remi_up", "ephedrine",
              "set_fio2", "sevo_mac"]
    base = 50001
    for i, lv in enumerate(levers):
        gen_pair(lv, base + i * 2, 7000 + i * 17)
        print(f"pair {lv} ok")
    print("done:", OUT)
