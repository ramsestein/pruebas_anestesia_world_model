"""Reconstruct per-case clinical rows for the vaso reinforcement set.

The vaso generator (scripts/generate_vaso_reinforcement.py, run_batch) seeded
CaseSimulator with config.random_seed = caseid * 77777 and sampled patients from
data/real/clinical_data_enriched.parquet. The clinical row was not persisted, so
we reproduce it deterministically here and save one parquet per case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from anessim.patients import PatientSampler

PROJECT_DIR = Path(__file__).resolve().parents[3]
ENRICHED = PROJECT_DIR / "data" / "real" / "clinical_data_enriched.parquet"
OUT_DIR = PROJECT_DIR / "data" / "synthetic_vaso_reinf" / "clinical"


def main() -> None:
    cases_dir = PROJECT_DIR / "data" / "synthetic_vaso_reinf" / "cases"
    caseids = sorted(int(p.stem) for p in cases_dir.glob("*.parquet"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sampler = PatientSampler(str(ENRICHED))

    for caseid in caseids:
        rng = np.random.default_rng(caseid * 77777)
        patient = sampler.sample(caseid, rng=rng)
        row = patient.to_clinical_dict()
        pl.DataFrame([row]).write_parquet(OUT_DIR / f"{caseid}_clinical.parquet")

    print(f"Wrote {len(caseids)} clinical rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
