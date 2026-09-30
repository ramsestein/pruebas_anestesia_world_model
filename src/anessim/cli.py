"""Command-line interface for the procedural simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

from anessim.config import DEFAULT_CONFIG_PATH, SimulatorConfig
from anessim.labs import generate_lab_data
from anessim.render import render_case
from anessim.simulate import CaseSimulator
from anessim.track_presence import TrackPresenceSampler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic VitalDB-style anesthesia cases.")
    parser.add_argument("--n-cases", type=int, default=10, help="Number of cases to generate")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--case-offset", type=int, default=9001, help="Starting case ID")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: config.n_workers)")
    parser.add_argument("--resume", type=lambda x: x.lower() in ("true", "1", "yes"), default=True, help="Skip existing cases")
    return parser


def _generate_one(config: SimulatorConfig, caseid: int, seed: int, presence_map: dict | None = None) -> tuple[int, str]:
    """Simulate and render a single case."""
    cfg = SimulatorConfig.from_dict(config.to_dict())
    cfg.random_seed = seed
    sim = CaseSimulator(cfg)
    result = sim.run(caseid, presence_override=presence_map.get(str(caseid)) if presence_map else None)
    render_case(
        caseid=caseid,
        tracks=result["tracks"],
        clinical_row=result["clinical_row"],
        truth=result["truth"],
        output_dir=cfg.output_dir,
        patient=result["patient"],
        timeline=result["timeline"],
        actions=result["actions"],
        duration_min=result["timeline"].total_duration_s / 60.0,
        metadata=result.get("metadata"),
    )
    return caseid, "ok"


def _should_skip(caseid: int, output_dir: Path, resume: bool) -> bool:
    if not resume:
        return False
    case_path = output_dir / "cases" / f"{caseid:04d}.parquet"
    return case_path.exists()


def _aggregate_clinical(output_dir: Path) -> None:
    """Combine per-case clinical rows into a single VitalDB-style clinical_data file."""
    clinical_dir = output_dir / "clinical"
    if not clinical_dir.exists():
        return
    paths = sorted(clinical_dir.glob("*.parquet"))
    if not paths:
        return
    dfs = [pl.read_parquet(p) for p in paths]
    combined = pl.concat(dfs, how="diagonal_relaxed")
    combined.write_parquet(output_dir / "clinical_data.parquet", compression="zstd")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.config.exists():
        config = SimulatorConfig.from_yaml(args.config)
    else:
        config = SimulatorConfig()

    if args.output_dir is not None:
        config.output_dir = args.output_dir
    config.random_seed = args.seed
    n_workers = args.workers if args.workers is not None else config.n_workers

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(config.output_dir / "config.yaml")

    config.resume = args.resume
    caseids = [args.case_offset + i for i in range(args.n_cases)]
    caseids = [cid for cid in caseids if not _should_skip(cid, config.output_dir, config.resume)]

    if not caseids:
        print("All requested cases already exist. Use --resume false to regenerate.")
        return

    # Precompute track presence assignments once for the whole cohort
    presence_sampler = TrackPresenceSampler(config.cases_dir or "dataset/cases", np.random.default_rng(args.seed))
    all_tracks = list(presence_sampler.prevalence().keys())
    presence_map = {str(cid): presence_sampler.sample(all_tracks, n_total=len(caseids)) for cid in caseids}

    if n_workers > 1:
        results = Parallel(n_jobs=n_workers)(
            delayed(_generate_one)(config, cid, args.seed + cid, presence_map) for cid in tqdm(caseids, desc="Generating cases")
        )
    else:
        results = []
        for cid in tqdm(caseids, desc="Generating cases"):
            results.append(_generate_one(config, cid, args.seed + cid, presence_map))

    _aggregate_clinical(config.output_dir)
    _generate_lab_data(config, caseids)
    print(f"Generated {len(results)} cases in {config.output_dir}")


def _generate_lab_data(config: SimulatorConfig, caseids: list[int]) -> None:
    """Generate a synthetic lab_data file for the produced cases."""
    real_lab_path = config.labs_path
    if not real_lab_path.exists():
        return
    rng = np.random.default_rng(config.random_seed)
    lab_df = generate_lab_data(str(real_lab_path), caseids, rng=rng)
    lab_df.write_parquet(config.output_dir / "lab_data.parquet", compression="zstd")


if __name__ == "__main__":
    main()
