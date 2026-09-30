"""Re-simulate the CF learning collection to write spo2/etco2 into truth.

Uses each pair's CURRENT manifest seed (post lineage repair), so the donor
patient, timeline, split_t and intervention are reproduced identically and the
lineage is preserved. Only the rendered truth changes (now includes clean
spo2/etco2). Case parquets are rewritten with identical content.

Usage:
  python scripts/rerender_learning_truth.py --max-pairs 25 --workers 1   # sample
  python scripts/rerender_learning_truth.py --workers 16                 # full
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from anessim.scripts.generate_f2_cf import _worker

LEARNING_DIR = Path("D:/data/anestesia_world/cf_learning_v5")


def load_tasks(
    collection_dir: Path,
    max_pairs: int | None,
    levers: set[str] | None = None,
) -> list[dict]:
    """Build one _worker task per pair from its current manifest.

    Args:
        levers: when set, only re-simulate pairs of these levers.
    """
    master_path = collection_dir / "metadata" / "counterfactual_manifest.json"
    master = json.loads(master_path.read_text(encoding="utf-8"))
    config_dict = None

    tasks = []
    for entry in master["pairs"]:
        if entry.get("skipped"):
            continue
        if levers is not None and entry["lever"] not in levers:
            continue
        if config_dict is None:
            from anessim.config import SimulatorConfig
            config_dict = SimulatorConfig.from_yaml(
                collection_dir / "config.yaml"
            ).to_dict()
        tasks.append({
            "out_dir": str(collection_dir),
            "caseid": int(entry["caseid_a"]),
            "lever": entry["lever"],
            "seed": int(entry["seed"]),
            "collection": "learning",
            "config_dict": config_dict,
            "force": True,
        })
    if max_pairs is not None:
        # Take a per-lever sample so the smoke covers all levers.
        per_lever: dict[str, list[dict]] = {}
        for task in tasks:
            per_lever.setdefault(task["lever"], []).append(task)
        n_per = max(1, max_pairs // max(1, len(per_lever)))
        sampled = [t for lever in sorted(per_lever) for t in per_lever[lever][:n_per]]
        return sampled[:max_pairs]
    return tasks


def _truth_compare(collection_dir: Path, caseid: int, backup_dir: Path) -> dict:
    """Compare new vs backup truth on pre-existing channels (identity check)."""
    new = pl.read_parquet(collection_dir / "truth" / f"{caseid}_truth.parquet")
    old = pl.read_parquet(backup_dir / f"{caseid}_truth.parquet")
    shared = [c for c in old.columns if c in new.columns and c != "time"]
    diffs = {}
    for col in shared:
        try:
            a = old[col].to_numpy().astype(np.float64)
            b = new[col].to_numpy().astype(np.float64)
        except Exception:  # non-numeric (phase)
            continue
        if a.shape != b.shape:
            diffs[col] = "shape"
            continue
        mad = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        if mad > 1e-3:
            diffs[col] = mad
    return {"caseid": caseid, "max_abs_diff": diffs, "ok": not diffs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, default=LEARNING_DIR)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verify-identity", type=Path, default=None,
                        help="backup dir to compare pre-existing truth channels")
    parser.add_argument("--levers", nargs="+", default=None,
                        help="only re-simulate these levers")
    args = parser.parse_args()

    lever_filter = set(args.levers) if args.levers else None
    tasks = load_tasks(args.collection_dir, args.max_pairs, levers=lever_filter)
    print(f"Re-simulating {len(tasks)} pairs with {args.workers} workers", flush=True)

    t0 = time.time()
    done = ok = errors = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                done += 1
                if "error" in r:
                    errors += 1
                    print(f"  ERROR {r}", flush=True)
                elif not r.get("skipped"):
                    ok += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s) ok={ok} err={errors}", flush=True)
    else:
        for task in tasks:
            r = _worker(task)
            done += 1
            if "error" in r:
                errors += 1
                print(f"  ERROR {r}", flush=True)
            elif not r.get("skipped"):
                ok += 1
            if done % 5 == 0:
                print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s) ok={ok} err={errors}", flush=True)

    print(f"Done: ok={ok} err={errors} in {time.time()-t0:.0f}s", flush=True)

    if args.verify_identity is not None and args.verify_identity.exists():
        print("Verifying identity of pre-existing truth channels...", flush=True)
        sample = tasks[: min(10, len(tasks))]
        bad = 0
        for task in sample:
            for cid in (task["caseid"], task["caseid"] + 1):
                res = _truth_compare(args.collection_dir, cid, args.verify_identity)
                if not res["ok"]:
                    bad += 1
                    print(f"  DIFF case {cid}: {res['max_abs_diff']}", flush=True)
        print(f"identity check: {len(sample)*2 - bad}/{len(sample)*2} identical", flush=True)


if __name__ == "__main__":
    main()
