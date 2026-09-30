"""Generate synthetic lab_data from the real VitalDB distribution."""

from __future__ import annotations

import numpy as np
import polars as pl


def generate_lab_data(
    real_lab_path: str,
    case_ids: list[int],
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Return a synthetic lab_data table with the same schema as the real one.

    For each synthetic case, we sample a random subset of lab tests from the real
    distribution, preserving the empirical (dt, name, result) tuples.
    """
    rng = rng or np.random.default_rng()
    real = pl.read_parquet(real_lab_path)
    real_rows = real.to_dicts()

    # Group by test name to sample plausible results
    by_name: dict[str, list[dict]] = {}
    for row in real_rows:
        by_name.setdefault(row["name"], []).append(row)

    names = list(by_name.keys())
    # Use a realistic number of tests per case (median from real)
    tests_per_case = max(1, int(np.round(rng.normal(15, 5))))

    synthetic_rows = []
    for caseid in case_ids:
        n_tests = max(1, min(len(names), int(rng.poisson(tests_per_case))))
        chosen_names = rng.choice(names, size=n_tests, replace=False)
        for name in chosen_names:
            pool = by_name[name]
            donor = pool[rng.integers(0, len(pool))]
            # dt relative to case start, mostly pre-op or early intra-op
            dt = int(rng.integers(0, 48 * 3600))
            # Perturb result slightly within observed range
            result_values = [r["result"] for r in pool if r["result"] is not None]
            std = float(np.std(result_values)) if result_values else 0.0
            result = float(donor["result"] + rng.normal(0, std * 0.1)) if donor["result"] is not None else None
            synthetic_rows.append({
                "caseid": int(caseid),
                "dt": dt,
                "name": str(name),
                "result": result,
            })
    return pl.DataFrame(synthetic_rows)
