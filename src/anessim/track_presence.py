"""Sample which tracks are present in a synthetic case based on real prevalence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


class TrackPresenceSampler:
    """Decide, per case, which VitalDB tracks are recorded."""

    def __init__(self, real_cases_dir: Path | str, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()
        self.real_cases_dir = Path(real_cases_dir)
        self._prevalence = self._compute_prevalence()

    def _compute_prevalence(self) -> dict[str, float]:
        paths = sorted(self.real_cases_dir.glob("*.parquet"))
        if not paths:
            return {}
        # Use a fixed random sample of real cases for speed and determinism
        rng = np.random.default_rng(2024)
        sample = paths if len(paths) <= 1000 else rng.choice(paths, 1000, replace=False).tolist()
        counts: dict[str, int] = {}
        for p in sample:
            df = pl.read_parquet(p)
            for c in df.columns:
                if c == "time":
                    continue
                counts[c] = counts.get(c, 0) + 1
        total = len(sample)
        return {c: n / total for c, n in counts.items()}

    def sample(self, tracks: list[str], n_total: int | None = None) -> list[str]:
        """Return the subset of tracks that should be present for this case.

        Uses deterministic stratified sampling to keep the synthetic cohort
        prevalence exactly on target and avoid binomial sampling variance.

        Args:
            tracks: All possible track names.
            n_total: Total number of cases in the cohort (used to scale targets).
                     If None, defaults to 100 (legacy behavior for tests/single runs).
        """
        if n_total is None:
            n_total = 100
        present = []
        for track in tracks:
            p = self._prevalence.get(track, 0.0)
            p = max(p, 0.01)
            n_target = int(round(p * n_total))
            n_seen = getattr(self, "_sampled_counts", {}).get(track, 0)
            if n_seen < n_target:
                present.append(track)
                counts = getattr(self, "_sampled_counts", {})
                counts[track] = n_seen + 1
                self._sampled_counts = counts
        return present

    def prevalence(self) -> dict[str, float]:
        return dict(self._prevalence)
