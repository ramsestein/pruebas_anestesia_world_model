"""Generate the anesthesia timeline (phases, durations, and scheduled events)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Phase:
    name: str
    start_s: float
    end_s: float


@dataclass
class Timeline:
    phases: List[Phase] = field(default_factory=list)
    total_duration_s: float = 0.0

    def get_phase_at(self, t: float) -> str | None:
        for p in self.phases:
            if p.start_s <= t < p.end_s or (t == p.end_s and p == self.phases[-1]):
                return p.name
        return None


class TimelineGenerator:
    """Produce a clinically plausible timeline for one case."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def generate(
        self,
        duration_min: float | None = None,
        preop_min: float = 3.0,
        emergence_min: float = 10.0,
    ) -> Timeline:
        """Create a timeline from induction to emergence."""
        if duration_min is None:
            duration_min = float(self.rng.lognormal(mean=5.25, sigma=0.55))
            duration_min = float(np.clip(duration_min, 45.0, 480.0))

        total_s = duration_min * 60.0
        preop_s = preop_min * 60.0
        emergence_s = emergence_min * 60.0
        intraop_s = max(0.0, total_s - preop_s - emergence_s)

        # Induction occupies first few minutes of intra-op.
        induction_s = float(self.rng.uniform(2.0, 5.0)) * 60.0
        intubation_s = float(self.rng.uniform(0.5, 1.5)) * 60.0
        maintenance_s = max(0.0, intraop_s - induction_s - intubation_s)

        phases = [
            Phase("preop", 0.0, preop_s),
            Phase("induction", preop_s, preop_s + induction_s),
            Phase("intubation", preop_s + induction_s, preop_s + induction_s + intubation_s),
            Phase("maintenance", preop_s + induction_s + intubation_s, preop_s + intraop_s),
            Phase("emergence", preop_s + intraop_s, total_s - emergence_s),
            Phase("extubation", total_s - emergence_s, total_s),
        ]
        return Timeline(phases=phases, total_duration_s=total_s)
