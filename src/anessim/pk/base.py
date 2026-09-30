"""Base PK/PD machinery: 3-compartment model with effect-site."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Infusion:
    """Piecewise constant infusion rate (mg/min or mcg/min)."""

    start_s: float
    end_s: float
    rate: float  # amount per minute
    drug: str

    def rate_at(self, t: float) -> float:
        if self.start_s <= t <= self.end_s:
            return self.rate
        return 0.0


class PKModel:
    """Interface for a PK model."""

    def simulate(
        self,
        infusions: list[Infusion],
        t_eval: np.ndarray,
        initial_state: np.ndarray | None = None,
        boluses: list[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Return state array with columns [C1, C2, C3, Ce]."""
        raise NotImplementedError

class ThreeCompartmentModel(PKModel):
    """Generic 3-compartment + effect-site PK model."""

    def __init__(
        self,
        drug: str,
        v1: float,
        v2: float,
        v3: float,
        k10: float,
        k12: float,
        k13: float,
        k21: float,
        k31: float,
        ke0: float,
    ) -> None:
        self.drug = drug
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.k10 = k10
        self.k12 = k12
        self.k13 = k13
        self.k21 = k21
        self.k31 = k31
        self.ke0 = ke0

    def _ode(self, t: float, y: np.ndarray, infusions: list[Infusion]) -> np.ndarray:
        c1, c2, c3, ce = y
        # Total input rate converted to mg/min divided by v1 -> (mg/min)/L = mg/L/min = concentration/min
        total_rate = sum(i.rate_at(t) for i in infusions if i.drug == self.drug)  # mg/min
        # Convert to concentration change per minute in V1
        input_c1 = total_rate / self.v1

        dc1 = (
            input_c1
            + self.k21 * c2 * (self.v2 / self.v1)
            + self.k31 * c3 * (self.v3 / self.v1)
            - (self.k10 + self.k12 + self.k13) * c1
        )
        dc2 = self.k12 * c1 * (self.v1 / self.v2) - self.k21 * c2
        dc3 = self.k13 * c1 * (self.v1 / self.v3) - self.k31 * c3
        dce = self.ke0 * (c1 - ce)
        return np.array([dc1, dc2, dc3, dce])

    def simulate(
        self,
        infusions: list[Infusion],
        t_eval: np.ndarray,
        initial_state: np.ndarray | None = None,
        boluses: list[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Return state array with columns [C1, C2, C3, Ce] on the t_eval grid.

        Fixed-step RK4 integration (A6): every step depends only on the previous
        state, so the pre-split prefix is bit-identical between two CF branches
        with the same seed even when the post-split infusions differ (the ODE is
        identical pre-split). Boluses are applied as instantaneous jumps of C1 at
        the nearest grid index (dose / V1).
        """
        if initial_state is None:
            initial_state = np.zeros(4)
        state = np.array(initial_state, dtype=np.float64)
        boluses = boluses or []
        t_min = t_eval / 60.0
        infusions_min = [
            Infusion(start_s=i.start_s / 60.0, end_s=i.end_s / 60.0, rate=i.rate, drug=i.drug)
            for i in infusions
        ]
        bolus_jumps: dict[int, float] = {}
        for t_s, dose in boluses:
            idx = int(np.argmin(np.abs(t_eval - t_s)))
            bolus_jumps[idx] = bolus_jumps.get(idx, 0.0) + dose / self.v1

        n = len(t_eval)
        states = np.zeros((n, 4), dtype=np.float64)
        for i in range(n):
            if i in bolus_jumps:
                state[0] += bolus_jumps[i]
            states[i] = state.copy()
            if i < n - 1:
                h = t_min[i + 1] - t_min[i]
                if h <= 0.0:
                    continue
                k1 = self._ode(t_min[i], state, infusions_min)
                k2 = self._ode(t_min[i] + h / 2.0, state + h / 2.0 * k1, infusions_min)
                k3 = self._ode(t_min[i] + h / 2.0, state + h / 2.0 * k2, infusions_min)
                k4 = self._ode(t_min[i] + h, state + h * k3, infusions_min)
                state = state + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return states

    @property
    def cp(self, state: np.ndarray) -> float:
        return float(state[0])

    @property
    def ce(self, state: np.ndarray) -> float:
        return float(state[3])
