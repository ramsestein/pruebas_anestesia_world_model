"""Domain randomization and policy perturbation utilities."""

from __future__ import annotations

from typing import Callable

import numpy as np

from anessim.actions import Action, ActionType


class Randomization:
    """Perturb a deterministic clinical policy with realistic variability."""

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        policy_fraction: float = 0.75,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        self.policy_fraction = policy_fraction

    def perturb_infusion_rate(self, rate: float, base_cv: float = 0.15) -> float:
        """Apply log-normal variability to a target infusion rate."""
        return float(rate * self.rng.lognormal(0.0, base_cv))

    def maybe_drop_action(self, probability: float = 0.05) -> bool:
        return self.rng.random() < probability

    def add_random_bolus(
        self,
        actions: list[Action],
        t_window: tuple[float, float],
        drug: str,
        dose_mean: float,
        dose_std: float,
        probability: float = 0.3,
        unit: str = "mg",
    ) -> list[Action]:
        """Stochastically add a bolus within a time window."""
        if self.rng.random() < probability:
            t = self.rng.uniform(t_window[0], t_window[1])
            dose = max(0.0, self.rng.normal(dose_mean, dose_std))
            actions.append(
                Action(
                    t_s=t,
                    action_type=ActionType.BOLUS,
                    drug=drug,
                    value=dose,
                    unit=unit,
                    route="IV",
                )
            )
        return actions

    def add_hypotension_treatment(
        self,
        actions: list[Action],
        t_window: tuple[float, float],
        weight_kg: float,
        probability: float = 0.6,
    ) -> list[Action]:
        """Add a vasoactive response to hypotension during maintenance."""
        if self.rng.random() < probability:
            t = self.rng.uniform(t_window[0], t_window[1])
            choice = self.rng.choice(["ephedrine", "phenylephrine", "noradrenaline"])
            if choice == "ephedrine":
                actions.append(
                    Action(
                        t_s=t,
                        action_type=ActionType.VASOACTIVE_BOLUS,
                        drug="ephedrine",
                        value=5.0 + self.rng.exponential(5.0),
                        unit="mg",
                        route="IV",
                    )
                )
            elif choice == "phenylephrine":
                actions.append(
                    Action(
                        t_s=t,
                        action_type=ActionType.VASOACTIVE_BOLUS,
                        drug="phenylephrine",
                        value=50.0 + self.rng.exponential(50.0),
                        unit="mcg",
                        route="IV",
                    )
                )
            else:
                actions.append(
                    Action(
                        t_s=t,
                        action_type=ActionType.VASOACTIVE_INFUSION,
                        drug="noradrenaline",
                        value=0.05 + self.rng.exponential(0.05),
                        unit="mcg/kg/min",
                        route="IV",
                    )
                )
        return actions

    def randomize_baselines(
        self,
        map_baseline: float,
        hr_baseline: float,
        cv: float = 0.08,
    ) -> tuple[float, float]:
        """Return patient-specific baseline MAP and HR with variability."""
        map_new = self.rng.normal(map_baseline, map_baseline * cv)
        hr_new = self.rng.normal(hr_baseline, hr_baseline * cv)
        return float(map_new), float(hr_new)
