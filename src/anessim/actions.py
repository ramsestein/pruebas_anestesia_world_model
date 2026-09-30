"""Clinical actions (drug boluses, infusions, ventilator changes, stimuli)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class ActionType(Enum):
    BOLUS = auto()
    INFUSION_START = auto()
    INFUSION_STOP = auto()
    INFUSION_CHANGE = auto()
    VENTILATOR_SETTING = auto()
    STIMULUS = auto()
    FLUID_BOLUS = auto()
    VASOACTIVE_BOLUS = auto()
    VASOACTIVE_INFUSION = auto()


@dataclass(frozen=True)
class Action:
    t_s: float
    action_type: ActionType
    drug: str | None = None
    value: float | None = None
    unit: str | None = None
    route: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def summary(self) -> str:
        parts = [f"t={self.t_s:.1f}s", self.action_type.name]
        if self.drug:
            parts.append(self.drug)
        if self.value is not None:
            parts.append(f"{self.value:.3g}{self.unit or ''}")
        return " ".join(parts)
