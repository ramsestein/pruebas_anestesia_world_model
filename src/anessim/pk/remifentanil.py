"""Remifentanil PK models."""

from __future__ import annotations

from anessim.pk.base import ThreeCompartmentModel


def _lean_body_weight_minto(sex: str, weight_kg: float, height_cm: float) -> float:
    """Minto remifentanil LBM formula (Jammahasatian)."""
    bmi = weight_kg / ((height_cm / 100.0) ** 2)
    if sex == "M":
        return 9270 * weight_kg / (6680 + 216 * bmi)
    return 9270 * weight_kg / (8780 + 244 * bmi)


class RemifentanilMinto(ThreeCompartmentModel):
    """Minto remifentanil model (Minto et al., 1997)."""

    def __init__(self, weight_kg: float, height_cm: float, age_y: float, sex: str) -> None:
        lbm = _lean_body_weight_minto(sex, weight_kg, height_cm)
        v1 = 5.1 - 0.0201 * (age_y - 40) + 0.0722 * (lbm - 50)
        v2 = 9.82 - 0.0811 * (age_y - 40) + 0.108 * (lbm - 50)
        v3 = 5.42
        k10 = 0.595 - 0.007 * (age_y - 40)
        k12 = 0.476
        k13 = 0.212
        k21 = 0.403 - 0.0026 * (age_y - 40)
        k31 = 0.029
        ke0 = 0.6
        super().__init__(
            drug="remifentanil",
            v1=v1,
            v2=v2,
            v3=v3,
            k10=k10,
            k12=k12,
            k13=k13,
            k21=k21,
            k31=k31,
            ke0=ke0,
        )
