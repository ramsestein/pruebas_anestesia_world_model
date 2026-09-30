"""Propofol PK models: Marsh, Schnider, Eleveld.

The three major published propofol PK models coexist in clinical practice
with no consensus winner.  For causal identifiability the generator
randomises the model per case, recording the choice as oracle metadata
(never exposed to the learning model).
"""

from __future__ import annotations

import numpy as np

from anessim.pk.base import ThreeCompartmentModel


class PropofolMarsh(ThreeCompartmentModel):
    """Marsh propofol model (Dyck & Shafer, 1993) with ke0 0.26/min.

    Weight-proportional volumes, age-independent.
    """

    def __init__(self, weight_kg: float, age_y: float = 50) -> None:
        v1 = 0.228 * weight_kg
        v2 = 0.463 * weight_kg
        v3 = 2.893 * weight_kg
        super().__init__(
            drug="propofol",
            v1=v1, v2=v2, v3=v3,
            k10=0.119, k12=0.112, k13=0.0419,
            k21=0.055, k31=0.0033,
            ke0=0.26,
        )


class PropofolSchnider(ThreeCompartmentModel):
    """Schnider propofol model (Schnider et al., 1998) with ke0 0.456/min.

    Age, weight, height, and LBM covariates.  Faster ke0 than Marsh
    (peak effect ~1.6 min vs ~2.5 min).
    """

    def __init__(self, weight_kg: float, age_y: float, height_cm: float, sex: str) -> None:
        lbm = _schnider_lbm(sex, weight_kg, height_cm)
        v1 = 4.27  # L — fixed central volume
        v2 = 18.9 - 0.391 * (age_y - 53)  # L
        v3 = 238.0  # L — fixed slow peripheral
        cl1 = 1.89 + 0.0456 * (weight_kg - 77) - 0.0681 * (lbm - 59) + 0.0264 * (height_cm - 177)
        cl2 = 1.29 - 0.024 * (age_y - 53)
        cl3 = 0.836
        k10 = cl1 / v1
        k12 = cl2 / v1
        k13 = cl3 / v1
        k21 = cl2 / v2
        k31 = cl3 / v3
        super().__init__(
            drug="propofol",
            v1=v1, v2=v2, v3=v3,
            k10=k10, k12=k12, k13=k13,
            k21=k21, k31=k31,
            ke0=0.456,
        )


class PropofolEleveld(ThreeCompartmentModel):
    """Eleveld propofol model (Eleveld et al., 2018).

    The most comprehensive covariate model: age, weight, height, sex, and
    additional scaling factors (PMA, central blood volume).  Uses allometric
    scaling on clearances and volumes.
    """

    def __init__(self, weight_kg: float, age_y: float, height_cm: float, sex: str) -> None:
        # Reference individual: 70 kg, 35 yr, 170 cm, male
        wt_ref = 70.0
        age_ref = 35.0

        # Fat-free mass (Al-Sallami formula — paediatric & adult)
        bmi = weight_kg / ((height_cm / 100.0) ** 2)
        if sex == "M":
            ffm = (0.88 * weight_kg + 12.8 * (height_cm / 100.0) ** 2) / (1.0 + 0.0273 * bmi)
        else:
            ffm = (1.11 * weight_kg + 7.52 * (height_cm / 100.0) ** 2) / (1.0 + 0.0236 * bmi)

        # Allometric exponents for CL and V
        theta_cl = 0.75
        theta_v = 1.0

        # Age sigmoid for clearance
        age50 = max(age_y, 18.0)
        age_factor_cl = age50 ** (-0.446) / (age_ref ** (-0.446)) if age50 > 0 else 1.0

        # FFM scaling
        ffm_factor_cl = (ffm / (0.88 * wt_ref + 12.8 * (1.70) ** 2 / (1.0 + 0.0273 * (wt_ref / (1.70 ** 2))))) ** theta_cl
        ffm_factor_v = (ffm / (0.88 * wt_ref + 12.8 * (1.70) ** 2 / (1.0 + 0.0273 * (wt_ref / (1.70 ** 2))))) ** theta_v

        # Published population parameters for the reference individual
        v1_ref = 6.28   # L
        v2_ref = 25.5   # L
        v3_ref = 273.0  # L
        cl_ref = 1.79   # L/min
        q2_ref = 1.75   # L/min
        q3_ref = 1.11   # L/min

        v1 = v1_ref * ffm_factor_v
        v2 = v2_ref * ffm_factor_v
        v3 = v3_ref * ffm_factor_v
        cl = cl_ref * ffm_factor_cl * age_factor_cl
        q2 = q2_ref * ffm_factor_cl
        q3 = q3_ref * ffm_factor_cl

        k10 = cl / v1
        k12 = q2 / v1
        k13 = q3 / v1
        k21 = q2 / v2
        k31 = q3 / v3

        # ke0 age-adjusted (slower equilibration in elderly)
        ke0 = 0.146 * (age_y / age_ref) ** (-0.25)

        super().__init__(
            drug="propofol",
            v1=v1, v2=v2, v3=v3,
            k10=k10, k12=k12, k13=k13,
            k21=k21, k31=k31,
            ke0=ke0,
        )


def _schnider_lbm(sex: str, weight_kg: float, height_cm: float) -> float:
    """Lean body mass per Schnider (James formula, 1976)."""
    if sex == "M":
        return 1.1 * weight_kg - 128.0 * (weight_kg / height_cm) ** 2
    return 1.07 * weight_kg - 148.0 * (weight_kg / height_cm) ** 2


# ── Model registry for per-case randomisation ────────────────────────────
PROPOFOL_MODELS = {
    "marsh": PropofolMarsh,
    "schnider": PropofolSchnider,
    "eleveld": PropofolEleveld,
}


def randomise_propofol_model(
    weight_kg: float,
    age_y: float,
    height_cm: float,
    sex: str,
    rng: np.random.Generator,
) -> tuple[ThreeCompartmentModel, str]:
    """Randomly select a propofol PK model for one case.

    Returns:
        (model_instance, model_name)
    """
    name = rng.choice(["marsh", "schnider", "eleveld"])
    cls = PROPOFOL_MODELS[name]
    kwargs = dict(weight_kg=weight_kg, age_y=age_y)
    if name in ("schnider", "eleveld"):
        kwargs["height_cm"] = height_cm
        kwargs["sex"] = sex
    return cls(**kwargs), name

