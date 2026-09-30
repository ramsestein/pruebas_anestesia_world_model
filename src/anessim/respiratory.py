"""respiratory.py — Causal respiratory physiology model (v3).

Flechas causales:
  1. FiO2 → SpO2 (gas alveolar → PaO2 → curva Hb)
  2. MV → EtCO2 (VA = (TV-Vd)*RR → PaCO2 = VCO2*k/VA)
  3. PEEP → oxigenación (reclutamiento → reduce shunt → sube SpO2)
  4. Compliance → PIP (PIP ≈ TV/compliance + PEEP)

Flechas cruzadas:
  5. PEEP → MAP (↑PEEP → ↓precarga → ↓MAP)
  6. Desaturación → hemodinámica (SpO2<85% → taquicardia inicial → bradicardia+hipotensión)

IIV: shunt_fraction, VCO2, Vd, compliance varían por paciente (lognormal CV~20-30%).
"""

from __future__ import annotations

import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────
PATM = 760.0       # mmHg, atmospheric pressure at sea level
PH2O = 47.0        # mmHg, water vapor pressure at 37°C
RQ = 0.8           # respiratory quotient
K_CO2 = 863.0      # PaCO2 = VCO2(mL/min)·K / VA(mL/min) — K in mmHg·mL/mL


class RespiratoryModel:
    """Per-patient respiratory physiology with IIV."""

    def __init__(
        self,
        rng: np.random.Generator,
        shunt_fraction: float | None = None,
        vco2: float | None = None,       # mL/min CO2 production
        vd: float | None = None,          # mL dead space
        compliance: float | None = None,  # mL/cmH2O
        resp_load: float | None = None,   # carga respiratoria compartida
        age: float = 50.0,
        weight_kg: float = 70.0,
    ) -> None:
        self.rng = rng

        # ── IIV: population means with lognormal variation ────────────────
        self.shunt_frac = shunt_fraction or float(np.clip(
            rng.lognormal(np.log(0.05), 0.40), 0.01, 0.30))
        # CO2 production scales with metabolic rate (allometric exponent 0.75).
        # 165 mL/min reference for a 70 kg anesthetised adult (reduced metabolic
        # rate under anaesthesia; calibrated so real EtCO2 p50 ≈ 34 mmHg).
        # v7 (C4): CV 0.25 -> 0.12 (VCO2 ajustada por peso; literatura 10-15 %).
        # La variabilidad residual entre casos que antes absorbía el CV 0.25 se
        # traslada al factor de carga respiratoria (resp_load), que acopla RR y
        # EtCO2 POSITIVAMENTE (ver compute_etco2 y sample_rr).
        vco2_ref = vco2_reference(weight_kg)
        self.vco2 = vco2 or float(np.clip(
            vco2_ref * rng.lognormal(0.0, 0.12), 70.0, 450.0))  # mL/min
        # Anatomical dead space ≈ 2.2 mL/kg (scales with size, not fixed).
        vd_ref = 2.2 * weight_kg
        self.vd = vd or float(np.clip(
            vd_ref * rng.lognormal(0.0, 0.08), 60.0, 400.0))  # mL
        # v7 (C4): CARGA RESPIRATORIA COMPARTIDA entre RR y EtCO2 (V/Q mismatch,
        # espacio muerto alveolar, impedancia del sistema respiratorio). El clínico
        # sube la RR para compensarla (sample_rr) pero deja un EtCO2 residual alto
        # (compute_etco2): es el mecanismo del corr(RR, EtCO2) = +0.336 real.
        self.resp_load = resp_load or float(rng.lognormal(0.0, 0.25))
        # v7 (C1): compliance ESTÁTICA del sistema respiratorio. En anestesia
        # general con relajación muscular, paciente en decúbito supino, la Crs
        # de un pulmón sano es ~50-70 mL/cmH2O (la FRC cae ~20 %, la compliance
        # efectiva queda ~50). Valores ~25-35 son de obesidad/neumoperitoneo,
        # capturados por la cola izquierda de la lognormal. Antes 27 de media
        # era un artefacto de no modelar el componente resistivo del PIP.
        self.compliance = compliance or float(np.clip(
            rng.lognormal(np.log(48.0), 0.25), 25.0, 100.0))  # mL/cmH2O
        # v7 (C1): RESISTENCIA de vía aérea (cmH2O·s/L). En pacientes intubados
        # la resistencia total es ~5-10 cmH2O·s/L (el tubo endotraqueal aporta
        # ~5-8); con secreciones/broncoespasmo sube a 10-25+. La cola derecha
        # (gamma) + eventos raros de broncoespasmo reproducen el PIP alto.
        self.resistance = float(
            4.0 + rng.gamma(1.5, 3.0)
            + (rng.uniform(10.0, 22.0) if rng.random() < 0.015 else 0.0))
        # v7 (C1): flujo inspiratorio PICO (L/s) del ventilador (30-54 L/min).
        self.peak_flow = float(rng.uniform(0.5, 0.9))
        self.weight_kg = weight_kg

    def compute_pao2(
        self,
        fio2: float,
        peep: float = 5.0,
        paco2: float = 40.0,
    ) -> float:
        """Compute alveolar PO2 (mmHg).

        PAO2 = FiO2·(Patm−PH2O) − PaCO2/RQ
        """
        return fio2 * (PATM - PH2O) - paco2 / RQ

    def compute_pao2_to_spo2(
        self,
        pao2: float,
        peep: float = 5.0,
    ) -> tuple[float, float, float]:
        """Convert PaO2 to SpO2 through shunt + PEEP recruitment.

        Returns:
            spo2: pulse oximetry saturation (%).
            pao2_effective: effective PaO2 after shunt penalty.
            shunt_effective: shunt fraction after PEEP recruitment.
        """
        # PEEP recruitment: reduces shunt with diminishing returns
        # Max ~40% shunt reduction at PEEP=15, half at PEEP=8
        peep_benefit = 0.40 * (1.0 - np.exp(-peep / 8.0))
        shunt_eff = self.shunt_frac * (1.0 - peep_benefit)
        shunt_eff = max(shunt_eff, 0.005)

        # Shunt penalty: shunt_eff fraction of blood bypasses oxygenation
        # PaO2_effective = PaO2 * (1 - shunt_eff) + PvO2 * shunt_eff
        # Simplified: effective PaO2 ≈ PaO2 * (1 - shunt_eff*0.8)
        pao2_eff = pao2 * (1.0 - shunt_eff * 0.8)
        pao2_eff = max(pao2_eff, 20.0)

        # Hill equation for hemoglobin saturation
        # P50 = 26.6 mmHg, n = 2.7 (adult Hb)
        p50 = 26.6
        n_hill = 2.7
        spo2 = 100.0 * (pao2_eff ** n_hill) / (pao2_eff ** n_hill + p50 ** n_hill)

        return float(np.clip(spo2, 50.0, 100.0)), float(pao2_eff), float(shunt_eff)

    def compute_etco2(
        self,
        tv_ml: float,
        rr: float,
    ) -> tuple[float, float]:
        """Compute EtCO2 from ventilation parameters.

        VA = (TV - Vd) * RR  (alveolar ventilation in mL/min)
        PaCO2 = VCO2 * K_CO2 / VA
        EtCO2 ≈ PaCO2 * 0.95  (end-tidal slightly lower than arterial)

        Returns:
            etco2: mmHg
            paco2: mmHg
        """
        va = max((tv_ml - self.vd) * rr, 500.0)  # minimum alveolar ventilation
        paco2 = self.vco2 * K_CO2 / va
        # v7 (C4): el factor de carga respiratoria eleva el PaCO2 para una misma
        # VA (espacio muerto alveolar / V/Q), dejando un EtCO2 residual alto en
        # los pacientes que el clínico intenta compensar subiendo la RR.
        paco2 = float(np.clip(paco2 * (self.resp_load ** 0.8), 0.0, 90.0))
        # v6 (M7): no recortar las colas reales de EtCO2 [10, 74].
        etco2 = paco2 * 0.95
        return etco2, paco2

    def compute_pip(
        self,
        tv_ml: float,
        peep: float = 5.0,
    ) -> float:
        """Compute peak inspiratory pressure (componente resistivo incluido).

        PIP = TV / Crs + R·V̇_peak + PEEP

        Crs: compliance estática; R: resistencia de vía aérea; V̇_peak: flujo
        inspiratorio pico. El término resistivo es lo que separa la presión
        PICO de la meseta (PPLAT = TV/Crs + PEEP).
        """
        return tv_ml / self.compliance + self.resistance * self.peak_flow + peep

    def compute_peep_map_effect(
        self,
        peep: float,
        hypovolemic: bool = False,
    ) -> float:
        """Compute PEEP → MAP depression (mmHg).

        Above PEEP ~8-10, intrathoracic pressure rises → reduced venous return
        → reduced preload → reduced MAP. Worse in hypovolemic patients.

        Returns:
            map_depression: negative value to add to MAP.
        """
        if peep <= 8.0:
            return 0.0
        # Sigmoid: starts at PEEP=8, half-max at PEEP=14
        excess = max(0.0, peep - 8.0)
        effect = -12.0 * (excess / (excess + 6.0))
        if hypovolemic:
            effect *= 1.5
        return float(effect)

    def compute_desaturation_hemodynamics(
        self,
        spo2: float,
        spo2_prev: float,
        dt_s: float,
        hr_current: float,
        map_current: float,
    ) -> tuple[float, float]:
        """Compute hemodynamic response to sustained desaturation.

        Biphasic response:
          SpO2 85-90%: mild sympathetic (HR +5-15, MAP +3-8)
          SpO2 <85% sustained: progressive bradycardia + hypotension

        Returns:
            (hr_delta, map_delta): values to ADD to current HR/MAP.
        """
        if spo2 > 90.0:
            return 0.0, 0.0

        if spo2 >= 85.0:
            # Mild hypoxia → sympathetic activation
            hr_delta = 8.0 * (90.0 - spo2) / 5.0
            map_delta = 4.0 * (90.0 - spo2) / 5.0
            return hr_delta, map_delta

        # Severe hypoxia → biphasic: initial sympathetic then depression
        severity = (85.0 - spo2) / 35.0  # normalized 0..1 for spo2 85→50
        severity = min(severity, 1.0)

        # If transitioning down from >90%, give initial sympathetic spike
        if spo2_prev > 90.0:
            hr_delta = 20.0 * severity  # transient tachycardia
            map_delta = 10.0 * severity
        else:
            # Sustained: bradycardia + hypotension
            hr_delta = -25.0 * severity
            map_delta = -20.0 * severity

        return float(hr_delta), float(map_delta)


# ── Exogenous ventilation perturbation ────────────────────────────────────────

# v6 (M2): distribución de PEEP recalibrada contra la cohorte real (P1b):
# masa puntual en 0 (46 % sin PEEP), nivel dominante 5, cola rara 10-25.
_PEEP_ZERO_PROB = 0.46
_PEEP_LEVELS = [5.0, 4.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 10.0, 9.0]
_PEEP_WEIGHTS = [564825, 133072, 47438, 18807, 8183, 7650, 4404, 4399, 2579, 1270]
_PEEP_WEIGHTS_NORM = np.asarray(_PEEP_WEIGHTS, dtype=float) / sum(_PEEP_WEIGHTS)
_PEEP_CUMSUM = np.cumsum(_PEEP_WEIGHTS_NORM)


def sample_peep(rng: np.random.Generator) -> float:
    """Nivel de PEEP (cmH2O, entero) de un tramo: 0 con masa 0.46, si no un
    nivel 1-25 con dominancia de 4-6 y cola larga poco frecuente."""
    if rng.random() < _PEEP_ZERO_PROB:
        return 0.0
    if rng.random() < 0.001:  # cola larga 10-25 (real: 10-25 muy raro)
        return float(rng.integers(10, 26))
    idx = int(np.searchsorted(_PEEP_CUMSUM, rng.random()))
    return float(_PEEP_LEVELS[min(idx, len(_PEEP_LEVELS) - 1)])


def vco2_reference(weight_kg: float) -> float:
    """Producción de CO2 de referencia (mL/min) para un adulto anestesiado.
    Escala alométrica 0.75 (calibrada a EtCO2 p50 real ≈ 34 mmHg)."""
    return 165.0 * (weight_kg / 70.0) ** 0.75


def sample_rr(rng: np.random.Generator, vco2: float | None = None,
              vco2_ref: float | None = None,
              weight_kg: float | None = None,
              resp_load: float | None = None) -> float:
    """Frecuencia respiratoria (rpm, entera). El clínico compensa la
    producción de CO2 subiendo la RR (acoplamiento VCO2^0.7), la carga
    respiratoria (resp_load^0.6: V/Q, espacio muerto) y el tamaño corporal
    bajándola (RR ∝ w^-0.25, porque un TV mayor ya ventila más), de modo que
    la RR varía entre casos (sd ~2.7) sin disparar el EtCO2 y con
    corr(RR, EtCO2) ≈ +0.336. v7 (C4): se elimina la rama de extremos raros
    (2-3 / 30-40) porque rompía la corr(RR, EtCO2) al forzar la física
    VA=(TV-Vd)·RR en pacientes taquipneicos (EtCO2 ~10, irreal); el real tiene
    p99 ≈ 23 y la variación lognormal ya lo reproduce."""
    metabolic = 1.0
    if vco2 is not None and vco2_ref is not None and vco2_ref > 0:
        metabolic = float((vco2 / vco2_ref) ** 0.7)
    if resp_load is not None:
        metabolic *= float(resp_load ** 0.6)
    if weight_kg is not None:
        metabolic *= float((weight_kg / 70.0) ** -0.25)
    return float(np.clip(int(round(14.0 * metabolic * rng.normal(1.0, 0.06))),
                         4, 30))


def generate_ventilation_profile(
    t: np.ndarray,
    timeline,
    rng: np.random.Generator,
    weight_kg: float = 70.0,
    fio2_baseline: float = 0.50,
    suppress_fio2_steps: bool = False,
    vco2: float | None = None,
    resp_load: float | None = None,
) -> dict:
    """Generate time-varying ventilation settings with exogenous perturbation.

    Args:
        fio2_baseline: baseline FiO2 before exogenous steps. Lower values
            (e.g. 0.35) leave headroom for PEEP/FiO2 interventions to show
            a measurable SpO2 effect (avoids Hill-curve saturation).
        suppress_fio2_steps: if True, skip exogenous FiO2 steps so the
            baseline stays low throughout (needed for ventilation CF pairs).

    Returns dict with arrays of same length as t:
      fio2, peep, rr_set, tv_set
    """
    n = len(t)
    maintenance = next(p for p in timeline.phases if p.name == "maintenance")
    emergence = next(p for p in timeline.phases if p.name == "emergence")

    # Baseline settings (v6: PEEP y RR muestreados por caso, no fijos).
    fio2 = np.full(n, fio2_baseline, dtype=np.float32)
    peep = np.full(n, sample_peep(rng), dtype=np.float32)
    rr_base = sample_rr(rng, vco2, vco2_reference(weight_kg), weight_kg, resp_load)
    rr = np.full(n, rr_base, dtype=np.float32)
    tv_ml_per_kg = np.full(n, 6.3, dtype=np.float32)  # mL/kg (real ~6.3)

    maint_mask = (t >= maintenance.start_s) & (t < emergence.start_s)
    if not maint_mask.any():
        return {"fio2": fio2, "peep": peep, "rr": rr, "tv": tv_ml_per_kg * weight_kg}

    # ── Exogenous FiO2 steps ─────────────────────────────────────────────
    if not suppress_fio2_steps:
        n_fio2 = rng.integers(3, 8)
        for _ in range(n_fio2):
            d = rng.uniform(300.0, 1200.0)
            s = rng.uniform(maintenance.start_s, emergence.start_s - d)
            val = rng.uniform(0.30, 1.0)
            fio2[(t >= s) & (t <= s + d)] = val

    # ── Exogenous PEEP steps (raros: el PEEP real cambia poco) ──────────
    n_peep = int(rng.integers(0, 3))
    for _ in range(n_peep):
        d = rng.uniform(300.0, 1200.0)
        s = rng.uniform(maintenance.start_s, emergence.start_s - d)
        val = float(sample_peep(rng))
        peep[(t >= s) & (t <= s + d)] = val

    # ── Exogenous RR steps (pequeñas desviaciones alrededor del basal: la RR
    # real cambia poco dentro del caso, frac_nochg ~0.98) ─────────────────
    n_rr = int(rng.integers(1, 4))
    for _ in range(n_rr):
        d = rng.uniform(300.0, 1200.0)
        s = rng.uniform(maintenance.start_s, emergence.start_s - d)
        val = float(np.clip(int(round(rr_base + rng.normal(0.0, 1.2))), 4, 30))
        rr[(t >= s) & (t <= s + d)] = val

    # ── Exogenous TV steps ───────────────────────────────────────────────
    n_tv = rng.integers(2, 5)
    for _ in range(n_tv):
        d = rng.uniform(300.0, 1200.0)
        s = rng.uniform(maintenance.start_s, emergence.start_s - d)
        val = rng.uniform(6.0, 6.5)  # mL/kg
        tv_ml_per_kg[(t >= s) & (t <= s + d)] = val

    return {
        "fio2": fio2,
        "peep": peep,
        "rr": rr,
        "tv": tv_ml_per_kg * weight_kg,
    }
