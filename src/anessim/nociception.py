"""Nociception driver — surgical stimulus with proper causal pathways.

Architecture:
  Aδ/C fibres → dorsal horn → spinothalamic tract → sympathetic activation.
  Propofol depresses thalamocortical (BIS) but NOT the spinal sympathetic arc.
  Remifentanil blocks at the dorsal horn (µ receptors) → attenuates the
  stimulus AT ORIGIN before it reaches MAP/HR.

Key design constraint:
  - MAP/HR: STRONG response to stimulus, strongly attenuated by remifentanil.
  - BIS:   WEAK, noisy, inconsistent response — different pathway.

Temporal profile:
  - Skin incision: pure transient, sharp peak 15-30s, decay τ≈60-90s.
  - Dissection/maintenance: sustained baseline (0.4-0.7 by optype) + phasic
    peaks (traction, cutting, coagulation) at random intervals.
  - Closure: lower baseline (0.3-0.5) with smaller peaks.

The module also models the "valle nociceptivo": the gap between the induction
remifentanil bolus decay and the infusion reaching steady-state, where the
incision lands — the moment of maximum vulnerability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from anessim.timeline import Timeline


# ═══════════════════════════════════════════════════════════════════════════
# Op-type → nociceptive intensity mapping (VitalDB opname → canonical class)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SurgeryClass:
    """Nociceptive intensity profile for one surgery class."""
    tonic: float   # sustained baseline during maintenance [0, 1]
    peak: float    # phasic peak amplitude [0, 1]
    label: str     # human-readable label


# Canonical surgery classes ordered by nociceptive intensity
SURGERY_CLASSES: dict[str, SurgeryClass] = {
    "laparotomia_mayor":     SurgeryClass(0.70, 0.90, "Laparotomía mayor (hepática, Whipple)"),
    "toracica":              SurgeryClass(0.65, 0.85, "Torácica (toracotomía)"),
    "trasplante_hepatico":   SurgeryClass(0.60, 0.85, "Trasplante hepático"),
    "ortopedia_mayor":       SurgeryClass(0.55, 0.75, "Ortopedia mayor (prótesis)"),
    "laparoscopia_mayor":    SurgeryClass(0.45, 0.65, "Laparoscopia mayor (colectomía, nefrectomía)"),
    "ginecologia_lap":       SurgeryClass(0.35, 0.55, "Ginecología laparoscópica"),
    "urologia_endoscopica":  SurgeryClass(0.30, 0.50, "Urología endoscópica (RTU)"),
    "cirugia_menor":         SurgeryClass(0.30, 0.60, "Cirugía menor (hernia, piel)"),
    "neurocirugia":          SurgeryClass(0.20, 0.35, "Neurocirugía (craneotomía)"),
    "default":               SurgeryClass(0.45, 0.65, "Default / unclassified"),
}

# Mapping from VitalDB optypes (as they appear in clinical data) to canonical classes.
# Based on empirical analysis of VitalDB opname/optype distribution.
_OPTYPE_MAP: dict[str, str] = {
    # Laparotomía mayor
    "HPB":  "laparotomia_mayor",       # Hepato-pancreato-biliary
    "GS":   "laparotomia_mayor",       # General surgery (many are major open)
    "CS":   "laparotomia_mayor",       # Colorectal (open)
    "GI":   "laparotomia_mayor",       # Upper GI (open)
    # Torácica
    "CV":   "toracica",                # Cardiovascular
    "TS":   "toracica",                # Thoracic surgery
    # Trasplante
    "TX":   "trasplante_hepatico",     # Transplant
    # Ortopedia mayor
    "OS":   "ortopedia_mayor",         # Orthopedic (many are major)
    # Laparoscopia mayor
    "UG":   "laparoscopia_mayor",      # Urology (many laparoscopic)
    # Ginecología
    "GY":   "ginecologia_lap",         # Gynecology
    # Urología endoscópica
    "UR":   "urologia_endoscopica",    # Urology
    # Neurocirugía
    "NS":   "neurocirugia",            # Neurosurgery
    "NE":   "neurocirugia",            # Neuro (old code)
    # Cirugía menor
    "PS":   "cirugia_menor",           # Plastic surgery
    "EN":   "cirugia_menor",           # ENT
    # Thoracic (some overlap)
    "TH":   "toracica",
    # Other → default
    "OT":   "default",
    "ETC":  "default",
}


def resolve_surgery_class(optype: str, opname: str = "") -> SurgeryClass:
    """Map VitalDB optype (+ optional opname) to a canonical surgery class.

    Falls back to heuristic keyword matching on opname when optype is ambiguous
    or missing.
    """
    optype_upper = optype.strip().upper() if optype else ""
    cls_key = _OPTYPE_MAP.get(optype_upper, "default")

    # Heuristic override from opname keywords
    if cls_key == "default" and opname:
        name_lower = opname.lower()
        if any(kw in name_lower for kw in ("whipple", "hepatectom", "liver resection", "pancreaticoduodenectom")):
            cls_key = "laparotomia_mayor"
        elif any(kw in name_lower for kw in ("thoracotom", "lobectom", "pneumonectom", "oesophagectom")):
            cls_key = "toracica"
        elif any(kw in name_lower for kw in ("transplant", "lt ", "olt", "ddlt", "ldlt")):
            cls_key = "trasplante_hepatico"
        elif any(kw in name_lower for kw in ("arthroplast", "hip replacement", "knee replacement", "spine fusion", "spinal fusion", "scoliosis")):
            cls_key = "ortopedia_mayor"
        elif any(kw in name_lower for kw in ("laparoscopic", "robot", "colectom", "nephrectom", "adrenalectom", "sleeve gastrectom", "gastric bypass")):
            cls_key = "laparoscopia_mayor"
        elif any(kw in name_lower for kw in ("craniotom", "craniectom", "brain tumor", "meningioma", "glioma")):
            cls_key = "neurocirugia"
        elif any(kw in name_lower for kw in ("turp", "turbt", "tur ", "cystoscop", "ureteroscop")):
            cls_key = "urologia_endoscopica"
        elif any(kw in name_lower for kw in ("hernia", "appendectom", "cholecystectom", "thyroidectom", "mastectom")):
            cls_key = "cirugia_menor"

    return SURGERY_CLASSES.get(cls_key, SURGERY_CLASSES["default"])


# ═══════════════════════════════════════════════════════════════════════════
# Nociceptive stimulus generator
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NociceptionConfig:
    """Configuration for nociceptive stimulus generation.

    All probabilities and ranges are clinically grounded defaults;
    override for sensitivity studies.
    """
    # ── Valle nociceptivo ────────────────────────────────────────────────
    remi_bolus_prob: float = 0.95          # prob of remi bolus at induction
    remi_bolus_dose_mcg: float = 80.0      # typical bolus (scaled by LBM)
    remi_bolus_peak_ce: float = 4.5        # target Ce peak ng/mL
    remi_bolus_decay_tau_s: float = 75.0   # decay time constant (clinical)
    incisión_delay_after_induction_s: float = 180.0  # typical incision timing
    remi_refuerzo_prob: float = 0.40       # prob of pre-incision reinforcement bolus
    remi_refuerzo_frac: float = 0.5        # reinforcement bolus as fraction of induction

    # ── Phasic event parameters ───────────────────────────────────────────
    n_phasic_events_mean: float = 4.0      # Poisson mean for maintenance events
    phasic_peak_range: tuple[float, float] = (0.3, 0.75)  # relative to tonic
    phasic_tau_range_s: tuple[float, float] = (40.0, 120.0)  # decay τ per event
    phasic_interval_mean_s: float = 600.0  # mean time between phasic events
    phasic_noise_cv: float = 0.30          # CV of event intensity

    # ── Closure ───────────────────────────────────────────────────────────
    closure_tonic_frac: float = 0.6         # closure baseline as fraction of maintenance
    closure_peak_frac: float = 0.5          # closure peaks as fraction of maintenance

    # ── IIV ───────────────────────────────────────────────────────────────
    sensitivity_cv: float = 0.25            # inter-individual stimulus sensitivity CV
    age_modulation: float = 0.005           # per-year sensitivity increase
    asa_modulation: float = 0.08            # per-ASA-point sensitivity increase


def _ar1_process(
    n: int,
    dt: float,
    tau: float,
    mean: float,
    sigma: float,
    rng: np.random.Generator,
    start_value: float | None = None,
) -> np.ndarray:
    """Generate an AR(1) process (Ornstein-Uhlenbeck) with given parameters.

    Args:
        n: number of time steps.
        dt: time step in seconds.
        tau: mean-reversion time constant in seconds.
        mean: long-run mean.
        sigma: steady-state standard deviation.
        rng: numpy random generator.
        start_value: initial value (default: draw from stationary distribution).

    Returns:
        (n,) array.
    """
    alpha = float(np.exp(-dt / tau))
    sigma_innov = sigma * float(np.sqrt(1.0 - alpha ** 2))
    if start_value is None:
        start_value = rng.normal(mean, sigma)
    x = np.empty(n)
    x[0] = start_value
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + (1.0 - alpha) * mean + rng.normal(0.0, sigma_innov)
    return x


def compute_remi_attenuation(
    ce_remi: float | np.ndarray,
    c50_map: float = 2.5,
    c50_hr: float = 3.0,
    gamma: float = 2.0,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Sigmoid attenuation of nociceptive response by remifentanil.

    Args:
        ce_remi: remifentanil effect-site concentration (ng/mL).
        c50_map: C50 for MAP attenuation (ng/mL). Lower → easier to attenuate.
        c50_hr:  C50 for HR attenuation (ng/mL). Slightly higher → HR harder
                 to attenuate than MAP at equal Ce.
        gamma:   Hill coefficient (steepness).

    Returns:
        (attenuation_map, attenuation_hr): fraction [0,1] of the stimulus
        that SURVIVES attenuation. 0 = complete block, 1 = no block.
    """
    r = np.asarray(ce_remi, dtype=float)
    att_map = 1.0 / (1.0 + (r / c50_map) ** gamma)
    att_hr = 1.0 / (1.0 + (r / c50_hr) ** gamma)
    return att_map, att_hr


class NociceptionDriver:
    """Generate the full nociceptive stimulus profile for one case.

    Produces:
      - surgical_stimulus[t]: raw stimulus intensity in [0, 1].
      - remi_attenuation_map[t], remi_attenuation_hr[t]: survival fractions.
      - The effective nociceptive drive to MAP = stimulus * att_map.
        The effective nociceptive drive to HR  = stimulus * att_hr.
      - The effective nociceptive drive to BIS = stimulus * 0.15 + noise
        (weak, inconsistent arousal — different pathway).

    The caller computes:
        map_nociceptive = surgical_stimulus * remi_attenuation_map * MAP_GAIN
        hr_nociceptive  = surgical_stimulus * remi_attenuation_hr * HR_GAIN
        bis_arousal     = surgical_stimulus * BIS_GAIN + noise
    """

    def __init__(
        self,
        rng: np.random.Generator,
        config: NociceptionConfig | None = None,
    ) -> None:
        self.rng = rng
        self.cfg = config or NociceptionConfig()

    def generate(
        self,
        t: np.ndarray,
        timeline: Timeline,
        op_type: str,
        op_name: str,
        ce_remi: np.ndarray,
        patient_age: float,
        patient_asa: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Generate the full nociceptive stimulus profile.

        Args:
            t: time grid in seconds.
            timeline: case timeline with phase boundaries.
            op_type: VitalDB optype (e.g. "GS", "GY").
            op_name: VitalDB opname (for keyword heuristics).
            ce_remi: remifentanil Ce at each time point (ng/mL).
            patient_age: patient age in years.
            patient_asa: ASA physical status (1-4).

        Returns:
            (stimulus, att_map, att_hr, metadata)
            stimulus: (T,) raw stimulus [0,1].
            att_map:  (T,) MAP attenuation survival fraction.
            att_hr:   (T,) HR attenuation survival fraction.
            metadata: dict with audit info (surgery class, IIV, valle details).
        """
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 0.5

        # ── Resolve surgery class ─────────────────────────────────────────
        sclass = resolve_surgery_class(op_type, op_name)
        tonic = sclass.tonic
        peak = sclass.peak

        # ── Inter-individual variability in stimulus sensitivity ──────────
        age_shift = max(0.0, patient_age - 50.0)
        asa_shift = max(0, (patient_asa or 2) - 2)
        iiv_factor = float(
            self.rng.lognormal(0.0, self.cfg.sensitivity_cv)
            * (1.0 + self.cfg.age_modulation * age_shift)
            * (1.0 + self.cfg.asa_modulation * asa_shift)
        )
        iiv_factor = float(np.clip(iiv_factor, 0.4, 2.0))
        tonic_eff = tonic * iiv_factor
        peak_eff = peak * iiv_factor

        stimulus = np.zeros(n)

        # ── Phase boundaries ──────────────────────────────────────────────
        preop = next(p for p in timeline.phases if p.name == "preop")
        induction = next(p for p in timeline.phases if p.name == "induction")
        intubation = next(p for p in timeline.phases if p.name == "intubation")
        maintenance = next(p for p in timeline.phases if p.name == "maintenance")
        emergence = next(p for p in timeline.phases if p.name == "emergence")

        # ── Preop: nothing ────────────────────────────────────────────────
        # stimulus stays at 0.

        # ── Intubation: laryngoscopy peak ─────────────────────────────────
        intub_mask = (t >= intubation.start_s) & (t < intubation.end_s)
        if intub_mask.any():
            intub_peak = float(self.rng.uniform(0.50, 0.85))  # strong stimulus
            intub_center = (intubation.start_s + intubation.end_s) / 2.0
            intub_tau = float(self.rng.uniform(20.0, 40.0))
            stimulus[intub_mask] = intub_peak * np.exp(
                -np.abs(t[intub_mask] - intub_center) / intub_tau
            )

        # ── Maintenance: tonic + phasic ───────────────────────────────────
        maint_mask = (t >= maintenance.start_s) & (t < emergence.start_s)
        meta_valle: dict = {}
        if maint_mask.any():
            maint_idx = np.where(maint_mask)[0]
            maint_t = t[maint_idx]

            # Tonic baseline with slow drift (AR(1), τ=300s). A5/A3: más varianza
            # continua para que HR/MAP/BIS tengan desviación realista en mantenimiento.
            drift = _ar1_process(
                n=len(maint_t), dt=dt, tau=300.0,
                mean=tonic_eff, sigma=0.20 * tonic_eff,
                rng=self.rng,
            )
            drift = np.clip(drift, 0.0, 1.0)

            # Phasic events: incision + random surgical events
            phasic = np.zeros(len(maint_t))

            # ── Incision event (first phasic peak) ────────────────────────
            # Timed relative to maintenance start to fall in the remi valley
            incision_delay = self.cfg.incisión_delay_after_induction_s
            incision_t = maintenance.start_s + incision_delay + self.rng.uniform(-30.0, 60.0)
            incision_peak = peak_eff * float(self.rng.uniform(0.80, 1.1))
            incision_tau = float(self.rng.uniform(60.0, 90.0))
            meta_valle["incision_t"] = float(incision_t)
            meta_valle["incision_peak"] = float(incision_peak)

            # ── Remi at incision (for audit) ──────────────────────────────
            inc_idx = int(np.searchsorted(t, incision_t))
            if inc_idx < n:
                meta_valle["ce_remi_at_incision"] = float(ce_remi[inc_idx])

            phasic += incision_peak * np.exp(-np.abs(maint_t - incision_t) / incision_tau)

            # Remaining random events during the rest of maintenance
            n_events = max(0, int(self.rng.poisson(self.cfg.n_phasic_events_mean)))
            event_start = incision_t + 120.0  # at least 2min after incision
            event_end = emergence.start_s - 300.0  # stop before emergence
            if event_end > event_start and n_events > 0:
                event_times = sorted(
                    self.rng.uniform(event_start, event_end, size=n_events)
                )
                for et in event_times:
                    ev_peak = peak_eff * float(self.rng.uniform(0.5, 1.0)) * self.rng.lognormal(0.0, self.cfg.phasic_noise_cv)
                    ev_tau = float(self.rng.uniform(*self.cfg.phasic_tau_range_s))
                    phasic += ev_peak * np.exp(-np.abs(maint_t - et) / ev_tau)

            phasic = np.clip(phasic, 0.0, 1.0)
            stimulus[maint_idx] = np.clip(drift + phasic, 0.0, 1.0)

        # ── Closure ramp (emergence phase) ────────────────────────────────
        emerg_mask = (t >= emergence.start_s) & (t < timeline.total_duration_s)
        if emerg_mask.any():
            emerg_t = t[emerg_mask]
            emerg_dur = max(emerg_t[-1] - emerg_t[0], 1.0)
            # Closure: lower tonic, smaller peaks
            closure_tonic = tonic_eff * self.cfg.closure_tonic_frac
            closure_peak = peak_eff * self.cfg.closure_peak_frac

            # Simple ramp-down plus small closure peaks
            ramp = 1.0 - (emerg_t - emerg_t[0]) / emerg_dur
            closure_stim = closure_tonic + closure_peak * 0.5 * (1.0 - ramp)

            # Where stimulus already exists, blend
            existing = stimulus[emerg_mask]
            stimulus[emerg_mask] = np.where(
                existing > 0,
                existing * np.clip(ramp, 0.0, 1.0),
                closure_stim * np.clip(ramp, 0.0, 1.0),
            )

        stimulus = np.clip(stimulus, 0.0, 1.0)

        # ── Remifentanil attenuation ──────────────────────────────────────
        att_map, att_hr = compute_remi_attenuation(
            ce_remi,
            c50_map=2.5,
            c50_hr=3.0,   # HR slightly harder to attenuate
            gamma=2.0,
        )
        # Ensure 1D arrays
        att_map = np.broadcast_to(np.asarray(att_map), n)
        att_hr = np.broadcast_to(np.asarray(att_hr), n)

        metadata = {
            "surgery_class": sclass.label,
            "surgery_key": op_type,
            "tonic_raw": tonic,
            "peak_raw": peak,
            "tonic_effective": tonic_eff,
            "peak_effective": peak_eff,
            "iiv_factor": iiv_factor,
            **meta_valle,
        }

        return stimulus, att_map, att_hr, metadata


# ═══════════════════════════════════════════════════════════════════════════
# Remifentanil "valle nociceptivo" — bolus + infusion gap modeling
# ═══════════════════════════════════════════════════════════════════════════

def model_remi_valle(
    t: np.ndarray,
    timeline: Timeline,
    rng: np.random.Generator,
    bolus_prob: float = 0.95,
    bolus_peak_ce: float = 5.0,
    refuerzo_prob: float = 0.40,
) -> dict:
    """Compute whether the remi bolus and reinforcement happen for this case.

    This does NOT simulate PK — it returns flags + timing that the main
    simulator uses to construct the remi infusion schedule.

    Returns a dict with:
      - has_bolus: bool
      - bolus_dose_mcg_per_kg: float (for PK simulation)
      - bolus_t_s: float (absolute time of bolus)
      - has_refuerzo: bool
      - refuerzo_dose_frac: float (fraction of induction bolus)
      - incision_expected_t_s: float (when incision is expected)
    """
    induction = next(p for p in timeline.phases if p.name == "induction")
    maintenance = next(p for p in timeline.phases if p.name == "maintenance")

    has_bolus = rng.random() < bolus_prob
    bolus_t = induction.start_s + 15.0  # bolus given ~15s into induction
    bolus_dose = rng.uniform(0.8, 1.5)  # mcg/kg (scaled by LBM in simulator)

    # Expected incision time
    incision_t = maintenance.start_s + rng.uniform(120.0, 240.0)

    has_refuerzo = has_bolus and rng.random() < refuerzo_prob
    refuerzo_frac = rng.uniform(0.3, 0.7) if has_refuerzo else 0.0
    refuerzo_t = incision_t - rng.uniform(30.0, 90.0) if has_refuerzo else 0.0

    return {
        "has_bolus": has_bolus,
        "bolus_dose_mcg_per_kg": float(bolus_dose),
        "bolus_t_s": float(bolus_t),
        "has_refuerzo": has_refuerzo,
        "refuerzo_dose_frac": float(refuerzo_frac),
        "refuerzo_t_s": float(refuerzo_t),
        "incision_expected_t_s": float(incision_t),
    }
