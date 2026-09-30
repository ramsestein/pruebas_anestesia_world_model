"""Cardiovascular response model with separated nociceptive pathways.

Pathway separation (critical for causal identifiability):
  - Aδ/C fibres → dorsal horn → spinothalamic → sympathetic activation → MAP↑, HR↑
  - Propofol depresses thalamocortical (BIS) but NOT the spinal sympathetic arc.
    → BIS=40 with MAP=120 is clinically REAL and the model must reproduce it.
  - Remifentanil blocks at the dorsal horn (µ receptors) IN ORIGIN.
    → Remi attenuates the STIMULUS, not directly MAP.

Nociceptive magnitudes (full stimulus, propofol at BIS~45, NO opioid):
  MAP: +20 to +30 mmHg (peak 15-30 s)
  HR:  +15 to +25 bpm  (peak 30-60 s — slower than MAP)
  BIS: +5 to +10 points, INCONSISTENT (small, noisy — different pathway)

Sevoflurane: Emax on the same hypnotic axis as propofol (MAC-equivalence).
  1 MAC ≈ 2.0 mcg/mL propofol Ce for BIS depression.
  1 MAC ≈ 0.75x the MAP depression of 2.0 mcg/mL propofol (vasodilation).
"""

from __future__ import annotations

import numpy as np


# ── Gain constants (tuned to clinical magnitudes) ─────────────────────────
MAP_NOCICEPTIVE_GAIN = 50.0    # mmHg at full stimulus, no remi (v3.1: reduced from 60)
HR_NOCICEPTIVE_GAIN = 35.0     # bpm at full stimulus, no remi (v3.1: reduced from 40)
BIS_AROUSAL_GAIN = 8.0         # BIS points at full stimulus, no remi

# Propofol → MAP/HR depression — v3.1: increased for more hypotensive events
PROPOFOL_MAP_SLOPE = -5.0      # mmHg per mcg/mL Ce (v3.1: -3.5→-5.0)
PROPOFOL_HR_SLOPE = -2.5       # bpm per mcg/mL Ce (v3.1: -2.0→-2.5)

# Remifentanil direct effects (independent of nociception)
REMI_MAP_DIRECT = -0.3         # mmHg per ng/mL Ce (small direct vasodilation)
REMI_HR_DIRECT = -2.5          # bpm per ng/mL Ce (opioid bradycardia, saturable)

# Sevoflurane: Emax on hypnotic axis
SEVO_MAP_PER_MAC = -12.0       # mmHg per MAC (vasodilation + myocardial depression)
SEVO_HR_PER_MAC = -5.0         # bpm per MAC (mild negative chronotrope)

# Vasopressor scales (empirically calibrated) — v3.1
NORADRENALINE_MAP_PER_MCGKG = 150.0   # mmHg per mcg/kg/min (v3.1: back to 150 for stronger effect)
PHENYLEPHRINE_MAP_PER_MCGKG = 12.0    # mmHg per mcg/kg (bolus, weight-scaled; F2 fix: was 0.012)
PHENYLEPHRINE_RATE_MAP_PER_MCGKG = 40.0  # mmHg per mcg/kg/min (continuous infusion; F2 fix: was 140)

# ── Ephedrine: Emax saturable + tachyphylaxis (v3.2) ────────────────────
# Ephedrine acts indirectly via norepinephrine release → tachyphylaxis.
# Repeated doses deplete NE stores: effect diminishes with cumulative exposure.
# Clinical: 10mg → +15-25 mmHg MAP, +10-20 bpm HR, asymptote ~130 mmHg MAP.
# After ~5 doses (cumulative ~50mg), effect is nearly gone.
EPH_EMAX_MAP = 30.0          # max MAP increase from ephedrine (mmHg) at 70kg
EPH_EMAX_HR = 25.0           # max HR increase from ephedrine (bpm) at 70kg
EPH_ED50_MG = 8.0            # effective dose at half-max effect (mg)
EPH_TACHY_HALF_MG = 25.0     # cumulative mg for 50% tachyphylaxis


class HemodynamicResponse:
    """Map drug concentrations, surgical stimulus, and vasopressors to MAP and HR.

    The remifentanil attenuation of nociception is applied to the STIMULUS
    externally (by the caller), not inside this class. The caller passes
    already-attenuated effective nociceptive drives.
    """

    def __init__(
        self,
        map_baseline: float = 98.0,
        hr_baseline: float = 70.0,
        prop_sensitivity: float = 1.0,
        vasopressor_response: float = 1.0,
    ) -> None:
        self.map_baseline = map_baseline
        self.hr_baseline = hr_baseline
        self.prop_sensitivity = prop_sensitivity
        self.vasopressor_response = vasopressor_response
        self._cumulative_eph_mg = 0.0  # tachyphylaxis tracker

    def reset_tachyphylaxis(self) -> None:
        """Reset cumulative ephedrine tracker (new case)."""
        self._cumulative_eph_mg = 0.0

    def register_eph_bolus(self, dose_mg: float) -> None:
        """Register an ephedrine bolus for tachyphylaxis tracking.
        
        Called by the simulator each time an ephedrine bolus is given.
        """
        self._cumulative_eph_mg += dose_mg

    def _eph_effect(self, decayed_dose_mg: float, weight_kg: float) -> tuple[float, float]:
        """Compute ephedrine MAP and HR effect with Emax + tachyphylaxis.
        
        Args:
            decayed_dose_mg: sum of decayed boluses at current time (mg).
            weight_kg: patient weight.
            
        Returns:
            (map_effect_mmHg, hr_effect_bpm)
            
        Emax model gives diminishing returns per mg at a given moment.
        Tachyphylaxis is driven by cumulative original dose (not decayed):
        each mg given depletes NE stores regardless of PK washout.
        After ~5×10mg (50mg cumulative), effect is nearly gone.
        """
        if decayed_dose_mg <= 0:
            return 0.0, 0.0
        
        # Emax: diminishing returns per mg at this moment (on PK-active dose)
        emax_factor = 1.0 - np.exp(-decayed_dose_mg / EPH_ED50_MG)
        
        # Tachyphylaxis: cumulative ORIGINAL dose depletes NE stores
        # (uses cumulative_eph_mg which tracks total mg given, not decayed)
        tachy_factor = np.exp(-self._cumulative_eph_mg / EPH_TACHY_HALF_MG)
        
        # Weight scaling (Emax values are for 70kg)
        w_factor = 70.0 / weight_kg
        
        map_effect = self.vasopressor_response * EPH_EMAX_MAP * emax_factor * tachy_factor * w_factor
        hr_effect = self.vasopressor_response * EPH_EMAX_HR * emax_factor * tachy_factor * w_factor
        
        return float(map_effect), float(hr_effect)

    def map_from_state(
        self,
        ce_prop: float,
        ce_rem: float,
        sevo_mac: float = 0.0,
        nociceptive_drive_map: float = 0.0,
        noradrenaline_rate: float = 0.0,
        phenylephrine_bolus: float = 0.0,
        phenylephrine_rate: float = 0.0,
        ephedrine_bolus: float = 0.0,
        weight_kg: float = 70.0,
    ) -> float:
        """Return MAP (mmHg).

        Args:
            ce_prop: propofol effect-site concentration (mcg/mL).
            ce_rem: remifentanil effect-site concentration (ng/mL).
            sevo_mac: sevoflurane MAC (age-adjusted).
            nociceptive_drive_map: stimulus × remi_attenuation_map [0, 1].
            noradrenaline_rate: mcg/kg/min.
            phenylephrine_bolus: decayed effective dose (mcg).
            phenylephrine_rate: continuous infusion rate, mcg/kg/min.
            ephedrine_bolus: decayed effective dose (mg).
            weight_kg: patient weight.
        """
        # Anaesthetic depression
        map_depression = self.prop_sensitivity * (
            PROPOFOL_MAP_SLOPE * ce_prop
            + REMI_MAP_DIRECT * ce_rem
            + SEVO_MAP_PER_MAC * sevo_mac
        )

        # Nociceptive drive (already attenuated by remi externally)
        map_noci = self.prop_sensitivity * MAP_NOCICEPTIVE_GAIN * nociceptive_drive_map

        # Vasopressors
        eph_map, eph_hr = self._eph_effect(ephedrine_bolus, weight_kg)
        map_vaso = self.vasopressor_response * (
            NORADRENALINE_MAP_PER_MCGKG * noradrenaline_rate
            + PHENYLEPHRINE_MAP_PER_MCGKG * phenylephrine_bolus / weight_kg
            + PHENYLEPHRINE_RATE_MAP_PER_MCGKG * phenylephrine_rate
        ) + eph_map

        map_value = self.map_baseline + map_depression + map_noci + map_vaso
        return float(np.clip(map_value, 50.0, 200.0))

    def hr_from_state(
        self,
        ce_prop: float,
        ce_rem: float,
        sevo_mac: float = 0.0,
        nociceptive_drive_hr: float = 0.0,
        ephedrine_bolus: float = 0.0,
        weight_kg: float = 70.0,
    ) -> float:
        """Return heart rate (bpm).

        Args:
            ce_prop: propofol Ce (mcg/mL).
            ce_rem: remifentanil Ce (ng/mL).
            sevo_mac: sevoflurane MAC.
            nociceptive_drive_hr: stimulus × remi_attenuation_hr [0, 1].
            ephedrine_bolus: decayed effective dose (mg).
            weight_kg: patient weight.
        """
        # Anaesthetic depression
        hr_depression = self.prop_sensitivity * (
            PROPOFOL_HR_SLOPE * ce_prop
            + SEVO_HR_PER_MAC * sevo_mac
        )

        # Opioid bradycardia (saturable, direct — NOT via nociception)
        hr_opioid = REMI_HR_DIRECT * (ce_rem / (ce_rem + 4.0))

        # Nociceptive drive (already attenuated by remi externally)
        hr_noci = self.prop_sensitivity * HR_NOCICEPTIVE_GAIN * nociceptive_drive_hr

        # Ephedrine chronotropic effect (Emax + tachyphylaxis)
        _, hr_eph = self._eph_effect(ephedrine_bolus, weight_kg)

        hr_value = self.hr_baseline + hr_depression + hr_opioid + hr_noci + hr_eph
        return float(np.clip(hr_value, 45.0, 200.0))

