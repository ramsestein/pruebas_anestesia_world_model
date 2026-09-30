"""Orchestrate one full synthetic case."""

from __future__ import annotations

import numpy as np
import polars as pl

from anessim.actions import Action, ActionType
from anessim.config import SimulatorConfig
from anessim.patients import Patient, PatientSampler
from anessim.pd import C50_PROP, C50_REM, C50_SEVO, HemodynamicResponse, bis_from_ce
from anessim.pk import Infusion, PropofolMarsh, RemifentanilMinto
from anessim.randomization import Randomization
from anessim.sensors import ObservationModel
from anessim.timeline import TimelineGenerator
from anessim.track_presence import TrackPresenceSampler


class CaseSimulator:
    """Simulate one anesthesia case from patient to tracks."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.sampler = PatientSampler(self.config.clinical_path)
        self.timeline_gen = TimelineGenerator(self.rng)
        self.sensor = ObservationModel(self.rng)
        self.track_presence = TrackPresenceSampler(self.config.cases_dir or "dataset/cases", self.rng)
        self.randomization = Randomization(self.rng, self.config.policy_fraction)
        # hr_model is (re)built per-case inside run(), once the patient is known,
        # so baseline + sensitivity can reflect that patient's age/ASA (see run()).

    def run(
        self,
        caseid: int,
        duration_min: float | None = None,
        presence_override: list[str] | None = None,
    ) -> dict:
        """Run a full case and return dict with latent trajectories and events.

        If ``presence_override`` is provided, it is used instead of sampling
        track presence, so the whole cohort can be coordinated from the CLI.
        """
        patient = self.sampler.sample(caseid, rng=self.rng)
        timeline = self.timeline_gen.generate(duration_min=duration_min)
        t = np.arange(0.0, timeline.total_duration_s + self.config.dt_seconds, self.config.dt_seconds)

        # Baseline MAP raised from 90 to 98: the closed-loop vasopressor logic targets
        # 70-85 mmHg once engaged (clinically appropriate), but transient pre-vasopressor
        # dips after induction were pulling the case-long mean below real ART_MBP data.
        # Raising the pre-induction baseline (still a normal awake MAP for a surgical
        # patient) reduces that gap at the source instead of patching it with a fake
        # per-case observation-layer offset.
        # cv raised from the 0.08 default: real cohorts include hypertensive and
        # hypotensive patients (baseline MAP/HR vary far more across individuals
        # than 8% would allow), and the pooled synthetic ART_MBP/HR mean+std were
        # both running low relative to real data (ART_MBP mean/std z~1.1/0.95,
        # HR mean/std z~0.8) despite the closed-loop vasopressor logic already
        # working correctly per-case. Widening the per-patient baseline spread
        # attacks that gap at its source rather than adding more artifacts.
        map_base, hr_base = self.randomization.randomize_baselines(98.0, 70.0, cv=0.16)
        # Inter-patient hemodynamic variability: real patients differ substantially in
        # how much their MAP/HR drop for the same propofol exposure and how well they
        # respond to vasopressors, independent of the population-mean dose-response
        # curve. Scaled by age/ASA (clinically, older/higher-ASA patients are more
        # hemodynamically fragile and have a blunted compensatory/vasopressor response)
        # rather than being pure unexplained noise, so the relationship stays learnable.
        _pd_base = float(self.rng.lognormal(0.0, 0.20))
        _age_shift = patient.age - 50.0
        _asa_shift = max(0, (patient.asa or 2) - 2)
        prop_sensitivity = float(np.clip(_pd_base * (1.0 + 0.008 * _age_shift) * (1.0 + 0.06 * _asa_shift), 0.5, 2.2))
        vasopressor_response = float(np.clip((1.0 / _pd_base) * (1.0 - 0.006 * _age_shift) * (1.0 - 0.05 * _asa_shift), 0.4, 1.6))
        self.hr_model = HemodynamicResponse(
            map_baseline=map_base,
            hr_baseline=hr_base,
            prop_sensitivity=prop_sensitivity,
            vasopressor_response=vasopressor_response,
        )

        actions = self._build_actions(timeline, patient)

        # Build infusions from actions
        propofol_infusions = self._actions_to_infusions(actions, "propofol")
        remifentanil_infusions = self._actions_to_infusions(actions, "remifentanil")
        rocuronium_boluses = [a for a in actions if a.drug == "rocuronium"]
        sevoflurane_changes = [a for a in actions if a.drug == "sevoflurane"]

        # PK simulation
        prop_model = PropofolMarsh(weight_kg=patient.weight, age_y=patient.age)
        remi_model = RemifentanilMinto(
            weight_kg=patient.weight,
            height_cm=patient.height,
            age_y=patient.age,
            sex=patient.sex,
        )
        # Inter-individual PK variability: Marsh/Minto give the population-mean rate
        # constants/volumes for a patient with given weight/age/sex, but real patients
        # vary around that mean (differing hepatic clearance, cardiac output, etc.).
        # Without this, two patients with identical demographics get byte-identical
        # concentration curves, which is not how real pharmacology behaves.
        self._apply_pk_iiv(prop_model)
        self._apply_pk_iiv(remi_model)
        prop_states = prop_model.simulate(propofol_infusions, t)
        remi_states = remi_model.simulate(remifentanil_infusions, t)

        cp_prop = prop_states[:, 0]
        ce_prop = prop_states[:, 3]
        cp_rem = remi_states[:, 0]
        ce_rem = remi_states[:, 3]

        # ── Surgical stimulus profile (nociception) ─────────────────────────
        surgical_stimulus = self._generate_surgical_stimulus(t, timeline, patient)

        # ── Sevoflurane MAC array (for PD coupling) ─────────────────────────
        mac_array = np.array(
            [self._sevoflurane_at(t_i, sevoflurane_changes) for t_i in t]
        )

        # ── PD outputs ──────────────────────────────────────────────────────
        # Inter-patient BIS sensitivity: population-mean C50 values (Bouillon-Bruhn)
        # represent an "average patient"; real sensitivity to propofol/opioid varies
        # appreciably (documented ~20-30% CV), independent of PK differences above.
        bis_sensitivity = float(self.rng.lognormal(0.0, 0.22))
        # Sevoflurane sensitivity also varies between patients
        sevo_sensitivity = float(self.rng.lognormal(0.0, 0.18))
        bis = np.array(
            [
                bis_from_ce(
                    p, r,
                    sevo_mac=m * sevo_sensitivity,
                    c50_prop=C50_PROP * bis_sensitivity,
                    c50_rem=C50_REM * bis_sensitivity,
                    surgical_stimulus=float(s),
                )
                for p, r, m, s in zip(ce_prop, ce_rem, mac_array, surgical_stimulus)
            ]
        )

        # Baseline MAP without vasoactive support, then close the loop
        map_baseline = np.array(
            [
                self.hr_model.map_from_state(
                    ce_prop=p, ce_rem=r,
                    sevo_mac=m, surgical_stimulus=float(s),
                    weight_kg=patient.weight,
                )
                for p, r, m, s in zip(ce_prop, ce_rem, mac_array, surgical_stimulus)
            ]
        )
        actions = self._close_vasoactive_loop(t, map_baseline, actions, patient.weight, patient)

        vasoactive = [a for a in actions if a.action_type in (ActionType.VASOACTIVE_BOLUS, ActionType.VASOACTIVE_INFUSION)]
        noradrenaline_infusions = self._actions_to_infusions(actions, "noradrenaline")

        map_values = np.array(
            [
                self.hr_model.map_from_state(
                    ce_prop=p,
                    ce_rem=r,
                    sevo_mac=m,
                    noradrenaline_rate=self._rate_at(ti, noradrenaline_infusions),
                    phenylephrine_bolus=self._decayed_bolus_effect_at(ti, vasoactive, "phenylephrine", half_life_s=300.0),
                    ephedrine_bolus=self._decayed_bolus_effect_at(ti, vasoactive, "ephedrine", half_life_s=600.0),
                    surgical_stimulus=float(s),
                    weight_kg=patient.weight,
                )
                for ti, p, r, m, s in zip(t, ce_prop, ce_rem, mac_array, surgical_stimulus)
            ]
        )
        hr_values = np.array(
            [
                self.hr_model.hr_from_state(
                    ce_prop=p,
                    ce_rem=r,
                    sevo_mac=m,
                    ephedrine_bolus=self._decayed_bolus_effect_at(ti, vasoactive, "ephedrine", half_life_s=600.0),
                    surgical_stimulus=float(s),
                    weight_kg=patient.weight,
                )
                for ti, p, r, m, s in zip(t, ce_prop, ce_rem, mac_array, surgical_stimulus)
            ]
        )
        # Add intubation tachycardia: sympathetic response to laryngoscopy.
        # Real VitalDB HR_max ~133 bpm (mean), synthetic without this fix ~88 bpm.
        # Spike at intubation.start_s with peak 110-180 bpm, duration 30-120s.
        intub_phase = next((p for p in timeline.phases if p.name == "intubation"), None)
        if intub_phase is not None:
            intub_t0 = float(intub_phase.start_s)
            hr_peak = float(self.rng.uniform(110.0, 180.0))
            hr_spike_dur = float(self.rng.uniform(30.0, 120.0))
            spike_half = hr_spike_dur / 2.0
            spike_center = intub_t0 + spike_half
            spike_mask = (t >= intub_t0) & (t <= intub_t0 + hr_spike_dur)
            if spike_mask.any():
                gaussian = hr_peak * np.exp(-0.5 * ((t[spike_mask] - spike_center) / (spike_half / 2.5)) ** 2)
                hr_values[spike_mask] = np.maximum(hr_values[spike_mask], gaussian)
        spo2 = self._generate_spo2(t)
        etco2 = self._generate_etco2(t)
        fio2 = np.full_like(t, 0.5)
        # MAC for observation tracks (already computed above as mac_array for PD)
        mac = mac_array  # reuse, avoid recomputing
        # The vaporizer dial position changes instantly (that's the truth/action),
        # but what the Primus monitor displays is measured end-tidal concentration,
        # which lags the dial via alveolar wash-in/out kinetics (time constant on
        # the order of a few minutes for sevoflurane). Smoothing here (observation
        # layer only; truth keeps the instantaneous step) turns the artificial
        # square-wave into the continuous curve real monitors actually record.
        from scipy.signal import lfilter as _lfilter_mac

        _mac_dt = float(np.mean(np.diff(t))) if len(t) > 1 else float(self.config.dt_seconds)
        _mac_tau = 150.0
        _mac_alpha = float(np.exp(-_mac_dt / _mac_tau))
        mac_observed_latent = _lfilter_mac([1.0 - _mac_alpha], [1.0, -_mac_alpha], mac)

        # Pre-compute drug rate arrays so they can be used in the observed tracks
        # dict below AND in the truth sidecar (single computation, not duplicated).
        propofol_rate_arr = np.array([self._propofol_rate_at(ti, propofol_infusions) for ti in t])
        remifentanil_rate_arr = np.array([self._propofol_rate_at(ti, remifentanil_infusions) for ti in t])

        # Case-level heavy-tailed gap parameters for ETCO2, matching real dt_std
        # mean=4.77±4.44 (std ≈ mean ⇒ wide spread across cases).
        if self.rng.random() < 0.55:
            etco2_gap_rate, etco2_gap_dur = 0.002, float(self.rng.uniform(15.0, 30.0))
        else:
            etco2_gap_rate = float(self.rng.uniform(0.003, 0.008))
            etco2_gap_dur = float(self.rng.uniform(30.0, 70.0))

        # Build tracks with realistic cadence
        tracks = {
            "time": t,
            "Solar8000/HR": self._observe_hr(t, hr_values),
            "Solar8000/ART_SBP": self._observe_art(t, map_values + 15.0, value_range=(-100, 350)),
            "Solar8000/ART_MBP": self._observe_art(t, map_values, value_range=(-100, 350)),
            "Solar8000/ART_DBP": self._observe_art(t, map_values - 15.0, value_range=(-100, 350)),
            "Solar8000/NIBP_SBP": self.sensor.observe(t, map_values + 15.0, sampling_interval=180.0, noise_sd=4.0, dropout_rate=0.05, value_range=(50, 200)),
            "Solar8000/NIBP_MBP": self.sensor.observe(t, map_values, sampling_interval=180.0, noise_sd=3.0, dropout_rate=0.05, value_range=(30, 180)),
            "Solar8000/NIBP_DBP": self.sensor.observe(t, map_values - 15.0, sampling_interval=180.0, noise_sd=3.0, dropout_rate=0.05, value_range=(30, 140)),
            "Solar8000/PLETH_HR": self.sensor.observe(t, hr_values, sampling_interval=1.0, noise_sd=2.0, dropout_rate=0.55, gap_rate=0.001, gap_duration_s=30.0, value_range=(35, 150)),
            "Solar8000/PLETH_SPO2": self._observe_spo2(t, spo2),
            "Solar8000/ETCO2": self.sensor.observe(t, etco2, sampling_interval=2.0, noise_sd=0.42, dropout_rate=0.45, gap_rate=etco2_gap_rate, gap_duration_s=etco2_gap_dur, value_range=(0, 60)),
            "Solar8000/RR_CO2": self.sensor.observe(t, np.full_like(t, 12.0), sampling_interval=2.0, noise_sd=1.0, dropout_rate=0.40, gap_rate=0.0005, gap_duration_s=20.0, value_range=(4, 40)),
            "Solar8000/BT": self.sensor.observe(t, np.full_like(t, 36.5), sampling_interval=30.0, noise_sd=0.1, dropout_rate=0.20, value_range=(34, 40)),
            "BIS/BIS": self._observe_bis(t, bis),
            "BIS/EMG": self.sensor.observe(t, np.full_like(t, 30.0), sampling_interval=1.0, noise_sd=5.0, dropout_rate=0.45, value_range=(0, 100)),
            "BIS/SQI": self.sensor.observe(t, np.full_like(t, 95.0), sampling_interval=1.0, noise_sd=3.0, dropout_rate=0.45, value_range=(0, 100)),
            "BIS/SEF": self.sensor.observe(t, 15.0 + 0.2 * bis, sampling_interval=1.0, noise_sd=1.0, dropout_rate=0.45, value_range=(0, 30)),
            "BIS/SR": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=1.0, noise_sd=1.0, dropout_rate=0.45, value_range=(0, 100)),
            "BIS/TOTPOW": self.sensor.observe(t, np.full_like(t, 50.0), sampling_interval=1.0, noise_sd=10.0, dropout_rate=0.45, value_range=(0, 100)),
            "Primus/FIO2": self.sensor.observe(t, fio2, sampling_interval=7.0, noise_sd=0.01, dropout_rate=0.80, value_range=(0.21, 1.0)),
            "Primus/ETCO2": self.sensor.observe(t, etco2, sampling_interval=7.0, noise_sd=1.0, dropout_rate=0.80, value_range=(15, 60)),
            "Primus/MAC": self._observe_mac(t, mac_observed_latent),
            "Primus/EXP_SEVO": self.sensor.observe(t, mac_observed_latent * 2.0, sampling_interval=7.0, noise_sd=0.05, dropout_rate=0.80, value_range=(0.0, 5.0)),
            "Primus/INSP_SEVO": self.sensor.observe(t, mac_observed_latent * 2.2, sampling_interval=7.0, noise_sd=0.05, dropout_rate=0.80, value_range=(0.0, 5.0)),
            "Primus/EXP_DES": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=7.0, noise_sd=0.0, dropout_rate=0.80, value_range=(0.0, 10.0)),
            "Primus/INSP_DES": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=7.0, noise_sd=0.0, dropout_rate=0.80, value_range=(0.0, 10.0)),
            "Primus/PEEP_MBAR": self.sensor.observe(t, np.full_like(t, 5.0), sampling_interval=7.0, noise_sd=0.2, dropout_rate=0.80, value_range=(0, 15)),
            "Primus/PIP_MBAR": self.sensor.observe(t, np.full_like(t, 18.0), sampling_interval=7.0, noise_sd=1.0, dropout_rate=0.80, value_range=(5, 40)),
            "Primus/PPLAT_MBAR": self.sensor.observe(t, np.full_like(t, 15.0), sampling_interval=7.0, noise_sd=1.0, dropout_rate=0.80, value_range=(5, 35)),
            "Primus/MV": self.sensor.observe(t, np.full_like(t, 6.0), sampling_interval=7.0, noise_sd=0.3, dropout_rate=0.80, value_range=(1, 20)),
            "Primus/TV": self.sensor.observe(t, np.full_like(t, 500.0), sampling_interval=7.0, noise_sd=30.0, dropout_rate=0.80, value_range=(100, 1500)),
            "Primus/RR_CO2": self.sensor.observe(t, np.full_like(t, 12.0), sampling_interval=7.0, noise_sd=0.5, dropout_rate=0.80, value_range=(4, 40)),
            "Primus/SET_FIO2": self.sensor.observe(t, fio2 * 100.0, sampling_interval=7.0, noise_sd=0.5, dropout_rate=0.80, value_range=(21, 100)),
            "Primus/SET_RR_IPPV": self.sensor.observe(t, np.full_like(t, 12.0), sampling_interval=7.0, noise_sd=0.5, dropout_rate=0.80, value_range=(4, 40)),
            "Primus/SET_TV_L": self.sensor.observe(t, np.full_like(t, 0.5), sampling_interval=7.0, noise_sd=0.01, dropout_rate=0.80, value_range=(0.1, 1.5)),
            "Primus/SET_PIP": self.sensor.observe(t, np.full_like(t, 18.0), sampling_interval=7.0, noise_sd=1.0, dropout_rate=0.80, value_range=(5, 40)),
            "Primus/SET_INSP_TM": self.sensor.observe(t, np.full_like(t, 1.0), sampling_interval=7.0, noise_sd=0.05, dropout_rate=0.80, value_range=(0.5, 3.0)),
            "Primus/SET_INSP_PAUSE": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=7.0, noise_sd=0.0, dropout_rate=0.80, value_range=(0, 1)),
            "Primus/SET_INTER_PEEP": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=7.0, noise_sd=0.0, dropout_rate=0.80, value_range=(0, 1)),
            "Primus/SET_FRESH_FLOW": self.sensor.observe(t, np.full_like(t, 1.0), sampling_interval=7.0, noise_sd=0.1, dropout_rate=0.80, value_range=(0.2, 3.0)),
            "Primus/SET_AGE": self.sensor.observe(t, np.full_like(t, float(patient.age)), sampling_interval=7.0, noise_sd=0.0, dropout_rate=0.80, value_range=(0, 120)),
            "Primus/MAWP_MBAR": self.sensor.observe(t, np.full_like(t, 10.0), sampling_interval=7.0, noise_sd=0.5, dropout_rate=0.80, value_range=(5, 25)),
            "Primus/PAMB_MBAR": self.sensor.observe(t, np.full_like(t, 1013.0), sampling_interval=7.0, noise_sd=1.0, dropout_rate=0.80, value_range=(950, 1050)),
            "Primus/VENT_LEAK": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=7.0, noise_sd=0.5, dropout_rate=0.80, value_range=(0, 10)),
            "Primus/COMPLIANCE": self.sensor.observe(t, np.full_like(t, 50.0), sampling_interval=7.0, noise_sd=3.0, dropout_rate=0.80, value_range=(10, 100)),
            "Primus/FLOW_AIR": self.sensor.observe(t, np.full_like(t, 0.5), sampling_interval=7.0, noise_sd=0.05, dropout_rate=0.80, value_range=(0, 2)),
            "Primus/FLOW_O2": self.sensor.observe(t, np.full_like(t, 0.5), sampling_interval=7.0, noise_sd=0.05, dropout_rate=0.80, value_range=(0, 2)),
            "Primus/FLOW_N2O": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=7.0, noise_sd=0.0, dropout_rate=0.80, value_range=(0, 1)),
            "Primus/FEN2O": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=7.0, noise_sd=0.0, dropout_rate=0.80, value_range=(0, 1)),
            "Primus/FIN2O": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=7.0, noise_sd=0.0, dropout_rate=0.80, value_range=(0, 1)),
            "Primus/FEO2": self.sensor.observe(t, fio2 * 100.0, sampling_interval=7.0, noise_sd=0.5, dropout_rate=0.80, value_range=(21, 100)),
            "Primus/INCO2": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=7.0, noise_sd=0.0, dropout_rate=0.80, value_range=(0, 1)),
            "Solar8000/VENT_RR": self.sensor.observe(t, np.full_like(t, 12.0), sampling_interval=2.0, noise_sd=0.5, dropout_rate=0.80, value_range=(4, 40)),
            "Solar8000/VENT_MAWP": self.sensor.observe(t, np.full_like(t, 10.0), sampling_interval=2.0, noise_sd=0.5, dropout_rate=0.80, value_range=(5, 25)),
            "Solar8000/VENT_INSP_TM": self.sensor.observe(t, np.full_like(t, 1.0), sampling_interval=2.0, noise_sd=0.05, dropout_rate=0.80, value_range=(0.5, 3.0)),
            "Solar8000/VENT_SET_TV": self.sensor.observe(t, np.full_like(t, 500.0), sampling_interval=2.0, noise_sd=30.0, dropout_rate=0.80, value_range=(100, 1500)),
            "Solar8000/VENT_SET_FIO2": self.sensor.observe(t, fio2 * 100.0, sampling_interval=2.0, noise_sd=0.5, dropout_rate=0.80, value_range=(21, 100)),
            "Solar8000/FIO2": self.sensor.observe(t, fio2 * 100.0, sampling_interval=2.0, noise_sd=0.5, dropout_rate=0.80, value_range=(21, 100)),
            "Solar8000/FEO2": self.sensor.observe(t, fio2 * 100.0, sampling_interval=2.0, noise_sd=0.5, dropout_rate=0.80, value_range=(21, 100)),
            "Solar8000/INCO2": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=2.0, noise_sd=0.0, dropout_rate=0.80, value_range=(0, 1)),
            "Solar8000/RR": self.sensor.observe(t, np.full_like(t, 12.0), sampling_interval=2.0, noise_sd=1.0, dropout_rate=0.80, value_range=(4, 40)),
            "Solar8000/ST_II": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=2.0, noise_sd=0.1, dropout_rate=0.80, value_range=(-2, 2)),
            "Solar8000/ST_III": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=2.0, noise_sd=0.1, dropout_rate=0.80, value_range=(-2, 2)),
            "Solar8000/ST_I": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=2.0, noise_sd=0.1, dropout_rate=0.80, value_range=(-2, 2)),
            "Solar8000/ST_AVR": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=2.0, noise_sd=0.1, dropout_rate=0.80, value_range=(-2, 2)),
            "Solar8000/ST_AVL": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=2.0, noise_sd=0.1, dropout_rate=0.80, value_range=(-2, 2)),
            "Solar8000/ST_AVF": self.sensor.observe(t, np.full_like(t, 0.0), sampling_interval=2.0, noise_sd=0.1, dropout_rate=0.80, value_range=(-2, 2)),
            "Solar8000/GAS2_INSPIRED": self.sensor.observe(t, mac * 2.0, sampling_interval=2.0, noise_sd=0.05, dropout_rate=0.80, value_range=(0.0, 5.0)),
            "Solar8000/GAS2_EXPIRED": self.sensor.observe(t, mac * 2.0, sampling_interval=2.0, noise_sd=0.05, dropout_rate=0.80, value_range=(0.0, 5.0)),
            "Solar8000/CVP": self.sensor.observe(t, np.full_like(t, 8.0), sampling_interval=2.0, noise_sd=1.0, dropout_rate=0.80, value_range=(0, 20)),
            "Solar8000/VENT_SET_PCP": self.sensor.observe(t, np.full_like(t, 15.0), sampling_interval=2.0, noise_sd=1.0, dropout_rate=0.80, value_range=(5, 30)),
            "Solar8000/VENT_PIP": self.sensor.observe(t, np.full_like(t, 18.0), sampling_interval=2.0, noise_sd=1.0, dropout_rate=0.80, value_range=(5, 40)),
            "Solar8000/VENT_PPLAT": self.sensor.observe(t, np.full_like(t, 15.0), sampling_interval=2.0, noise_sd=1.0, dropout_rate=0.80, value_range=(5, 35)),
            "Solar8000/VENT_TV": self.sensor.observe(t, np.full_like(t, 500.0), sampling_interval=2.0, noise_sd=30.0, dropout_rate=0.80, value_range=(100, 1500)),
            "Solar8000/VENT_MV": self.sensor.observe(t, np.full_like(t, 6.0), sampling_interval=2.0, noise_sd=0.3, dropout_rate=0.80, value_range=(1, 20)),
            "Orchestra/PPF20_RATE": self._observe_drug_rate(
                t, propofol_rate_arr, dropout_rate=0.46, value_range=(0, 2000), unit_factor=1.4,
                burst_range=(150.0, 1100.0), noise_log_mean=5.0,
            ),
            "Orchestra/PPF20_CE": self.sensor.observe(t, ce_prop, sampling_interval=1.0, noise_sd=0.05, dropout_rate=0.0, value_range=(0, 20)),
            "Orchestra/PPF20_CP": self.sensor.observe(t, cp_prop, sampling_interval=1.0, noise_sd=0.05, dropout_rate=0.0, value_range=(0, 20)),
            "Orchestra/PPF20_CT": self.sensor.observe(t, prop_states[:, 2], sampling_interval=1.0, noise_sd=0.05, dropout_rate=0.0, value_range=(0, 20)),
            "Orchestra/PPF20_VOL": np.cumsum(np.array([self._propofol_rate_at(ti, propofol_infusions) for ti in t]) / 60.0 * self.config.dt_seconds),
            "Orchestra/RFTN20_RATE": self._observe_drug_rate(
                t, remifentanil_rate_arr, dropout_rate=0.46, value_range=(0, 1500), unit_factor=2.0,
                burst_range=(80.0, 920.0), noise_log_mean=2.2,
            ),
            "Orchestra/RFTN20_CE": self.sensor.observe(t, ce_rem, sampling_interval=1.0, noise_sd=0.1, dropout_rate=0.0, value_range=(0, 50)),
            "Orchestra/RFTN20_CP": self.sensor.observe(t, cp_rem, sampling_interval=1.0, noise_sd=0.1, dropout_rate=0.0, value_range=(0, 50)),
            "Orchestra/RFTN20_CT": self.sensor.observe(t, remi_states[:, 2], sampling_interval=1.0, noise_sd=0.1, dropout_rate=0.0, value_range=(0, 50)),
            "Orchestra/RFTN20_VOL": np.cumsum(np.array([self._propofol_rate_at(ti, remifentanil_infusions) for ti in t]) / 60.0 * self.config.dt_seconds),
            "Orchestra/ROC_RATE": self.sensor.observe_discrete_event(t, [a.t_s for a in rocuronium_boluses], [a.value or 0 for a in rocuronium_boluses]),
            "Orchestra/ROC_VOL": np.cumsum(np.array([self._rocuronium_rate_at(ti, rocuronium_boluses) for ti in t])),
            "Orchestra/EPH_RATE": self.sensor.observe_discrete_event(t, [a.t_s for a in vasoactive if a.drug == "ephedrine"], [a.value or 0 for a in vasoactive if a.drug == "ephedrine"]),
            "Orchestra/EPH_VOL": np.cumsum(np.array([self._vasoactive_bolus_rate_at(ti, vasoactive, "ephedrine") for ti in t])),
            "Orchestra/PHEN_RATE": self.sensor.observe_discrete_event(t, [a.t_s for a in vasoactive if a.drug == "phenylephrine"], [a.value or 0 for a in vasoactive if a.drug == "phenylephrine"]),
            "Orchestra/PHEN_VOL": np.cumsum(np.array([self._vasoactive_bolus_rate_at(ti, vasoactive, "phenylephrine") for ti in t])),
            "Orchestra/NEPI_RATE": self.sensor.observe_discrete_event(t, [i.start_s for i in noradrenaline_infusions], [i.rate for i in noradrenaline_infusions]),
            "Orchestra/NEPI_VOL": np.cumsum(np.array([self._rate_at(ti, noradrenaline_infusions) for ti in t]) / 60.0 * self.config.dt_seconds),
        }

        # Action-derived signals for truth sidecar
        # (propofol_rate_arr and remifentanil_rate_arr already computed above)
        rocuronium_dose_arr = np.array([self._cumulative_bolus_at(ti, rocuronium_boluses, "rocuronium") for ti in t])
        noradrenaline_rate_arr = np.array([self._rate_at(ti, noradrenaline_infusions) for ti in t])
        ephedrine_dose_arr = np.array([self._cumulative_bolus_at(ti, vasoactive, "ephedrine") for ti in t])
        phenylephrine_dose_arr = np.array([self._cumulative_bolus_at(ti, vasoactive, "phenylephrine") for ti in t])
        sevoflurane_mac_arr = np.array([self._sevoflurane_at(ti, sevoflurane_changes) for ti in t])

        # Latent truth sidecar
        truth = {
            "caseid": caseid,
            "time": t,
            "phase": [timeline.get_phase_at(ti) for ti in t],
            "ce_propofol": ce_prop,
            "cp_propofol": cp_prop,
            "ce_remifentanil": ce_rem,
            "cp_remifentanil": cp_rem,
            "bis": bis,
            "map": map_values,
            "map_baseline": map_baseline,
            "hr": hr_values,
            "propofol_rate": propofol_rate_arr,
            "remifentanil_rate": remifentanil_rate_arr,
            "rocuronium_dose": rocuronium_dose_arr,
            "noradrenaline_rate": noradrenaline_rate_arr,
            "ephedrine_dose": ephedrine_dose_arr,
            "phenylephrine_dose": phenylephrine_dose_arr,
            "sevoflurane_mac": sevoflurane_mac_arr,
            "surgical_stimulus": surgical_stimulus,
            "actions": [a.summary() for a in actions],
        }

        # Sample which tracks are present for this case based on real prevalence
        always_present = {"time"}
        optional_tracks = [c for c in tracks.keys() if c not in always_present]
        if presence_override is not None:
            present_tracks = [c for c in presence_override if c in tracks]
        else:
            present_tracks = self.track_presence.sample(optional_tracks)
        tracks = {c: tracks[c] for c in always_present | set(present_tracks)}

        clinical_row = patient.to_clinical_dict()
        clinical_row["caseend"] = int(timeline.total_duration_s)
        clinical_row["aneend"] = float(timeline.total_duration_s)
        clinical_row["opend"] = int(timeline.total_duration_s - 10 * 60)
        clinical_row["dis"] = int(timeline.total_duration_s + 24 * 3600)
        clinical_row["intraop_ppf"] = int(tracks["Orchestra/PPF20_VOL"][-1]) if "Orchestra/PPF20_VOL" in tracks else 0
        clinical_row["intraop_ftn"] = int(tracks["Orchestra/RFTN20_VOL"][-1]) if "Orchestra/RFTN20_VOL" in tracks else 0

        return {
            "patient": patient,
            "timeline": timeline,
            "tracks": tracks,
            "truth": truth,
            "clinical_row": clinical_row,
            "actions": actions,
        }

    def _apply_pk_iiv(
        self,
        model,
        k_cv: float = 0.20,
        v_cv: float = 0.15,
        ke0_cv: float = 0.15,
    ) -> None:
        """Apply inter-individual PK variability on top of the population-mean model.

        Published PK models (Marsh, Minto) give a single 'typical patient' set of
        rate constants/volumes scaled by weight/age/sex; real inter-patient
        variability around that mean (differing hepatic clearance, cardiac output,
        body composition, etc.) is normally ~15-25% CV. This is what makes
        pharmacology genuinely uncertain rather than a deterministic function of
        demographics, and is the kind of variability a model should learn to expect.
        """
        k_factor = float(self.rng.lognormal(0.0, k_cv))
        v_factor = float(self.rng.lognormal(0.0, v_cv))
        ke0_factor = float(self.rng.lognormal(0.0, ke0_cv))
        model.k10 *= k_factor
        model.k12 *= k_factor
        model.k13 *= k_factor
        model.k21 *= k_factor
        model.k31 *= k_factor
        model.v1 *= v_factor
        model.v2 *= v_factor
        model.v3 *= v_factor
        model.ke0 *= ke0_factor

    def _generate_surgical_stimulus(
        self,
        t: np.ndarray,
        timeline: Timeline,
        patient: Patient,
    ) -> np.ndarray:
        """Generate a nociceptive stimulus profile from surgery type and timeline.

        The stimulus models three components:
          1. Tonic baseline during maintenance (intensity depends on surgery type).
          2. Phasic peaks at surgical events (incision, dissection, closure) with
             sharp onset and exponential decay (τ ≈ 60-120 s).
          3. Random low-amplitude fluctuations (AR(1), τ ≈ 30 s).

        Surgery type → baseline intensity mapping:
          GS (general):  0.50   GY (gynecology): 0.40   NE (neuro):   0.30
          OR (orthopedic): 0.60 TH (thoracic):   0.50   UR (urology): 0.40
          OT (other):    0.50

        Returns:
            stimulus: (T,) array in [0, 1] range.
        """
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else self.config.dt_seconds

        # Surgery-type baseline intensity
        _OPTYPE_BASELINE = {
            "GS": 0.50, "GY": 0.40, "NE": 0.30,
            "OR": 0.60, "TH": 0.50, "UR": 0.40,
        }
        baseline = _OPTYPE_BASELINE.get(patient.optype, 0.50)
        # Inter-case variability in stimulus perception (±20%)
        baseline *= float(self.rng.lognormal(0.0, 0.15))
        baseline = float(np.clip(baseline, 0.15, 0.85))

        stimulus = np.zeros(n)

        # Phases
        preop = next(p for p in timeline.phases if p.name == "preop")
        induction = next(p for p in timeline.phases if p.name == "induction")
        intubation = next(p for p in timeline.phases if p.name == "intubation")
        maintenance = next(p for p in timeline.phases if p.name == "maintenance")
        emergence = next(p for p in timeline.phases if p.name == "emergence")

        # Preop / induction: low stimulus (no surgical manipulation)
        # Small peak at intubation (laryngoscopy is a strong stimulus)
        intub_mask = (t >= intubation.start_s) & (t < intubation.end_s)
        if intub_mask.any():
            intub_peak = float(self.rng.uniform(0.4, 0.7))
            intub_center = (intubation.start_s + intubation.end_s) / 2.0
            intub_tau = float(self.rng.uniform(15.0, 30.0))
            stimulus[intub_mask] = intub_peak * np.exp(
                -np.abs(t[intub_mask] - intub_center) / intub_tau
            )

        # Maintenance: tonic baseline + phasic surgical events
        maint_mask = (t >= maintenance.start_s) & (t < emergence.start_s)
        if maint_mask.any():
            maint_t = t[maint_mask]
            maint_idx = np.where(maint_mask)[0]
            # Tonic baseline with slow drift (AR(1), τ=120s)
            tau_drift = 120.0
            alpha_d = float(np.exp(-dt / tau_drift))
            sigma_drift = 0.05
            drift = np.zeros(len(maint_t))
            drift[0] = baseline
            for i in range(1, len(maint_t)):
                drift[i] = (
                    alpha_d * drift[i - 1]
                    + (1.0 - alpha_d) * baseline
                    + self.rng.normal(0.0, sigma_drift * np.sqrt(1.0 - alpha_d**2))
                )

            # Phasic peaks: incision, periodic dissection events, closure
            n_events = int(self.rng.poisson(4) + 2)  # 2-6 surgical events
            event_times = sorted(
                self.rng.uniform(maintenance.start_s + 120.0, emergence.start_s - 300.0, size=n_events)
            )
            phasic = np.zeros(len(maint_t))
            for et in event_times:
                peak = float(self.rng.uniform(0.3, 0.6))
                tau_p = float(self.rng.uniform(60.0, 120.0))
                phasic += peak * np.exp(-np.abs(maint_t - et) / tau_p)
            phasic = np.clip(phasic, 0.0, 1.0)

            stimulus[maint_idx] = np.clip(drift + phasic, 0.0, 1.0)

        # Emergence: ramp down stimulus
        emerg_mask = (t >= emergence.start_s) & (t < timeline.total_duration_s)
        if emerg_mask.any():
            emerg_t = t[emerg_mask]
            emerg_dur = max(emerg_t[-1] - emerg_t[0], 1.0)
            ramp = 1.0 - (emerg_t - emerg_t[0]) / emerg_dur
            stimulus[emerg_mask] = np.where(
                stimulus[emerg_mask] > 0,
                stimulus[emerg_mask] * np.clip(ramp, 0.0, 1.0),
                0.0,
            )

        return np.clip(stimulus, 0.0, 1.0)

    def _build_actions(self, timeline, patient: Patient) -> list[Action]:
        actions = []
        # Pre-op monitor
        actions.append(Action(t_s=0, action_type=ActionType.VENTILATOR_SETTING, drug="fio2", value=0.5, unit="fraction"))

        # Induction: propofol bolus + infusion
        induction = next(p for p in timeline.phases if p.name == "induction")
        propofol_bolus = 3.0 * patient.weight  # mg
        actions.append(Action(t_s=induction.start_s, action_type=ActionType.BOLUS, drug="propofol", value=propofol_bolus, unit="mg", route="IV"))
        actions.append(Action(t_s=induction.start_s, action_type=ActionType.INFUSION_START, drug="propofol", value=10.0, unit="mg/min", route="IV"))

        # Remifentanil infusion throughout
        actions.append(Action(t_s=induction.start_s, action_type=ActionType.INFUSION_START, drug="remifentanil", value=15.0, unit="mcg/min", route="IV"))

        # Intubation: rocuronium bolus
        intubation = next(p for p in timeline.phases if p.name == "intubation")
        actions.append(Action(t_s=intubation.start_s, action_type=ActionType.BOLUS, drug="rocuronium", value=0.6 * patient.weight, unit="mg", route="IV"))

        # Maintenance: adjust propofol and remifentanil by simple closed loop
        maintenance = next(p for p in timeline.phases if p.name == "maintenance")
        n_adjustments = max(1, int((maintenance.end_s - maintenance.start_s) // 300))
        remi_running = True
        for i in range(1, n_adjustments + 1):
            t_adj = maintenance.start_s + i * 300
            base_rate = 12.0 + self.rng.normal(0, 3.0)
            rate = self.randomization.perturb_infusion_rate(max(0.0, base_rate))
            actions.append(Action(t_s=t_adj, action_type=ActionType.INFUSION_CHANGE, drug="propofol", value=rate, unit="mg/min"))
            # Intermittent remifentanil pauses to mimic real sparse rate recordings.
            # Titration variability (std=2.5 mcg/min) reflects genuine clinical dose
            # adjustment rather than being artificially flattened just to shrink IQR.
            if self.rng.random() < 0.10:
                if remi_running:
                    actions.append(Action(t_s=t_adj, action_type=ActionType.INFUSION_STOP, drug="remifentanil", value=0.0, unit="mcg/min"))
                    remi_running = False
                else:
                    remi_rate = 15.0 + self.rng.normal(0, 2.5)
                    actions.append(Action(t_s=t_adj, action_type=ActionType.INFUSION_START, drug="remifentanil", value=max(0.0, remi_rate), unit="mcg/min"))
                    remi_running = True
            elif remi_running:
                remi_rate = 15.0 + self.rng.normal(0, 2.5)
                actions.append(Action(t_s=t_adj, action_type=ActionType.INFUSION_CHANGE, drug="remifentanil", value=max(0.0, remi_rate), unit="mcg/min"))

        # Emergence: stop propofol and remifentanil
        emergence = next(p for p in timeline.phases if p.name == "emergence")
        actions.append(Action(t_s=emergence.start_s, action_type=ActionType.INFUSION_STOP, drug="propofol", value=0, unit="mg/min"))
        actions.append(Action(t_s=emergence.start_s, action_type=ActionType.INFUSION_STOP, drug="remifentanil", value=0, unit="mcg/min"))

        # Optional sevoflurane during maintenance. Case mix of TIVA (MAC=0, this
        # branch skipped) vs volatile-agent cases gives the inter-case variance seen
        # in real VitalDB MAC. Values capped at 2.0 (clinically deep anesthesia,
        # rarely exceeded intraoperatively without severe cardiovascular depression),
        # not 3.0, which is essentially never used clinically.
        if self.rng.random() < 0.50:
            n_mac_changes = 1 + self.rng.poisson(1.5)
            mac_times = sorted(self.rng.uniform(maintenance.start_s, emergence.start_s, size=n_mac_changes))
            # Only the first setting is an independent draw; later adjustments are
            # small clinician nudges (+/-) from the current dial position, matching
            # how MAC is actually titrated in practice (never reset to an unrelated
            # random value mid-case).
            mac = float(self.rng.lognormal(np.log(1.1), 0.3))
            mac = min(mac, 2.0)
            for i, mac_t in enumerate(mac_times):
                if i > 0:
                    mac = float(np.clip(mac * self.rng.lognormal(0.0, 0.12), 0.3, 2.0))
                actions.append(Action(t_s=mac_t, action_type=ActionType.VENTILATOR_SETTING, drug="sevoflurane", value=mac, unit="MAC"))
            actions.append(Action(t_s=emergence.start_s, action_type=ActionType.VENTILATOR_SETTING, drug="sevoflurane", value=0.0, unit="MAC"))

        return sorted(actions, key=lambda a: a.t_s)

    def _actions_to_infusions(self, actions: list[Action], drug: str) -> list[Infusion]:
        """Convert INFUSION_START/CHANGE/VASOACTIVE_INFUSION/STOP events into piecewise constant Infusions."""
        infusions = []
        current_rate = 0.0
        last_start = 0.0
        for a in sorted(actions, key=lambda x: x.t_s):
            if a.drug != drug:
                continue
            if a.action_type in (ActionType.INFUSION_START, ActionType.INFUSION_CHANGE, ActionType.VASOACTIVE_INFUSION):
                if current_rate > 0:
                    infusions.append(Infusion(start_s=last_start, end_s=a.t_s, rate=current_rate, drug=drug))
                current_rate = a.value or 0.0
                last_start = a.t_s
            elif a.action_type == ActionType.INFUSION_STOP:
                if current_rate > 0:
                    infusions.append(Infusion(start_s=last_start, end_s=a.t_s, rate=current_rate, drug=drug))
                current_rate = 0.0
                last_start = a.t_s
        if current_rate > 0:
            infusions.append(Infusion(start_s=last_start, end_s=float("inf"), rate=current_rate, drug=drug))
        return infusions

    def _propofol_rate_at(self, t: float, infusions: list[Infusion]) -> float:
        for i in infusions:
            if i.start_s <= t <= i.end_s or (t >= i.start_s and i.end_s == float("inf")):
                return i.rate
        return 0.0

    def _rocuronium_rate_at(self, t: float, boluses: list[Action]) -> float:
        # Approximate: bolus delivered over 1 min.
        rate = 0.0
        for b in boluses:
            if b.t_s <= t <= b.t_s + 60:
                rate += (b.value or 0.0) / 60.0
        return rate

    def _sevoflurane_at(self, t: float, events: list[Action]) -> float:
        current = 0.0
        for a in sorted(events, key=lambda x: x.t_s):
            if a.t_s <= t:
                current = a.value or 0.0
        return current

    def _rate_at(self, t: float, infusions: list[Infusion]) -> float:
        for i in infusions:
            if i.start_s <= t <= i.end_s or (t >= i.start_s and i.end_s == float("inf")):
                return i.rate
        return 0.0

    def _cumulative_bolus_at(self, t: float, actions: list[Action], drug: str) -> float:
        return sum(a.value or 0.0 for a in actions if a.drug == drug and a.t_s <= t)

    def _decayed_bolus_effect_at(
        self, t: float, actions: list[Action], drug: str, half_life_s: float
    ) -> float:
        """Return decayed bolus effect at time t using exponential washout."""
        if half_life_s <= 0:
            return self._cumulative_bolus_at(t, actions, drug)
        tau = half_life_s / np.log(2)
        return sum(
            (a.value or 0.0) * np.exp(-(t - a.t_s) / tau)
            for a in actions
            if a.drug == drug and a.t_s <= t
        )

    def _vasoactive_bolus_rate_at(self, t: float, actions: list[Action], drug: str) -> float:
        rate = 0.0
        for a in actions:
            if a.drug == drug and a.t_s <= t <= a.t_s + 60:
                rate += (a.value or 0.0) / 60.0
        return rate

    def _close_vasoactive_loop(
        self,
        t: np.ndarray,
        map_baseline: np.ndarray,
        actions: list[Action],
        weight_kg: float,
        patient: Patient,
    ) -> list[Action]:
        """Reactively add vasopressors when baseline MAP falls below thresholds.

        Uses a continuous noradrenaline infusion for sustained hypotension and
        intermittent boluses for acute drops.  Includes stochastic perturbations:
          - Random delay before starting noradrenaline (clinician reaction time).
          - Dose variability (CV ~25%) beyond the exponential draw.
          - Occasional ineffective bolus (5-10% chance of no MAP response).
        """
        MAP_BOLUS = 60.0  # mmHg (acute drop)
        MAP_NORAD_START = 70.0  # mmHg
        MAP_NORAD_STOP = 85.0  # mmHg
        BOLUS_INTERVAL_S = 300.0  # minimum time between boluses
        COOLDOWN_AFTER_STOP_S = 300.0  # do not treat during emergence

        stop_times = [a.t_s for a in actions if a.action_type == ActionType.INFUSION_STOP and a.drug in ("propofol", "remifentanil")]
        emergence_s = min(stop_times) if stop_times else t[-1]
        active_window = t <= emergence_s - COOLDOWN_AFTER_STOP_S

        new_actions = list(actions)
        norad_running = False
        norad_last_toggle_s = -np.inf
        last_bolus_s = -np.inf
        window = int(60.0 / self.config.dt_seconds)  # 60-second moving average
        MIN_NORAD_DURATION_S = 600.0

        # ── Stochastic perturbation parameters (per-case) ──────────────────
        # Clinician reaction delay before starting noradrenaline
        norad_start_delay = float(self.rng.uniform(0.0, 120.0))  # 0-2 min
        norad_dose_cv = float(self.rng.uniform(0.15, 0.35))       # dose variability
        bolus_failure_prob = float(self.rng.uniform(0.05, 0.10))   # 5-10% ineffective
        norad_start_pending_since = float("inf")

        for i in range(len(t)):
            if not active_window[i]:
                continue
            map_ma = float(np.mean(map_baseline[max(0, i - window + 1) : i + 1]))
            can_toggle = t[i] - norad_last_toggle_s >= MIN_NORAD_DURATION_S

            # Start noradrenaline with stochastic delay
            if not norad_running and map_ma < MAP_NORAD_START and can_toggle:
                if norad_start_pending_since == float("inf"):
                    norad_start_pending_since = t[i]
                if t[i] - norad_start_pending_since >= norad_start_delay:
                    base_dose = 0.1 + self.rng.exponential(0.05)
                    dose = base_dose * self.rng.lognormal(0.0, norad_dose_cv)
                    new_actions.append(
                        Action(
                            t_s=t[i],
                            action_type=ActionType.VASOACTIVE_INFUSION,
                            drug="noradrenaline",
                            value=float(np.clip(dose, 0.03, 0.5)),
                            unit="mcg/kg/min",
                            route="IV",
                        )
                    )
                    norad_running = True
                    norad_last_toggle_s = t[i]
                    norad_start_pending_since = float("inf")
            else:
                norad_start_pending_since = float("inf")

            # Stop noradrenaline with hysteresis
            if norad_running and map_ma > MAP_NORAD_STOP and can_toggle:
                new_actions.append(
                    Action(
                        t_s=t[i],
                        action_type=ActionType.INFUSION_STOP,
                        drug="noradrenaline",
                        value=0.0,
                        unit="mcg/kg/min",
                    )
                )
                norad_running = False
                norad_last_toggle_s = t[i]

            # Acute bolus for moderate hypotension (with failure probability)
            if map_baseline[i] < MAP_BOLUS and t[i] - last_bolus_s >= BOLUS_INTERVAL_S:
                if self.rng.random() > bolus_failure_prob:  # effective bolus
                    choice = self.rng.choice(["ephedrine", "phenylephrine"])
                    dose_cv_bolus = float(self.rng.uniform(0.1, 0.3))
                    if choice == "ephedrine":
                        base = 5.0 + self.rng.exponential(5.0)
                        new_actions.append(
                            Action(
                                t_s=t[i],
                                action_type=ActionType.VASOACTIVE_BOLUS,
                                drug="ephedrine",
                                value=float(np.clip(
                                    base * self.rng.lognormal(0.0, dose_cv_bolus), 2.0, 25.0,
                                )),
                                unit="mg",
                                route="IV",
                            )
                        )
                    else:
                        base = 50.0 + self.rng.exponential(50.0)
                        new_actions.append(
                            Action(
                                t_s=t[i],
                                action_type=ActionType.VASOACTIVE_BOLUS,
                                drug="phenylephrine",
                                value=float(np.clip(
                                    base * self.rng.lognormal(0.0, dose_cv_bolus), 20.0, 300.0,
                                )),
                                unit="mcg",
                                route="IV",
                            )
                        )
                # Whether effective or not, record the attempt
                last_bolus_s = t[i]

        # Ensure noradrenaline is stopped before emergence if still running
        if norad_running:
            new_actions.append(
                Action(
                    t_s=emergence_s - COOLDOWN_AFTER_STOP_S,
                    action_type=ActionType.INFUSION_STOP,
                    drug="noradrenaline",
                    value=0.0,
                    unit="mcg/kg/min",
                )
            )
        return sorted(new_actions, key=lambda a: a.t_s)

    # ------------------------------------------------------------------
    # Signal-processing helpers
    # ------------------------------------------------------------------
    def _ema(self, x: np.ndarray, tau: float, dt: float = 0.5) -> np.ndarray:
        """Apply causal exponential moving average to array x.

        Args:
            tau: time constant in seconds.
            dt:  simulation timestep in seconds (default 0.5 s).
        """
        from scipy.signal import lfilter
        alpha = float(np.exp(-dt / tau))
        return lfilter([1.0 - alpha], [1.0, -alpha], x).astype(x.dtype)

    def _observe_spo2(self, t: np.ndarray, spo2_values: np.ndarray) -> np.ndarray:
        """Observe SpO2 with block masking to match real VitalDB pattern.

        Real PLETH_SPO2: missingness=86.2%, dt_median=2.0s, jitter_dt≈2.8s.
        Scattered dropout gives dt_median≈3.5s (wrong). Block masking with
        occasional large gaps achieves dt_median=2.0s and correct jitter.
        Block covers ~52-64% of case duration; with dropout=5% and 2s cadence:
          missingness = 1 - block_frac × 0.95 × 0.25 ≈ 86.2%.
        Large gaps (gap_rate=0.002, gap_duration=60s) generate jitter_dt≈2.8s.
        """
        observed = self.sensor.observe(
            t, spo2_values,
            sampling_interval=2.0, noise_sd=0.0, dropout_rate=0.05,
            gap_rate=0.002, gap_duration_s=60.0, value_range=(60, 100),
        )
        duration = t[-1] - t[0]
        block_frac = float(self.rng.uniform(0.52, 0.64))
        block_duration = block_frac * duration
        block_start = float(self.rng.uniform(0, max(0.0, duration - block_duration)))
        block_end = block_start + block_duration
        observed[(t < block_start) | (t > block_end)] = np.nan

        # Occasional desaturation events: SPO2 dips to 80-94% for 20-60s.
        # Real SPO2 distribution has negative skew and elevated kurtosis from these events.
        n_desat = int(self.rng.poisson(2.5))
        for _ in range(n_desat):
            dip_start = float(self.rng.uniform(block_start, max(block_start, block_end - 60.0)))
            dip_dur = float(self.rng.exponential(30.0))
            target = float(self.rng.uniform(80.0, 94.0))
            dip_mask = (t >= dip_start) & (t <= dip_start + dip_dur) & (~np.isnan(observed))
            if dip_mask.sum() > 0:
                observed[dip_mask] = np.clip(
                    target + self.rng.normal(0.0, 2.0, size=dip_mask.sum()), 60.0, 100.0
                )
        # Brief 100% plateau (pre-oxygenation / peak oxygenation moment): real
        # SPO2_max = 100 in ~100% of cases. Insert a short window at 100.0 so the
        # observed case max matches, regardless of where the block/desat land.
        if self.rng.random() < 0.97:
            plateau_start = float(self.rng.uniform(block_start, max(block_start, block_end - 30.0)))
            plateau_dur = float(self.rng.uniform(10.0, 40.0))
            plateau_mask = (t >= plateau_start) & (t <= plateau_start + plateau_dur) & (~np.isnan(observed))
            if plateau_mask.sum() > 0:
                observed[plateau_mask] = 100.0
        # Round to nearest integer — real pulse oximeters report integer %
        observed = np.where(np.isnan(observed), np.nan, np.round(observed))
        return observed

    def _observe_art(self, t: np.ndarray, latent: np.ndarray, value_range: tuple) -> np.ndarray:
        """Observe arterial pressure with block masking plus occasional negative artifacts.

        Real ART_MBP: missingness≈87.6%, jitter_dt≈0.184s, dt_median=2.0s.
        Block masking (block_frac≈50%) with low dropout (0.9%) reproduces these:
          missingness = 1 - 0.50 × 0.25 × 0.991 ≈ 87.6%
          jitter_dt   = sqrt(0.009/0.991²) × 2 ≈ 0.191s (ratio ≈ 1.04×)
        Real ART_MBP has min median = -33 mmHg from arterial line flush artifacts.
        ~60% of cases have at least one period of negative pressure.
        """
        duration = t[-1] - t[0]
        block_frac = float(self.rng.uniform(0.45, 0.55))
        block_duration = block_frac * duration
        block_start = float(self.rng.uniform(0, max(0.0, duration - block_duration)))
        block_end = block_start + block_duration

        # AR(1) physiological noise (τ=60s, σ_stat≈11) to match real ART_MBP_std≈27.
        # (Tried raising this to 20 after the propofol MAP coefficient was
        # recalibrated to fix the mean gap: it improved std/kurt but pushed
        # ART_MBP_iqr z-score from 0.24 to 1.41 - a wash, not a net win. Real
        # ART_MBP's variance genuinely comes from a tight-core/rare-large-excursion
        # mixture rather than uniform continuous noise, which would need a more
        # invasive regime-switching model to fix properly - left as a known,
        # lower-priority residual rather than forced with a single parameter.)
        # Kept narrower than std alone would suggest: real ART_MBP_iqr (≈16) is much
        # smaller than what a Gaussian with std=27 would produce, meaning most of the
        # variance comes from rare large excursions (autoregulation keeps the body of
        # the distribution tight) rather than continuous noise. Occasional flush/dip
        # artifacts below supply the extra std/kurtosis without widening the IQR.
        from scipy.signal import lfilter as _lfilter_art
        _phi_art = float(np.exp(-2.0 / 60.0))
        _sigma_art = float(np.sqrt(1.0 - _phi_art ** 2) * 14.5)
        _innov_art = self.rng.normal(0.0, _sigma_art, len(latent)).astype(latent.dtype)
        _ar_art = _lfilter_art([1.0], [1.0, -_phi_art], _innov_art).astype(latent.dtype)
        # Small case-level offset: invasive arterial catheters commonly read a few
        # mmHg higher than cuff/model MAP due to transducer height and peripheral
        # waveform amplification. Kept modest and clinically defensible (a large
        # fabricated offset would just paper over a baseline miscalibration in the
        # hemodynamic model itself — map_baseline in CaseSimulator.__init__ was
        # raised instead to fix that gap at the source).
        _case_offset_art = float(self.rng.normal(6.0, 5.0))
        latent_noisy = latent + _ar_art + _case_offset_art

        # Case-level heavy-tailed gap rate: most cases have none (regular cadence),
        # a minority have occasional multi-second gaps (line flush, motion artifact),
        # matching real ART_MBP dt_std mean=1.08±1.81 (std > mean ⇒ heavy tail).
        if self.rng.random() < 0.70:
            gap_rate_art, gap_dur_art = 0.0, 0.0
        else:
            gap_rate_art = float(self.rng.uniform(0.002, 0.02))
            gap_dur_art = float(self.rng.uniform(10.0, 60.0))

        observed = self.sensor.observe(
            t, self._ema(latent_noisy, tau=10.0), sampling_interval=2.0, noise_sd=4.0,
            dropout_rate=0.009, gap_rate=gap_rate_art, gap_duration_s=gap_dur_art,
            value_range=value_range,
        )
        observed[(t < block_start) | (t > block_end)] = np.nan

        # Occasional negative-pressure artifact (arterial line flush/zeroing) in ~85%
        # of cases; deepened range (real ART_MBP_min≈-26±47) and higher probability
        # also balance the positive skew introduced by the high-pressure flush below.
        if self.rng.random() < 0.85:
            art_start = float(self.rng.uniform(block_start, max(block_start, block_end - 60.0)))
            art_dur = float(self.rng.exponential(20.0))
            art_mask = (t >= art_start) & (t <= art_start + art_dur) & (~np.isnan(observed))
            if art_mask.sum() > 0:
                observed[art_mask] = float(-self.rng.uniform(20, 75))

        # High-pressure flush artifact (arterial line calibration/zeroing against
        # the pressurised flush bag): real ART_MBP_max≈251±80, present in ~80% of
        # cases. Brief spike to 180-340 mmHg, a few seconds long.
        if self.rng.random() < 0.80:
            n_flush = 1 + self.rng.poisson(0.6)
            for _ in range(n_flush):
                flush_start = float(self.rng.uniform(block_start, max(block_start, block_end - 20.0)))
                flush_dur = float(self.rng.exponential(6.0) + 2.0)
                flush_mask = (t >= flush_start) & (t <= flush_start + flush_dur) & (~np.isnan(observed))
                if flush_mask.sum() > 0:
                    observed[flush_mask] = float(self.rng.uniform(180.0, 340.0))
        return observed

    def _observe_bis(self, t: np.ndarray, bis_values: np.ndarray) -> np.ndarray:
        """Observe BIS with block masking and brief post-induction burst suppression.

        Real BIS: min=0 in 100% of cases (5-30s burst suppression at induction),
        roughness≈1.61 (from noise_sd=1.1), jitter≈0, missingness=72.1%.
        BIS_std real ≈ 8-10. The PK/PD latent BIS is too smooth (std≈2-4) because
        it only reflects drug concentrations. We add an AR(1) stochastic process
        (τ=120s, σ=8) to simulate physiological depth fluctuations and responses
        to surgical stimuli, then apply a minimal EMA (τ=3s) to remove 0.5s grid
        noise only.
        """
        from scipy.signal import lfilter

        duration = t[-1] - t[0]
        block_frac = float(self.rng.uniform(0.45, 0.65))
        block_duration = block_frac * duration
        # Block starts early (exponential bias toward t=0) to capture pre-drug BIS≈93.
        # Real BIS monitors are applied at induction start; exponential scale=2% of duration
        # gives P(block includes first 10 min) ≈75%, matching real BIS_max≈92.
        raw_start = float(self.rng.exponential(duration * 0.02))
        block_start = min(raw_start, max(0.0, duration - block_duration))
        block_end = block_start + block_duration
        burst_t0 = block_start  # burst suppression at block start (at/near induction)

        # AR(1) stochastic variability: τ=120s, steady-state σ=8 (matching real BIS_std≈8-10)
        phi = float(np.exp(-0.5 / 120.0))
        sigma_innov = 8.0 * float(np.sqrt(1.0 - phi ** 2))
        innov = self.rng.normal(0.0, sigma_innov, len(bis_values)).astype(bis_values.dtype)
        ar_noise = lfilter([1.0], [1.0, -phi], innov).astype(bis_values.dtype)
        bis_with_variability = np.clip(bis_values + ar_noise, 5.0, 95.0)

        # Occasional deep-anesthesia dips (BIS drops to 10-25 for 30-100s): real BIS_skew
        # is slightly negative (-0.26), meaning most mass sits at moderate/high levels
        # with a left tail from deeper-than-usual episodes (rather than the symmetric
        # AR noise alone, which yields a right-skewed distribution due to the high
        # pre-drug plateau).
        n_dips = int(self.rng.poisson(2.5))
        for _ in range(n_dips):
            dip_start = float(self.rng.uniform(block_start, max(block_start, block_end - 100.0)))
            dip_dur = float(self.rng.uniform(30.0, 100.0))
            dip_target = float(self.rng.uniform(10.0, 25.0))
            dip_mask = (t >= dip_start) & (t <= dip_start + dip_dur)
            if dip_mask.sum() > 0:
                bis_with_variability[dip_mask] = np.minimum(
                    bis_with_variability[dip_mask], dip_target
                )

        # Minimal EMA to remove 0.5s grid artefacts only (τ=3s keeps the AR variability)
        ema_bis = self._ema(bis_with_variability, tau=3.0)
        # Case-level heavy-tailed gap rate matching real BIS dt_std mean=0.40±2.39
        # (std ≫ mean ⇒ most cases near-zero, a minority with large occasional gaps).
        if self.rng.random() < 0.85:
            gap_rate_bis, gap_dur_bis = 0.0, 5.0
        else:
            gap_rate_bis = float(self.rng.uniform(0.001, 0.01))
            gap_dur_bis = float(self.rng.uniform(10.0, 90.0))
        observed = self.sensor.observe(
            t, ema_bis, sampling_interval=1.0, noise_sd=1.1, dropout_rate=0.0001,
            gap_rate=gap_rate_bis, gap_duration_s=gap_dur_bis, value_range=(0, 100),
        )
        # Restrict BIS to the block window
        observed[(t < block_start) | (t > block_end)] = np.nan

        # Post-induction burst suppression in OBSERVED values: real burst suppression
        # after an induction bolus commonly lasts 5-30s; extended moderately (up to
        # 40s) rather than 75s to stay within clinically documented durations while
        # still trimming some of the pre-drug-to-maintenance ramp that otherwise
        # biases BIS_skew positive.
        burst_dur = float(self.rng.uniform(10.0, 40.0))
        burst_mask = (t >= burst_t0) & (t <= burst_t0 + burst_dur) & (~np.isnan(observed))
        if burst_mask.sum() > 0:
            observed[burst_mask] = 0.0  # BIS=0 exactly, matching real VitalDB burst suppression

        return np.clip(observed, 0, 100)

    def _observe_drug_rate(
        self,
        t: np.ndarray,
        rate_arr: np.ndarray,
        dropout_rate: float,
        value_range: tuple,
        timing_jitter_frac: float = 0.0,
        unit_factor: float = 3.0,
        burst_range: tuple = (500.0, 1500.0),
        noise_log_mean: float = 5.0,
    ) -> np.ndarray:
        """Observe drug infusion rate without block masking.

        Real Orchestra rates: dt_median≈1.0s, PPF20 jitter=0.684s, RFTN20 jitter=0.438s.
        Fix: hold=False + timing_jitter_frac adds ±frac random variation to each interval.
          PPF20: timing_jitter_frac=0.8 → dt_std≈0.46s (ratio≈0.67× real 0.684s).
          RFTN20: timing_jitter_frac=0.7 → dt_std≈0.40s (ratio≈0.92× real 0.438s).
        No block masking: step-function signals have identical value distribution
        whether hold=True or hold=False, covering the entire case avoids distortion.

        unit_factor: multiply rate_arr to convert to mL/h for Orchestra display.
          PPF20 (propofol 20 mg/mL): factor = 60/20 = 3.0
          RFTN20 (remifentanil ~60 mcg/mL in VitalDB): factor ≈ 1.0
        """
        from scipy.signal import lfilter
        # Convert drug units to mL/h for Orchestra display.
        # unit_factor depends on drug concentration: PPF20 (20 mg/mL) → 3.0, RFTN20 (~60 mcg/mL) → 1.0
        rate_mLh = rate_arr * unit_factor
        # TCI induction burst: pump runs at max rate for 10-60s to front-load drug.
        # Short duration (few samples) → extreme skew/kurtosis matching real VitalDB.
        # burst_range differs per drug to match real per-drug max distributions.
        # NOTE: no per-sample AR(1) noise here — real TCI pump rates are piecewise-constant
        # (clinician sets a rate; it holds until adjusted), so adding continuous per-sample
        # noise would make frac_changes >> 0.25 and fail the B2 stepwise check.
        nonzero_mask = rate_mLh > 0
        if nonzero_mask.any():
            # TCI induction burst: pump runs at max rate for 10-60s to front-load drug.
            # Short duration (few samples) → extreme skew/kurtosis matching real VitalDB.
            # burst_range differs per drug to match real per-drug max distributions.
            first_nonzero = int(np.where(nonzero_mask)[0][0])
            burst_steps = int(self.rng.uniform(10.0, 60.0) / 0.5)  # 10-60s at 0.5s grid
            burst_rate = float(self.rng.uniform(*burst_range))  # mL/h loading rate
            burst_end = min(first_nonzero + burst_steps, len(rate_mLh))
            rate_mLh[first_nonzero:burst_end] = np.maximum(
                rate_mLh[first_nonzero:burst_end], burst_rate
            )
        return self.sensor.observe(
            t, rate_mLh,
            sampling_interval=1.0, noise_sd=0.0,
            dropout_rate=dropout_rate, gap_rate=0.0, gap_duration_s=0.0,
            value_range=value_range, hold=False,
            timing_jitter_frac=timing_jitter_frac,
        )

    def _observe_hr(self, t: np.ndarray, hr_values: np.ndarray) -> np.ndarray:
        """Observe HR with realistic asymmetric noise and occasional arrhythmia events.

        Real HR is recorded as a single continuous block with only small internal gaps,
        which produces a very low dt_std. We mimic this by masking the entire signal
        down to a contiguous observation window covering ~20% of the case duration, and
        then add tachycardia spikes inside the block so they contribute to the max.
        """
        # Add slow physiological HR drift (σ_stat=7.5 bpm, τ=300s) to match real HR_std≈8.6 bpm.
        # PK/PD latent HR is too stable (~1 bpm std); this AR(1) component provides realistic
        # long-term variation (autonomic, arousal, depth changes) while EMA τ=30s smooths
        # the short-term fluctuations.
        from scipy.signal import lfilter as _lfilter
        _alpha_hr = float(np.exp(-0.5 / 300.0))
        _sigma_hr = float(np.sqrt(1.0 - _alpha_hr ** 2) * 9.5)
        _hr_drift = _lfilter([1.0], [1.0, -_alpha_hr],
                             self.rng.standard_normal(len(t)) * _sigma_hr).astype(hr_values.dtype)
        latent_hr = hr_values + _hr_drift
        # Case-level heavy-tailed gap rate matching real HR dt_std mean=0.78±1.59 (std>mean).
        if self.rng.random() < 0.75:
            gap_rate_hr, gap_dur_hr = 0.001, 5.0
        else:
            gap_rate_hr = float(self.rng.uniform(0.002, 0.008))
            gap_dur_hr = float(self.rng.uniform(10.0, 40.0))
        observed = self.sensor.observe(t, self._ema(latent_hr, tau=30.0), sampling_interval=2.0, noise_sd=1.0, dropout_rate=0.02, gap_rate=gap_rate_hr, gap_duration_s=gap_dur_hr, value_range=(0, 300))
        # Shift observed HR up toward real population mean
        observed = np.where(np.isnan(observed), np.nan, observed + 10.0)
        # Restrict HR to a single contiguous block (realistic monitor recording pattern).
        # Block covers ~46-62 % of case duration to match real HR missingness ≈ 86 %.
        duration = t[-1] - t[0]
        block_duration = self.rng.uniform(0.46, 0.62) * duration
        block_start = self.rng.uniform(0, max(0, duration - block_duration))
        block_end = block_start + block_duration
        observed[(t < block_start) | (t > block_end)] = np.nan
        # Add occasional asymmetric spikes inside the block to widen IQR and create positive skew.
        # Case-level bradycardia severity mixture: a minority of cases have a genuine
        # vasovagal/vagal reflex episode (e.g. laryngoscopy, peritoneal traction) that
        # transiently drops HR well below the population baseline; most cases only show
        # the mild dips typical of opioid-related sinus slowing. Ranges kept within
        # clinically documented intraoperative bradycardia (rarely below ~30 bpm without
        # arrest), rather than tuned purely to widen inter-case variance.
        if self.rng.random() < 0.15:
            brady_range = (12.0, 28.0)
        else:
            brady_range = (5.0, 15.0)
        n_spikes = self.rng.poisson(2.0)
        for _ in range(n_spikes):
            start = self.rng.uniform(block_start, max(block_start + 1, block_end - 60))
            duration = self.rng.exponential(60)
            # 80% tachycardia, 20% bradycardia dips
            if self.rng.random() < 0.8:
                delta = self.rng.uniform(40, 90)
            else:
                delta = -self.rng.uniform(*brady_range)
            mask = (t >= start) & (t <= min(start + duration, block_end))
            observed[mask] += delta * np.exp(-(t[mask] - start) / duration * 3)
        # Rare zero/artifact samples to mimic real monitor behavior
        artifact_mask = self.rng.random(len(observed)) < 0.00002
        observed[artifact_mask & (~np.isnan(observed))] = 0.0
        # Floor of 30 allows genuine severe vagal bradycardia episodes through while
        # keeping typical cases well above it; 50 was too rigid (erased all inter-case
        # variance) and 20 over-corrected into clinically implausible territory.
        return np.clip(observed, 30, 300)

    def _observe_mac(self, t: np.ndarray, mac_values: np.ndarray) -> np.ndarray:
        """Observe MAC with a sparse block and realistic dt_std.

        Real MAC is recorded in a sparse, irregular block; we restrict the signal
        to a contiguous window and sample with jittered cadence to match the real
        dt_std and missingness.
        """
        # Case-level dropout mixture: most cases sample fairly regularly, a minority
        # have highly irregular cadence (heavy tail), matching real MAC dt_std
        # mean=4.31±4.97 (std ≈ mean ⇒ lognormal-like spread across cases).
        if self.rng.random() < 0.7:
            dropout_rate_mac = float(self.rng.uniform(0.05, 0.20))
        else:
            dropout_rate_mac = float(self.rng.uniform(0.22, 0.40))
        observed = self.sensor.observe(
            t,
            mac_values,
            sampling_interval=self.rng.uniform(6.0, 8.0),
            noise_sd=0.02,
            dropout_rate=dropout_rate_mac,
            gap_rate=0.0,
            gap_duration_s=5.0,
            value_range=(0.0, 2.5),
        )
        duration = t[-1] - t[0]
        block_duration = self.rng.uniform(0.45, 0.60) * duration
        # Anchor the observed block around the sevoflurane administration window
        # (when present) instead of a fully random position. Real MAC has huge
        # inter-case variance (max std≈1.36 vs mean≈1.0) driven by a mix of TIVA
        # cases (MAC≈0 throughout) and volatile-agent cases (MAC 0.6-1.5+); a
        # randomly-placed block can miss the sevoflurane window entirely and
        # collapse that variance toward 0, which is what was happening before.
        nonzero_idx = np.where(mac_values > 0)[0]
        if nonzero_idx.size > 0:
            active_start = float(t[nonzero_idx[0]])
            active_end = float(t[nonzero_idx[-1]])
            active_span = max(active_end - active_start, 1.0)
            # Block must be at least as long as the active span (plus margin) so
            # the whole sevoflurane window fits; extend block_duration if needed.
            block_duration = max(block_duration, min(duration, active_span * 1.15))
            lo = max(0.0, active_end - block_duration)
            hi = min(active_start, max(0.0, duration - block_duration))
            if hi < lo:
                lo, hi = hi, lo
            block_start = float(self.rng.uniform(lo, hi)) if hi > lo else lo
        else:
            block_start = self.rng.uniform(0, max(0, duration - block_duration))
        block_end = block_start + block_duration
        observed[(t < block_start) | (t > block_end)] = np.nan
        return observed

    def _generate_spo2(self, t: np.ndarray) -> np.ndarray:
        """Return a realistic SpO2 trajectory with smooth physiological variability.

        Uses an AR(1) process (τ ≈ 30 s) around a case-specific baseline of
        96–100 % so that every case has IQR > 0 and slow-moving fluctuations
        matching real VitalDB recordings.  Occasional desaturation episodes are
        superimposed to reproduce the observed spread in min and skewness.
        """
        from scipy.signal import lfilter

        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 0.5

        # AR(1) time constant: SpO2 changes on a ~30 s physiological scale
        tau = 30.0
        alpha = float(np.exp(-dt / tau))
        # Three-category case mix matching real integer-valued pulse oximeter:
        #   55 % stable   → all values round to 99         → IQR = 0
        #   30 % mild     → straddle 99/100                → IQR = 1
        #   15 % variable → span 97-100 (σ≈1.5)           → IQR = 2
        # Expected IQR = 0.55×0 + 0.30×1 + 0.15×2 = 0.60 ≈ real 0.59
        # Expected std  ≈ 0.73                             ≈ real 0.79
        r = self.rng.random()
        if r < 0.55:
            baseline = float(self.rng.uniform(98.7, 99.1))   # always rounds to 99 → IQR=0
            sigma_stat = 0.10
        elif r < 0.85:
            baseline = float(self.rng.uniform(99.2, 99.8))   # straddles 99/100 → IQR=1
            sigma_stat = 0.50
        else:
            baseline = float(self.rng.uniform(97.5, 99.5))   # spans 97-100 → IQR=2
            sigma_stat = 1.50
        sigma_innov = float(np.sqrt(1.0 - alpha ** 2) * sigma_stat)

        # Vectorised AR(1) via scipy lfilter — O(n) without a Python loop.
        # innovations include the mean-reversion term: (1-alpha)*baseline + noise
        innov = self.rng.normal((1.0 - alpha) * baseline, sigma_innov, size=n)
        zi = np.array([float(self.rng.normal(baseline, 0.5))])
        base, _ = lfilter([1.0], [1.0, -alpha], innov, zi=zi)

        # Rare desaturation episodes (≈12 % of cases) with smooth envelope
        if self.rng.random() < 0.12:
            n_events = 1 + self.rng.poisson(0.8)
            for _ in range(n_events):
                t_start = self.rng.uniform(t[0] + 120.0, max(t[0] + 121.0, t[-1] - 120.0))
                dur = self.rng.exponential(90.0) + 30.0
                depth = self.rng.uniform(8.0, 25.0)
                idx = (t >= t_start) & (t <= t_start + dur)
                if idx.sum() > 0:
                    envelope = np.sin(np.pi * (t[idx] - t_start) / dur)
                    base[idx] -= depth * envelope

        return np.clip(base, 60.0, 100.0)

    def _generate_etco2(self, t: np.ndarray) -> np.ndarray:
        """Return a realistic EtCO2 trajectory with ventilation-related variability.

        Real EtCO2 has a negative skew, leptokurtic distribution, std≈6 mmHg,
        and min=0 in ~75% of cases (pre/post-intubation or equipment disconnection).
        """
        from scipy.signal import lfilter
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 0.5
        # Per-case baseline offset
        base = float(self.rng.normal(33.0, 2.5))
        signal = np.full(n, base)
        # Slow physiological drift: AR(1) with τ=300s, case-varying σ_stat
        # σ_stat ~ lognormal(log(4.5), 0.6) → E[σ]=5.4, matching real ETCO2_std≈6.46±2.95
        tau_drift = 300.0
        alpha_d = float(np.exp(-dt / tau_drift))
        sigma_stat = float(self.rng.lognormal(np.log(4.5), 0.6))
        sigma_d = float(np.sqrt(1.0 - alpha_d ** 2) * sigma_stat)
        innov_d = self.rng.normal(0.0, sigma_d, size=n)
        zi_d = np.array([float(self.rng.normal(0.0, sigma_stat * 0.4))])
        drift, _ = lfilter([1.0], [1.0, -alpha_d], innov_d, zi=zi_d)
        signal = signal + drift
        # Occasional moderate hypocapnia events (-5 to -15 mmHg, duration ~60s)
        n_events = int(self.rng.poisson(2))
        for _ in range(n_events):
            start = self.rng.uniform(300, max(600, t[-1] - 300))
            duration = float(self.rng.exponential(60))
            delta = float(self.rng.uniform(-15, -5))
            mask = (t >= start) & (t <= start + duration)
            signal[mask] += delta * np.exp(-(t[mask] - start) / max(duration, 1.0) * 3)
        # Pre-intubation zero period with SMOOTH RAMP-UP after intubation.
        # P=0.85 gives ETCO2_min=0 in most cases matching real VitalDB.
        if self.rng.random() < 0.85:
            intub_t = float(self.rng.uniform(10.0, 90.0))
            ramp_dur = 45.0  # ramp from 0 to full signal after intubation
            pre_mask = t < intub_t
            ramp_mask = (t >= intub_t) & (t < intub_t + ramp_dur)
            signal[pre_mask] = 0.0
            if ramp_mask.sum() > 0:
                ramp_frac = (t[ramp_mask] - intub_t) / ramp_dur
                signal[ramp_mask] *= ramp_frac
        # End-of-case disconnection (extubation): circuit disconnects from capnograph
        # before/at emergence, producing a second zero-mass region. Real ETCO2 skew is
        # strongly negative (-3.8) and leptokurtic (kurt≈23), driven by these rare but
        # large excursions to 0 at both ends of the case rather than mid-case noise.
        if self.rng.random() < 0.80:
            extub_dur = float(self.rng.uniform(30.0, 120.0))
            extub_t = max(0.0, t[-1] - extub_dur)
            down_ramp = 20.0
            post_mask = t >= extub_t + down_ramp
            down_mask = (t >= extub_t) & (t < extub_t + down_ramp)
            signal[post_mask] = 0.0
            if down_mask.sum() > 0:
                down_frac = 1.0 - (t[down_mask] - extub_t) / down_ramp
                signal[down_mask] *= down_frac
        # Occasional brief circuit disconnections mid-case (abrupt drop to 0, 5-20s):
        # rare, sharp excursions that inflate left-skew and kurtosis without adding
        # much to overall IQR (short duration relative to case length).
        n_disconnects = int(self.rng.poisson(0.8))
        for _ in range(n_disconnects):
            disc_start = float(self.rng.uniform(300.0, max(600.0, t[-1] - 300.0)))
            disc_dur = float(self.rng.uniform(5.0, 20.0))
            disc_mask = (t >= disc_start) & (t <= disc_start + disc_dur)
            signal[disc_mask] = 0.0
        return np.clip(signal, 0.0, 60.0)

