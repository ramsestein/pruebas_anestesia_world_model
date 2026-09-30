"""Orchestrate one full synthetic case  v2 with causal identifiability.

Key improvements over v1:
  - Exogenous perturbations of ALL actions (propofol, remi, vasopressors)
    independent of physiological state.
  - Explicit a_null segments: everything frozen, physiology drifts alone.
  - Nociception with separated pathways (MAP/HR strong, BIS weak+noisy).
  - Remifentanil valle nociceptivo: bolus decay + infusion ramp gap.
  - Randomised PK model per case (Marsh / Schnider / Eleveld).
  - Sevoflurane as Emax on hypnotic axis (not decorrelated noise).
  - Expanded dose coverage matching real VitalDB ranges.
  - Counterfactual evaluation: same-seed bifurcated trajectories.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from anessim.actions import Action, ActionType
from anessim.config import SimulatorConfig
from anessim.nociception import (
    NociceptionDriver,
    model_remi_valle,
)
from anessim.patients import Patient, PatientSampler
from anessim.pd import C50_PROP, C50_REM, C50_SEVO, HemodynamicResponse, bis_from_ce
from anessim.pk import (
    Infusion,
    RemifentanilMinto,
    randomise_propofol_model,
)
from anessim.randomization import Randomization
from anessim.respiratory import (
    RespiratoryModel,
    generate_ventilation_profile,
    vco2_reference,
)
from anessim.sensors import ObservationModel
from anessim.timeline import TimelineGenerator
from anessim.track_presence import TrackPresenceSampler


BIS_AROUSAL_MAX = 12.0
BIS_AROUSAL_NOISE_SD = 3.0

# ── A6: RNG streams separados por componente ──────────────────────────────
# Cada componente consume su propio generador, derivado de la semilla del caso
# más un offset fijo. Así una palanca contrafactual (que solo altera el stream
# de "acciones") no desplaza el consumo aleatorio de paciente, timeline, PK/PD,
# presencia ni sensor: el prefijo pre-split de ambas ramas es idéntico bit a bit.
RNG_OFFSET_PATIENT = 1_000_000
RNG_OFFSET_TIMELINE = 2_000_000
RNG_OFFSET_ACTIONS = 3_000_000
RNG_OFFSET_PKPD = 4_000_000
RNG_OFFSET_NOCICEPTION = 5_000_000
RNG_OFFSET_PRESENCE = 6_000_000
RNG_OFFSET_SENSOR = 7_000_000

# ── v3: Vasopressor exogenous fraction ────────────────────────────────────
# Fraction of vasopressor administrations that come from the exogenous
# process (independent of MAP) vs the reactive loop (MAP<70).
# Tune to achieve NORA↔MAP ~0.45-0.55.
# v3.1: reduced from 0.55 to 0.15 — most nora should be reactive.
VASO_EXOGENOUS_FRACTION = 0.15

# Tracks that encode actions/control signals. These must never be dropped by
# track-presence sampling: they are the levers the world model is conditioned on.
CONTROL_TRACKS = {
    "ppf_bolus_mg", "remi_bolus_ug", "roc_bolus_mg", "phen_bolus_mcg", "eph_bolus_mg",
    "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE",
    "Orchestra/PHEN_RATE", "Orchestra/NEPI_RATE",
    "Primus/SET_FIO2", "Primus/SET_RR_IPPV", "Primus/SET_TV_L", "Primus/SET_PIP",
    "Primus/SET_INTER_PEEP", "Primus/SET_MAC",
}

# ── v6 (M1): BIS/EMG recalibrado. Nivel real ~27.35±3.69, asimétrico a la
# derecha (p50 26.5, p95 33, p99 46), con acoplamiento positivo a BIS. ──
_EMG_SHIFT = 24.7
_EMG_GAMMA_K = 1.8
_EMG_GAMMA_TH = 1.05
_EMG_BIS_SLOPE = 0.04
_EMG_TAIL_PROB = 0.05
_EMG_TAIL_MU = 2.45
_EMG_TAIL_SIGMA = 0.65
_EMG_NOISE_SD = 0.8


def emg_latent(bis, rng: np.random.Generator) -> np.ndarray:
    """EMG latente: núcleo gamma (cola derecha) + acoplamiento a BIS + cola
    lognormal poco frecuente (episodios de actividad/artefacto)."""
    bis = np.asarray(bis, dtype=float)
    central = (
        _EMG_SHIFT
        + rng.gamma(_EMG_GAMMA_K, _EMG_GAMMA_TH, bis.shape)
        + _EMG_BIS_SLOPE * (bis - 40.0)
    )
    tail = rng.random(bis.shape) < _EMG_TAIL_PROB
    extra = np.where(tail, rng.lognormal(_EMG_TAIL_MU, _EMG_TAIL_SIGMA, bis.shape), 0.0)
    return np.clip(central + extra, 0.0, 100.0)


# ── v6 (M6): rejilla de cuantización de la capa de medida. El monitor real
# emite enteros (HR/SpO2/ETCO2/PEEP/PIP/TV/RR/ART_*), paso 0.1 (BIS/BIS, MV,
# BT) o 0.01 (BIS/EMG). El modelo fisiológico sigue en continuo; el redondeo
# se aplica SOLO al emitir el valor observado. ──
QUANT_GRID: dict[str, float] = {
    "Solar8000/HR": 1.0, "Solar8000/PLETH_HR": 1.0,
    "Solar8000/PLETH_SPO2": 1.0, "Solar8000/ETCO2": 1.0,
    "Solar8000/RR_CO2": 1.0, "Solar8000/RR": 1.0,
    "Solar8000/BT": 0.1,
    "Solar8000/ART_MBP": 1.0, "Solar8000/ART_SBP": 1.0,
    "Solar8000/ART_DBP": 1.0, "Solar8000/NIBP_SBP": 1.0,
    "Solar8000/NIBP_MBP": 1.0, "Solar8000/NIBP_DBP": 1.0,
    "Solar8000/VENT_RR": 1.0, "Solar8000/VENT_PIP": 1.0,
    "Solar8000/VENT_PPLAT": 1.0, "Solar8000/VENT_TV": 1.0,
    "Solar8000/VENT_SET_TV": 1.0, "Solar8000/VENT_MV": 0.1,
    "Solar8000/VENT_MAWP": 1.0, "Solar8000/VENT_INSP_TM": 0.1,
    "Solar8000/VENT_SET_PCP": 1.0, "Solar8000/FIO2": 1.0,
    "Solar8000/FEO2": 1.0,
    "BIS/BIS": 0.1, "BIS/EMG": 0.01, "BIS/SQI": 1.0,
    "BIS/SEF": 0.1, "BIS/SR": 0.1, "BIS/TOTPOW": 1.0,
    "Primus/ETCO2": 1.0, "Primus/PEEP_MBAR": 1.0, "Primus/PIP_MBAR": 1.0,
    "Primus/PPLAT_MBAR": 1.0, "Primus/MV": 0.1, "Primus/TV": 1.0,
    "Primus/RR_CO2": 1.0, "Primus/FIO2": 1.0, "Primus/FEO2": 1.0,
    "Primus/SET_FIO2": 1.0, "Primus/SET_RR_IPPV": 1.0,
    "Primus/SET_TV_L": 0.01, "Primus/SET_PIP": 1.0,
    "Primus/SET_INTER_PEEP": 1.0, "Primus/SET_INSP_TM": 0.1,
    "Primus/SET_INSP_PAUSE": 1.0, "Primus/SET_FRESH_FLOW": 0.1,
    "Primus/MAWP_MBAR": 1.0, "Primus/COMPLIANCE": 1.0,
    "Primus/VENT_LEAK": 1.0,
}


class CaseSimulator:
    """Simulate one anesthesia case from patient to tracks  v2."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        base = self.config.random_seed
        # A6: per-component RNG streams derived from the case seed.
        self.rng = np.random.default_rng(base)
        self.rng_patient = np.random.default_rng(base + RNG_OFFSET_PATIENT)
        self.rng_timeline = np.random.default_rng(base + RNG_OFFSET_TIMELINE)
        self.rng_actions = np.random.default_rng(base + RNG_OFFSET_ACTIONS)
        self.rng_pkpd = np.random.default_rng(base + RNG_OFFSET_PKPD)
        self.rng_nociception = np.random.default_rng(base + RNG_OFFSET_NOCICEPTION)
        self.rng_presence = np.random.default_rng(base + RNG_OFFSET_PRESENCE)
        self.rng_sensor = np.random.default_rng(base + RNG_OFFSET_SENSOR)

        self.sampler = PatientSampler(self.config.clinical_path)
        self.timeline_gen = TimelineGenerator(self.rng_timeline)
        self.sensor = ObservationModel(self.rng_sensor)
        self.track_presence = TrackPresenceSampler(
            self.config.cases_dir or "dataset/cases", self.rng_presence,
        )
        self.randomization = Randomization(self.rng_actions, self.config.policy_fraction)
        self.noci_driver = NociceptionDriver(self.rng_nociception)

    # ==================================================================
    # Main entry point
    # ==================================================================
    def run(
        self,
        caseid: int,
        duration_min: float | None = None,
        presence_override: list[str] | None = None,
        counterfactual_split_t: float | None = None,
        counterfactual_actions_override: list[Action] | None = None,
        counterfactual_vent_override: dict[str, float] | None = None,
        physiology_override: dict[str, float] | None = None,
    ) -> dict:
        """Run a full synthetic case.

        Args:
            caseid: unique case identifier.
            duration_min: force case duration (None = random).
            presence_override: force track presence for cohort coordination.
            counterfactual_split_t: if set, this is the SECOND branch of a
                counterfactual pair. Actions before this time are identical
                to the first branch; after this time they diverge.
            counterfactual_actions_override: for the second branch, the
                actions to use after split_t.
            counterfactual_vent_override: post-split additive deltas for
                ventilation settings, e.g. {"peep_delta": 5.0}. Supported
                keys: fio2_delta, peep_delta, rr_delta, tv_delta.
            physiology_override: per-case physiological parameters applied to
                BOTH branches (identical patient). Supported keys:
                "fio2_baseline" (lower FiO2 so PEEP/FiO2 have headroom),
                "shunt_fraction" (higher shunt so PEEP recruitment matters).

        Returns:
            dict with patient, timeline, tracks, truth, clinical_row, actions,
            and metadata (pk_model, iiv, nociception_info).
        """
        patient = self.sampler.sample(caseid, rng=self.rng_patient)
        timeline = self.timeline_gen.generate(duration_min=duration_min)
        dt = self.config.dt_seconds
        t = np.arange(0.0, timeline.total_duration_s + dt, dt)

        # -- Baseline MAP / HR --
        map_base, hr_base = self.randomization.randomize_baselines(
            81.0, 71.0, cv=0.20,  # F2 recalib r3: MAP limpio 82.6 / HR 73
        )

        # -- Inter-patient PD variability — v3.1: wider range for more hypotensive cases
        _pd_base = float(self.rng_pkpd.lognormal(0.0, 0.25))
        _age_shift = patient.age - 50.0
        _asa_shift = max(0, (patient.asa or 2) - 2)
        prop_sensitivity = float(np.clip(
            _pd_base * (1.0 + 0.010 * _age_shift) * (1.0 + 0.08 * _asa_shift),
            0.4, 2.5,   # v3.1: lower min, higher max
        ))
        vasopressor_response = float(np.clip(
            (1.0 / _pd_base) * (1.0 - 0.006 * _age_shift) * (1.0 - 0.05 * _asa_shift),
            0.4, 1.6,
        ))
        self.hr_model = HemodynamicResponse(
            map_baseline=map_base, hr_baseline=hr_base,
            prop_sensitivity=prop_sensitivity,
            vasopressor_response=vasopressor_response,
        )

        # -- Build actions --
        actions = self._build_actions(timeline, patient)

        # Counterfactual branch override
        is_cf = counterfactual_split_t is not None
        if is_cf and counterfactual_actions_override is not None:
            # A5: la palanca se AÑADE a las acciones base (que son idénticas en
            # ambas ramas, mismo seed). Antes se REEMPLAZABAN las acciones
            # post-split, dejando a la rama control sin ajustes de mantenimiento
            # ni parada de emergencia -> fisiología plana post-split.
            override = [
                Action(t_s=a.t_s, action_type=a.action_type, drug=a.drug,
                       value=a.value, unit=a.unit, route=a.route,
                       metadata=a.metadata)
                for a in (counterfactual_actions_override or [])
                if a.t_s > counterfactual_split_t
            ]
            actions = sorted(actions + override, key=lambda a: a.t_s)

        # -- Build infusion schedules and bolus lists --
        ppf_inf = self._actions_to_infusions(actions, "propofol")
        remi_inf = self._actions_to_infusions(actions, "remifentanil")
        ppf_boluses = self._extract_boluses(actions, "propofol")
        remi_boluses = self._extract_boluses(actions, "remifentanil")
        roc_boluses = [a for a in actions if a.drug == "rocuronium"]
        sevo_changes = [a for a in actions if a.drug == "sevoflurane"]

        # -- Exogenous perturbations (INDEPENDENT of state) --
        perturb = self._build_exogenous_perturbations(t, timeline)
        ppf_inf = self._apply_rate_perturbations(
            ppf_inf, perturb["ppf_mult"], t, "propofol",
        )
        remi_inf = self._apply_rate_perturbations(
            remi_inf, perturb["remi_mult"], t, "remifentanil",
        )

        # -- v3: Inject exogenous vasopressor actions BEFORE reactive loop --
        exo_vaso = perturb.get("exo_vaso_actions", [])
        if exo_vaso:
            actions = sorted(actions + exo_vaso, key=lambda a: a.t_s)

        # -- PK simulation with randomised model --
        prop_model, pk_name = randomise_propofol_model(
            weight_kg=patient.weight, age_y=patient.age,
            height_cm=patient.height, sex=patient.sex,
            rng=self.rng_pkpd,
        )
        remi_model = RemifentanilMinto(
            weight_kg=patient.weight, height_cm=patient.height,
            age_y=patient.age, sex=patient.sex,
        )
        self._apply_pk_iiv(prop_model)
        self._apply_pk_iiv(remi_model)

        prop_st = prop_model.simulate(ppf_inf, t, boluses=ppf_boluses)
        remi_st = remi_model.simulate(remi_inf, t, boluses=remi_boluses)

        cp_prop = prop_st[:, 0]; ce_prop = prop_st[:, 3]
        cp_rem = remi_st[:, 0]; ce_rem = remi_st[:, 3]

        # -- Nociceptive stimulus --
        stim, att_map, att_hr, noci_meta = self.noci_driver.generate(
            t, timeline,
            op_type=patient.optype, op_name=patient.opname,
            ce_remi=ce_rem,
            patient_age=patient.age, patient_asa=patient.asa,
        )
        noci_map = stim * att_map
        noci_hr = stim * att_hr
        # BIS arousal: weak + noisy (A4/A3: suficiente varianza para corr_bis >= 0.95).
        # El arousal de BIS está MENOS atenuado por remi que MAP/HR (vía distinta).
        from scipy.signal import lfilter as _lft
        _att_bis = 1.0 / (1.0 + (ce_rem / 8.0) ** 2)
        _phi_bn = float(np.exp(-dt / 30.0))
        _sigma_bn = BIS_AROUSAL_NOISE_SD * float(np.sqrt(1.0 - _phi_bn ** 2))
        _noise_bn = _lft([1.0], [1.0, -_phi_bn], self.rng_nociception.normal(0.0, _sigma_bn, len(t)))
        noci_bis = np.clip(stim * _att_bis * BIS_AROUSAL_MAX + _noise_bn, 0.0, 15.0)

        # -- Respiratory physiology (v3) --
        phys = physiology_override or {}
        # v7 (C4): la VCO2 (CV 0.12, literatura) y la CARGA RESPIRATORIA se
        # sortean UNA vez y se pasan al modelo respiratorio y al perfil de
        # ventilación, de modo que la RR compense la producción de CO2 y la
        # carga (corr(RR, EtCO2) ≈ +0.336).
        _vco2 = phys.get("vco2")
        if _vco2 is None:
            _vco2 = float(np.clip(
                vco2_reference(patient.weight)
                * self.rng_pkpd.lognormal(0.0, 0.12), 70.0, 450.0))
        _resp_load = phys.get("resp_load")
        if _resp_load is None:
            _resp_load = float(self.rng_pkpd.lognormal(0.0, 0.25))
        resp_model = RespiratoryModel(
            self.rng_pkpd, age=patient.age, weight_kg=patient.weight,
            shunt_fraction=phys.get("shunt_fraction"), vco2=_vco2,
            resp_load=_resp_load,
        )
        vent = generate_ventilation_profile(
            t, timeline, self.rng_actions, patient.weight,
            fio2_baseline=phys.get("fio2_baseline", 0.50),
            suppress_fio2_steps=phys.get("fio2_baseline") is not None,
            vco2=_vco2, resp_load=_resp_load,
        )
        fio2_arr = vent["fio2"]
        peep_arr = vent["peep"]
        rr_arr = vent["rr"]
        tv_arr = vent["tv"]

        # Counterfactual ventilation override (post-split additive deltas).
        if is_cf and counterfactual_vent_override:
            post = t > counterfactual_split_t
            if "fio2_set" in counterfactual_vent_override:
                fio2_arr = fio2_arr.copy()
                fio2_arr[post] = np.clip(
                    counterfactual_vent_override["fio2_set"], 0.15, 1.0,
                )
            elif "fio2_delta" in counterfactual_vent_override:
                fio2_arr = fio2_arr.copy()
                fio2_arr[post] = np.clip(
                    fio2_arr[post] + counterfactual_vent_override["fio2_delta"], 0.15, 1.0,
                )
            if "peep_delta" in counterfactual_vent_override:
                peep_arr = peep_arr.copy()
                peep_arr[post] = np.clip(
                    peep_arr[post] + counterfactual_vent_override["peep_delta"], 0.0, 25.0,
                )
            if "rr_delta" in counterfactual_vent_override:
                rr_arr = rr_arr.copy()
                rr_arr[post] = np.clip(
                    rr_arr[post] + counterfactual_vent_override["rr_delta"], 0.0, 70.0,
                )
            if "tv_delta" in counterfactual_vent_override:
                tv_arr = tv_arr.copy()
                tv_arr[post] = np.clip(
                    tv_arr[post] + counterfactual_vent_override["tv_delta"], 0.0, 1600.0,
                )

        # Compute respiratory signals (first pass: no desaturation hemodynamics yet)
        paco2_arr = np.zeros(len(t), dtype=np.float32)
        etco2_arr = np.zeros(len(t), dtype=np.float32)
        spo2_arr = np.zeros(len(t), dtype=np.float32)
        pao2_arr = np.zeros(len(t), dtype=np.float32)
        pip_arr = np.zeros(len(t), dtype=np.float32)
        peep_map_effect_arr = np.zeros(len(t), dtype=np.float32)

        for i in range(len(t)):
            _, paco2_arr[i] = resp_model.compute_etco2(tv_arr[i], rr_arr[i])
            pao2_arr[i] = resp_model.compute_pao2(fio2_arr[i], peep_arr[i], paco2_arr[i])
            spo2_arr[i], _, _ = resp_model.compute_pao2_to_spo2(pao2_arr[i], peep_arr[i])
            pip_arr[i] = resp_model.compute_pip(tv_arr[i], peep_arr[i])
            peep_map_effect_arr[i] = resp_model.compute_peep_map_effect(peep_arr[i])

        # A2: PaCO2 con constante de tiempo fisiológica (depósito corporal de CO2,
        # ~2-4 min, no segundos). Bajar MV → sube PaCO2/EtCO2 en minutos.
        # Se añade una deriva metabólica lenta (producción de CO2 variable) para
        # que el EtCO2 tenga variabilidad intracaso realista.
        _tau_co2 = float(self.rng_pkpd.uniform(90.0, 180.0))
        _a_co2 = float(np.exp(-dt / _tau_co2))
        paco2_filt = _lft([1.0 - _a_co2], [1.0, -_a_co2], paco2_arr)
        _a_drift = float(np.exp(-dt / 300.0))
        # v6 (M5): deriva metabólica σ 0.4 -> 0.10 (la anterior, con tau 300 s,
        # daba sd estacionaria ~6.9 mmHg y era la causa del sd 6.26 del EtCO2).
        _drift = _lft([1.0], [1.0, -_a_drift],
                      self.rng_pkpd.normal(0.0, 0.10, len(t)))
        etco2_arr = np.clip((paco2_filt + _drift) * 0.95, 5.0, 80.0).astype(np.float32)
        paco2_arr = (paco2_filt + _drift).astype(np.float32)

        # -- Sevoflurane MAC --
        mac_arr = np.array([self._sevoflurane_at(ti, sevo_changes) for ti in t])

        # -- PD outputs --
        bis_sens = float(self.rng_pkpd.lognormal(0.0, 0.22))
        sevo_sens = float(self.rng_pkpd.lognormal(0.0, 0.18))

        bis = np.array([
            bis_from_ce(p, r, sevo_mac=m * sevo_sens,
                        c50_prop=C50_PROP * bis_sens, c50_rem=C50_REM * bis_sens,
                        nociceptive_arousal=float(nb))
            for p, r, m, nb in zip(ce_prop, ce_rem, mac_arr, noci_bis)
        ])

        map_baseline = np.array([
            self.hr_model.map_from_state(
                ce_prop=p, ce_rem=r, sevo_mac=m,
                nociceptive_drive_map=float(nm), weight_kg=patient.weight,
            )
            for p, r, m, nm in zip(ce_prop, ce_rem, mac_arr, noci_map)
        ])

        # -- Vasoactive: compute nora rate directly from MAP_bl deficit (v3.1) --
        # This replaces the fragile reactive loop with a direct proportional controller.
        # nora_rate = clip((65 - MAP_bl) * 0.008, 0, 0.35) + noise
        # Only applies during active anesthesia (before emergence - CD).
        CD = 300.0
        stops = [a.t_s for a in actions if a.action_type == ActionType.INFUSION_STOP
                 and a.drug in ("propofol", "remifentanil")]
        em_s = min(stops) if stops else t[-1]
        active = t <= (em_s - CD)

        # -- Vasoactive controller --
        # In counterfactual mode, the post-split plan is authoritative. Do NOT
        # re-synthesise a reactive noradrenaline loop that would silently
        # override the intervention branch (C13).
        from scipy.signal import lfilter as _lft
        _dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.5
        _alpha = float(np.exp(-_dt / 30.0))

        # Reactive noradrenaline controller over the FULL timeline. The RNG is
        # consumed identically for every branch (and for non-CF cases), so two
        # CF branches with the same seed share an identical pre-split prefix.
        nora_reactive = np.zeros(len(t), dtype=np.float64)
        for i in range(len(t)):
            deficit = max(0.0, 65.0 - map_baseline[i])
            base_dose = deficit * 0.012
            base_dose *= float(self.rng_actions.lognormal(0.0, 0.15))
            nora_reactive[i] = float(np.clip(base_dose, 0.0, 0.35))
        nora_reactive = _lft([1.0 - _alpha], [1.0, -_alpha], nora_reactive)

        if not is_cf:
            nora_rate_arr = nora_reactive

            # Build synthetic nora actions for rendering
            nora_active = nora_rate_arr > 0.005
            if nora_active.any():
                changes = np.diff(nora_active.astype(int))
                starts = np.where(changes == 1)[0] + 1
                stops_idx = np.where(changes == -1)[0] + 1
                if nora_active[0]:
                    starts = np.concatenate([[0], starts])
                if nora_active[-1]:
                    stops_idx = np.concatenate([stops_idx, [len(t) - 1]])
                for s, e in zip(starts, stops_idx):
                    dose = float(np.mean(nora_rate_arr[s:e+1]))
                    actions.append(Action(
                        t_s=float(t[s]), action_type=ActionType.VASOACTIVE_INFUSION,
                        drug="noradrenaline",
                        value=float(np.clip(dose, 0.03, 0.35)),
                        unit="mcg/kg/min", route="IV",
                    ))
                    actions.append(Action(
                        t_s=float(t[min(e, len(t)-1)]), action_type=ActionType.INFUSION_STOP,
                        drug="noradrenaline", value=0.0, unit="mcg/kg/min",
                    ))
        else:
            # CF mode (C13): pre-split keeps the realistic reactive loop; after
            # the split the supplied plan is authoritative and no action is
            # silently re-synthesised.
            nora_inf_explicit = self._actions_to_infusions(actions, "noradrenaline")
            nora_explicit = np.array([self._rate_at(ti, nora_inf_explicit) for ti in t], dtype=np.float64)
            nora_explicit = _lft([1.0 - _alpha], [1.0, -_alpha], nora_explicit)
            split_idx = int(np.searchsorted(t, counterfactual_split_t, side="right"))
            nora_rate_arr = nora_reactive.copy()
            nora_rate_arr[split_idx:] = nora_explicit[split_idx:]

            # Append pre-split reactive nora actions for rendering (post-split
            # plan actions are already in `actions`).
            pre_active = nora_rate_arr[:split_idx] > 0.005
            if pre_active.any():
                changes = np.diff(pre_active.astype(int))
                starts = np.where(changes == 1)[0] + 1
                stops_idx = np.where(changes == -1)[0] + 1
                if pre_active[0]:
                    starts = np.concatenate([[0], starts])
                if pre_active[-1]:
                    stops_idx = np.concatenate([stops_idx, [split_idx - 1]])
                for s, e in zip(starts, stops_idx):
                    e = min(e, split_idx - 1)
                    dose = float(np.mean(nora_rate_arr[s:e+1]))
                    actions.append(Action(
                        t_s=float(t[s]), action_type=ActionType.VASOACTIVE_INFUSION,
                        drug="noradrenaline",
                        value=float(np.clip(dose, 0.03, 0.35)),
                        unit="mcg/kg/min", route="IV",
                    ))
                    actions.append(Action(
                        t_s=float(t[e]), action_type=ActionType.INFUSION_STOP,
                        drug="noradrenaline", value=0.0, unit="mcg/kg/min",
                    ))

        # Rebuild nora/phen infusion schedules
        nora_inf = self._actions_to_infusions(actions, "noradrenaline")
        phen_inf = self._actions_to_infusions(actions, "phenylephrine")
        phen_rate_arr = np.array([self._rate_at(ti, phen_inf) for ti in t], dtype=np.float64)
        vasoactive = [a for a in actions if a.action_type in (
            ActionType.VASOACTIVE_BOLUS, ActionType.VASOACTIVE_INFUSION,
        )]

        # ── Ephedrine tachyphylaxis: cumulative BEFORE current timestep ──
        # The current bolus should NOT suffer tachyphylaxis from itself,
        # only from previous boluses. Use strict < for cumulative.
        cum_eph = np.zeros(len(t), dtype=np.float64)
        eph_actions = [(a.t_s, a.value or 0) for a in vasoactive
                       if a.drug == "ephedrine"]
        for ts, dose in eph_actions:
            cum_eph[t > ts] += dose  # strict > : bolus at t_s counts for t > t_s
        self.hr_model.reset_tachyphylaxis()

        # Final MAP / HR — v3.2: ephedrine with Emax + tachyphylaxis
        map_vals = np.zeros(len(t), dtype=np.float64)
        hr_vals = np.zeros(len(t), dtype=np.float64)
        for i, (ti, p, r, m, nm, nh) in enumerate(zip(
            t, ce_prop, ce_rem, mac_arr, noci_map, noci_hr,
        )):
            self.hr_model._cumulative_eph_mg = float(cum_eph[i])
            map_vals[i] = self.hr_model.map_from_state(
                ce_prop=p, ce_rem=r, sevo_mac=m,
                nociceptive_drive_map=float(nm),
                noradrenaline_rate=float(nora_rate_arr[i]),
                phenylephrine_bolus=self._decayed_bolus(ti, vasoactive, "phenylephrine", 300.0),
                phenylephrine_rate=float(phen_rate_arr[i]) / patient.weight,
                ephedrine_bolus=self._decayed_bolus(ti, vasoactive, "ephedrine", 600.0),
                weight_kg=patient.weight,
            )
            hr_vals[i] = self.hr_model.hr_from_state(
                ce_prop=p, ce_rem=r, sevo_mac=m,
                nociceptive_drive_hr=float(nh),
                ephedrine_bolus=self._decayed_bolus(ti, vasoactive, "ephedrine", 600.0),
                weight_kg=patient.weight,
            )

        # Apply cross-arrows: PEEP→MAP and desaturation→HR/MAP
        desat_hr_arr = np.zeros(len(t), dtype=np.float32)
        desat_map_arr = np.zeros(len(t), dtype=np.float32)
        for i in range(1, len(t)):
            desat_hr_arr[i], desat_map_arr[i] = resp_model.compute_desaturation_hemodynamics(
                spo2_arr[i], spo2_arr[i-1], dt,
                hr_vals[i], map_vals[i],
            )
        map_vals = map_vals + peep_map_effect_arr + desat_map_arr
        map_vals = np.clip(map_vals, 50.0, 140.0)
        hr_vals = hr_vals + desat_hr_arr
        hr_vals = np.clip(hr_vals, 40.0, 180.0)

        # Intubation tachycardia
        intub_p = next((p for p in timeline.phases if p.name == "intubation"), None)
        if intub_p is not None:
            intub_t0 = float(intub_p.start_s)
            hr_peak = float(self.rng_pkpd.uniform(110.0, 180.0))
            hr_dur = float(self.rng_pkpd.uniform(30.0, 120.0))
            sh = hr_dur / 2.0
            sc = intub_t0 + sh
            sm = (t >= intub_t0) & (t <= intub_t0 + hr_dur)
            if sm.any():
                g = hr_peak * np.exp(-0.5 * ((t[sm] - sc) / (sh / 2.5)) ** 2)
                hr_vals[sm] = np.maximum(hr_vals[sm], g)

        # -- Ventilation & other signals (v3: physiological, not AR(1)) --
        spo2 = spo2_arr
        etco2 = etco2_arr
        fio2 = np.full_like(t, 0.5)

        # MAC smoothing
        _md = float(np.mean(np.diff(t))) if len(t) > 1 else float(dt)
        _ma = float(np.exp(-_md / 150.0))
        mac_obs = _lft([1.0 - _ma], [1.0, -_ma], mac_arr)

        ppf_rate_arr = np.array([self._rate_at(ti, ppf_inf) for ti in t])
        remi_rate_arr = np.array([self._rate_at(ti, remi_inf) for ti in t])

        # -- Observed tracks --
        tracks = self._build_tracks(
            t, patient, map_vals, hr_vals, bis, spo2_arr, etco2_arr, fio2_arr,
            ce_prop, cp_prop, prop_st, ppf_rate_arr,
            ce_rem, cp_rem, remi_st, remi_rate_arr,
            mac_arr, mac_obs, roc_boluses, vasoactive, nora_inf,
            peep=peep_arr, rr=rr_arr, tv=tv_arr, pip=pip_arr,
            compliance=resp_model.compliance,
            ppf_boluses=ppf_boluses, remi_boluses=remi_boluses,
            phen_inf=phen_inf, timeline=timeline,
        )

        # -- Presence --
        # Control tracks (actions/bolus events/setpoints) must always be present
        # in synthetic cases; only sensor observations are subject to dropout.
        always = {"time"} | CONTROL_TRACKS
        opt = [c for c in tracks if c not in always]
        present = (
            [c for c in presence_override if c in tracks]
            if presence_override is not None
            else self.track_presence.sample(opt)
        )
        tracks = {c: tracks[c] for c in always | set(present)}

        # -- Truth sidecar --
        roc_dose_arr = np.array([
            self._cum_bolus(ti, roc_boluses, "rocuronium") for ti in t
        ])
        # nora_rate_arr already computed above (direct MAP deficit controller)
        eph_dose_arr = np.array([
            self._cum_bolus(ti, vasoactive, "ephedrine") for ti in t
        ])
        phen_dose_arr = np.array([
            self._cum_bolus(ti, vasoactive, "phenylephrine") for ti in t
        ])

        truth = {
            "caseid": caseid, "time": t,
            "phase": [timeline.get_phase_at(ti) for ti in t],
            "ce_propofol": ce_prop, "cp_propofol": cp_prop,
            "ce_remifentanil": ce_rem, "cp_remifentanil": cp_rem,
            "bis": bis, "map": map_vals, "map_baseline": map_baseline,
            "hr": hr_vals,
            # Clean physiological respiratory truth (no sensor dropout), so the
            # F2.1 action-support gate can evaluate ventilation levers.
            "spo2": spo2_arr, "etco2": etco2_arr,
            "propofol_rate": ppf_rate_arr, "remifentanil_rate": remi_rate_arr,
            "rocuronium_dose": roc_dose_arr,
            "noradrenaline_rate": nora_rate_arr,
            "ephedrine_dose": eph_dose_arr,
            "phenylephrine_dose": phen_dose_arr,
            "sevoflurane_mac": mac_arr,
            "surgical_stimulus": stim,
            "nociceptive_map": noci_map, "nociceptive_hr": noci_hr,
            "nociceptive_bis": noci_bis,
            "remi_attenuation_map": att_map, "remi_attenuation_hr": att_hr,
            "actions": [a.summary() for a in actions],
            "a_null_mask": perturb["a_null"],
            "propofol_perturbation": perturb["ppf_mult"],
            "remi_perturbation": perturb["remi_mult"],
            # v3: count of direct exogenous vasopressor actions in this case
            "n_exo_vaso_actions": np.full_like(t, len(perturb.get("exo_vaso_actions", [])), dtype=np.float32),
            # ── v3 respiratory oracle ────────────────────────────────────
            "pao2_true": pao2_arr,
            "shunt_fraction": np.full_like(t, resp_model.shunt_frac),
            "vco2": np.full_like(t, resp_model.vco2),
            "fio2_applied": fio2_arr,
            "peep_applied": peep_arr,
            "rr_applied": rr_arr,
            "tv_applied": tv_arr,
            "peep_map_effect": peep_map_effect_arr,
            "desaturation_event": (spo2_arr < 90.0).astype(np.float32),
            "desat_hr_delta": desat_hr_arr,
            "desat_map_delta": desat_map_arr,
            "pip_true": pip_arr,
            "compliance": np.full_like(t, resp_model.compliance),
        }

        # -- Clinical row --
        crow = patient.to_clinical_dict()
        crow["caseend"] = int(timeline.total_duration_s)
        crow["aneend"] = float(timeline.total_duration_s)
        crow["opend"] = int(timeline.total_duration_s - 10 * 60)
        crow["dis"] = int(timeline.total_duration_s + 24 * 3600)
        crow["intraop_ppf"] = int(tracks.get("Orchestra/PPF20_VOL", [0])[-1]) if "Orchestra/PPF20_VOL" in tracks else 0
        crow["intraop_ftn"] = int(tracks.get("Orchestra/RFTN20_VOL", [0])[-1]) if "Orchestra/RFTN20_VOL" in tracks else 0

        metadata = {
            "pk_model": pk_name,
            "prop_sensitivity": prop_sensitivity,
            "vasopressor_response": vasopressor_response,
            "nociception": noci_meta,
            "is_counterfactual": is_cf,
            "counterfactual_split_t": counterfactual_split_t,
        }

        return {
            "patient": patient, "timeline": timeline,
            "tracks": tracks, "truth": truth,
            "clinical_row": crow, "actions": actions,
            "metadata": metadata,
        }

    # ==================================================================
    # Action building  v2 with valle nociceptivo + dose expansion
    # ==================================================================
    def _build_actions(self, timeline, patient: Patient) -> list[Action]:
        actions = []
        actions.append(Action(t_s=0, action_type=ActionType.VENTILATOR_SETTING,
                              drug="fio2", value=0.5, unit="fraction"))

        induction = next(p for p in timeline.phases if p.name == "induction")
        intubation = next(p for p in timeline.phases if p.name == "intubation")
        maintenance = next(p for p in timeline.phases if p.name == "maintenance")
        emergence = next(p for p in timeline.phases if p.name == "emergence")

        # Propofol induction: A5 recalibración — tasas realistas para ce_propofol
        # de mantenimiento ~2-4.5 µg/mL (antes 9-11 → MAP/BIS saturados en piso).
        ppf_bolus = float(self.rng_actions.uniform(1.5, 3.0)) * patient.weight
        actions.append(Action(t_s=induction.start_s, action_type=ActionType.BOLUS,
                              drug="propofol", value=ppf_bolus, unit="mg", route="IV"))
        ppf_init = float(self.rng_actions.uniform(3.0, 6.0)) * patient.weight / 60.0
        actions.append(Action(t_s=induction.start_s + 30.0,
                              action_type=ActionType.INFUSION_START,
                              drug="propofol", value=ppf_init, unit="mg/min", route="IV"))

        # Remifentanil: valle nociceptivo
        # Target Ce ≈ 1-3 ng/mL for typical practice (CL=2.5, R = Ce×CL = 2.5-7.5 mcg/kg/h)
        valle = model_remi_valle(np.array([0.0]), timeline, self.rng_actions)
        if valle["has_bolus"]:
            bolus_mcg = valle["bolus_dose_mcg_per_kg"] * patient.lean_body_weight
            actions.append(Action(t_s=valle["bolus_t_s"], action_type=ActionType.BOLUS,
                                  drug="remifentanil", value=bolus_mcg, unit="mcg", route="IV"))
        remi_initial = float(self.rng_actions.uniform(0.03, 0.12)) * patient.weight
        actions.append(Action(t_s=induction.start_s + 60.0,
                              action_type=ActionType.INFUSION_START,
                              drug="remifentanil", value=remi_initial, unit="mcg/min", route="IV"))
        if valle["has_refuerzo"]:
            ref_mcg = valle["refuerzo_dose_frac"] * valle["bolus_dose_mcg_per_kg"] * patient.lean_body_weight
            actions.append(Action(t_s=float(valle["refuerzo_t_s"]),
                                  action_type=ActionType.BOLUS,
                                  drug="remifentanil", value=ref_mcg, unit="mcg", route="IV"))

        # Rocuronium
        roc_dose = float(self.rng_actions.uniform(0.5, 1.0)) * patient.weight
        actions.append(Action(t_s=intubation.start_s, action_type=ActionType.BOLUS,
                              drug="rocuronium", value=roc_dose, unit="mg", route="IV"))

        # Maintenance adjustments: A5 recalibración — tasas realistas y frecuentes
        # para que ce_propofol/BIS tengan varianza de mantenimiento realista.
        # v5.1: rango algo más amplio (2.0-9.0) para que el rango de ce_propofol
        # en mantenimiento supere 0.5 µg/mL (ce_unresponsive_ppf) en TODOS los casos.
        n_adj = max(1, int((maintenance.end_s - maintenance.start_s) // 180))
        remi_on = True
        for i in range(1, n_adj + 1):
            ta = maintenance.start_s + i * 180
            ppf_r = float(self.rng_actions.uniform(1.5, 10.0)) * patient.weight / 60.0
            ppf_r = self.randomization.perturb_infusion_rate(max(0.0, ppf_r), base_cv=0.25)
            actions.append(Action(t_s=ta, action_type=ActionType.INFUSION_CHANGE,
                                  drug="propofol", value=ppf_r, unit="mg/min", route="IV"))
            # v5.1: remifentanilo es infusión continua. Antes el toggle simétrico
            # (10% apagar / 10% encender) dejaba remi apagado ~50% del tiempo y en
            # ~0.5% de casos >90% apagado → p90(remi_rate)=0 → unit_issue_rftn20_rate.
            # Ahora: parada rara y breve (4%), rearranque casi seguro (80%).
            if remi_on:
                if self.rng_actions.random() < 0.04:
                    actions.append(Action(t_s=ta, action_type=ActionType.INFUSION_STOP,
                                          drug="remifentanil", value=0.0, unit="mcg/min"))
                    remi_on = False
                else:
                    remi_r = float(self.rng_actions.uniform(0.02, 0.15)) * patient.weight
                    actions.append(Action(t_s=ta, action_type=ActionType.INFUSION_CHANGE,
                                          drug="remifentanil", value=max(0.0, remi_r), unit="mcg/min"))
            else:
                if self.rng_actions.random() < 0.80:
                    remi_r = float(self.rng_actions.uniform(0.02, 0.15)) * patient.weight
                    actions.append(Action(t_s=ta, action_type=ActionType.INFUSION_START,
                                          drug="remifentanil", value=max(0.0, remi_r), unit="mcg/min"))
                    remi_on = True

        # Emergence: stop
        actions.append(Action(t_s=emergence.start_s, action_type=ActionType.INFUSION_STOP,
                              drug="propofol", value=0, unit="mg/min"))
        actions.append(Action(t_s=emergence.start_s, action_type=ActionType.INFUSION_STOP,
                              drug="remifentanil", value=0, unit="mcg/min"))

        # Sevoflurane: 50% of cases, Emax on hypnotic axis.
        # v5.2: MAC de mantenimiento 0.3-1.0 (mediana ~0.5) — anestesia balanceada
        # todavía más conservadora: la combinación con propofol+remi debe dejar el
        # BIS en la zona sensible (no saturada) para que ce_propofol lo mueva.
        if self.rng_actions.random() < 0.50:
            n_mac = 1 + self.rng_actions.poisson(1.5)
            mac_ts = sorted(self.rng_actions.uniform(maintenance.start_s, emergence.start_s, size=n_mac))
            mac = min(float(self.rng_actions.lognormal(np.log(0.5), 0.25)), 1.0)
            for j, mt in enumerate(mac_ts):
                if j > 0:
                    mac = float(np.clip(mac * self.rng_actions.lognormal(0.0, 0.12), 0.3, 1.0))
                actions.append(Action(t_s=mt, action_type=ActionType.VENTILATOR_SETTING,
                                      drug="sevoflurane", value=mac, unit="MAC"))
            actions.append(Action(t_s=emergence.start_s, action_type=ActionType.VENTILATOR_SETTING,
                                  drug="sevoflurane", value=0.0, unit="MAC"))

        return sorted(actions, key=lambda a: a.t_s)

    # ==================================================================
    # Exogenous perturbations  INDEPENDENT of state
    # ==================================================================
    def _build_exogenous_perturbations(self, t, timeline) -> dict:
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else self.config.dt_seconds
        maintenance = next(p for p in timeline.phases if p.name == "maintenance")
        emergence = next(p for p in timeline.phases if p.name == "emergence")

        # a_null segments: 2-4 per case, 3-10 min
        a_null = np.zeros(n, dtype=bool)
        n_seg = self.rng_actions.integers(2, 5)
        valid = max(1.0, emergence.start_s - maintenance.start_s - 600.0)
        for _ in range(n_seg):
            d = self.rng_actions.uniform(180.0, 600.0)
            s = self.rng_actions.uniform(maintenance.start_s + 120.0,
                                         maintenance.start_s + 120.0 + valid)
            a_null[(t >= s) & (t <= s + d)] = True

        # Propofol perturbations: A5 — rango realista 0.5x-1.5x (antes 0.25-3.5).
        ppf_m = np.ones(n)
        for _ in range(self.rng_actions.integers(3, 7)):
            d = self.rng_actions.uniform(120.0, 900.0)
            s = self.rng_actions.uniform(0, max(1.0, t[-1] - d))
            ppf_m[(t >= s) & (t <= s + d)] = self.rng_actions.uniform(0.45, 1.8)

        # Remi perturbations: rango realista
        remi_m = np.ones(n)
        for _ in range(self.rng_actions.integers(3, 6)):
            d = self.rng_actions.uniform(120.0, 900.0)
            s = self.rng_actions.uniform(0, max(1.0, t[-1] - d))
            remi_m[(t >= s) & (t <= s + d)] = self.rng_actions.uniform(0.5, 2.0)

        # Vasopressor multipliers (legacy, not primary mechanism in v3)
        vaso_m = np.ones(n)

        # PROPOFOL 0.6x while BIS rises (A5: caída moderada, no 0.3x)
        if self.rng_actions.random() < 0.50:
            ps = self.rng_actions.uniform(maintenance.start_s + 600.0, emergence.start_s - 600.0)
            pd = self.rng_actions.uniform(300.0, 900.0)
            ppf_m[(t >= ps) & (t <= ps + pd)] = 0.6

        # REMI HIGH without stimulus
        if self.rng_actions.random() < 0.40:
            rs = self.rng_actions.uniform(maintenance.start_s + 600.0, emergence.start_s - 600.0)
            rd = self.rng_actions.uniform(300.0, 900.0)
            remi_m[(t >= rs) & (t <= rs + rd)] = 2.0

        # ── v3.1: Minimal exogenous vasopressor (preserves NORA↔MAP causal link) ──
        # Only 0-2 short exogenous nora windows per case, low doses.
        # The reactive loop is the PRIMARY source of norepinephrine.
        exo_vaso_actions: list[Action] = []
        maint_dur = emergence.start_s - maintenance.start_s
        n_exo_windows = self.rng_actions.integers(0, 3)       # v3.1: 0-2 only

        for _ in range(n_exo_windows):
            t_start = self.rng_actions.uniform(maintenance.start_s + 60.0, emergence.start_s - 600.0)
            t_dur = self.rng_actions.uniform(180.0, 600.0)     # v3.1: shorter
            dose = self.rng_actions.uniform(0.02, 0.06)         # v3.1: very low
            exo_vaso_actions.append(Action(
                t_s=t_start, action_type=ActionType.VASOACTIVE_INFUSION,
                drug="noradrenaline", value=dose, unit="mcg/kg/min", route="IV",
                metadata={"exogenous": True, "duration_s": t_dur},
            ))
            exo_vaso_actions.append(Action(
                t_s=t_start + t_dur, action_type=ActionType.INFUSION_STOP,
                drug="noradrenaline", value=0.0, unit="mcg/kg/min",
                metadata={"exogenous": True},
            ))

        # Exogenous continuous phenylephrine infusion windows (0-1 per case)
        # so the phen_rate lever has observational support (µg/min).
        n_phen_inf = self.rng_actions.integers(0, 2)
        for _ in range(n_phen_inf):
            t_start = self.rng_actions.uniform(maintenance.start_s + 60.0, emergence.start_s - 600.0)
            t_dur = self.rng_actions.uniform(300.0, 900.0)
            dose = self.rng_actions.uniform(20.0, 80.0)  # µg/min
            exo_vaso_actions.append(Action(
                t_s=t_start, action_type=ActionType.VASOACTIVE_INFUSION,
                drug="phenylephrine", value=dose, unit="mcg/min", route="IV",
                metadata={"exogenous": True, "duration_s": t_dur},
            ))
            exo_vaso_actions.append(Action(
                t_s=t_start + t_dur, action_type=ActionType.INFUSION_STOP,
                drug="phenylephrine", value=0.0, unit="mcg/min",
                metadata={"exogenous": True},
            ))

        n_phen = self.rng_actions.integers(1, 3)              # v3.1: minimal exogenous boluses
        for _ in range(n_phen):
            t_b = self.rng_actions.uniform(maintenance.start_s + 60.0, emergence.start_s - 300.0)
            exo_vaso_actions.append(Action(
                t_s=t_b, action_type=ActionType.VASOACTIVE_BOLUS,
                drug="phenylephrine", value=self.rng_actions.uniform(40.0, 100.0),
                unit="mcg", route="IV", metadata={"exogenous": True},
            ))

        n_eph = self.rng_actions.integers(0, 2)               # v3.1: minimal exogenous boluses
        for _ in range(n_eph):
            t_b = self.rng_actions.uniform(maintenance.start_s + 60.0, emergence.start_s - 300.0)
            exo_vaso_actions.append(Action(
                t_s=t_b, action_type=ActionType.VASOACTIVE_BOLUS,
                drug="ephedrine", value=self.rng_actions.uniform(4.0, 10.0),
                unit="mg", route="IV", metadata={"exogenous": True},
            ))

        return {
            "ppf_mult": ppf_m, "remi_mult": remi_m,
            "vaso_mult": vaso_m, "a_null": a_null,
            "exo_vaso_actions": exo_vaso_actions,
        }

    def _apply_rate_perturbations(self, infusions, multipliers, t, drug):
        if not infusions:
            return []
        diffs = np.diff(multipliers)
        chg = np.where(np.abs(diffs) > 1e-6)[0]
        if len(chg) == 0:
            return infusions
        chg_t = t[chg + 1]
        new = []
        for inf in infusions:
            if inf.drug != drug:
                new.append(inf)
                continue
            rel = [ct for ct in chg_t if inf.start_s < ct < inf.end_s]
            if not rel:
                mask = (t >= inf.start_s) & (t <= inf.end_s)
                avg = float(np.mean(multipliers[mask])) if mask.any() else 1.0
                new.append(Infusion(start_s=inf.start_s, end_s=inf.end_s,
                                    rate=inf.rate * avg, drug=inf.drug))
                continue
            bounds = sorted([inf.start_s] + rel + [inf.end_s])
            for i in range(len(bounds) - 1):
                a, b = bounds[i], bounds[i + 1]
                mid = (a + b) / 2.0
                mi = min(int(np.searchsorted(t, mid)), len(multipliers) - 1)
                mult = float(multipliers[mi])
                new.append(Infusion(start_s=a, end_s=b,
                                    rate=inf.rate * mult, drug=inf.drug))
        return new

    # ==================================================================
    # Vasoactive loop — REACTIVE ONLY (exogenous actions injected directly)
    # ==================================================================
    def _close_vasoactive_loop_reactive(
        self, t, map_bl, actions, w, patient,
    ) -> list[Action]:
        """Pure reactive vasopressor loop: treat hypotension aggressively — v3.1.

        - Starts nora at MAP<68, titrates up if MAP stays low.
        - Escalates dose every 60s while MAP<65 (up to 0.35 mcg/kg/min).
        - Stops nora when MAP>75.
        - Rescue boluses at MAP<60.
        """
        MAP_B = 60.0; MAP_NS = 70.0; MAP_NX = 78.0
        MAP_ESCALATE = 65.0  # escalate if MAP still below this
        BI = 60.0; CD = 300.0; MIN_D = 30.0   # v3.1: faster response
        WIN = int(10.0 / self.config.dt_seconds)  # 10s MA (was 30s)

        stops = [a.t_s for a in actions if a.action_type == ActionType.INFUSION_STOP
                 and a.drug in ("propofol", "remifentanil")]
        em_s = min(stops) if stops else t[-1]
        aw = t <= em_s - CD

        new_a = list(actions)

        # Detect if exogenous nora is already running at t=0
        exo_nora_running = False
        for a in sorted(actions, key=lambda x: x.t_s):
            if a.drug == "noradrenaline":
                if a.action_type == ActionType.VASOACTIVE_INFUSION:
                    exo_nora_running = True
                elif a.action_type == ActionType.INFUSION_STOP:
                    exo_nora_running = False

        nr = exo_nora_running; nlt = -np.inf; lbs = -np.inf
        nd = 0.0  # v3.1: no delay, immediate response
        ndc = float(self.rng.uniform(0.12, 0.28))
        bfp = float(self.rng.uniform(0.03, 0.08))
        nsp = float("inf")
        current_dose = 0.0  # track current nora dose for escalation

        for i in range(len(t)):
            if not aw[i]: continue
            ma_bl = float(np.mean(map_bl[max(0, i - WIN + 1): i + 1]))
            # v3.1: use EFFECTIVE MAP (including estimated nora effect)
            nora_effect = current_dose * 150.0 if nr else 0.0  # approximate
            ma_eff = ma_bl + nora_effect
            ct_start = t[i] - nlt >= MIN_D

            # Start nora if hypotensive
            if not nr and ma_bl < MAP_NS and ct_start:
                if nsp == float("inf"): nsp = t[i]
                if t[i] - nsp >= nd:
                    deficit = max(0.0, 70.0 - ma_bl)
                    dose = deficit * 0.008  # ~0.08 at MAP=60
                    dose *= self.rng.lognormal(0.0, ndc)
                    dose = float(np.clip(dose, 0.05, 0.35))
                    new_a.append(Action(
                        t_s=t[i], action_type=ActionType.VASOACTIVE_INFUSION,
                        drug="noradrenaline",
                        value=dose, unit="mcg/kg/min", route="IV",
                    ))
                    nr = True; nlt = t[i]; nsp = float("inf")
                    current_dose = dose
            else:
                nsp = float("inf")

            # Escalate nora dose if EFFECTIVE MAP still low
            if nr and ma_eff < MAP_ESCALATE and ct_start:
                deficit = max(0.0, 70.0 - ma_eff)
                add_dose = deficit * 0.004  # additional dose
                new_dose = float(np.clip(current_dose + add_dose, 0.05, 0.35))
                if new_dose > current_dose * 1.1:  # only if >10% increase
                    new_a.append(Action(
                        t_s=t[i], action_type=ActionType.VASOACTIVE_INFUSION,
                        drug="noradrenaline",
                        value=new_dose, unit="mcg/kg/min", route="IV",
                    ))
                    nlt = t[i]
                    current_dose = new_dose

            # Stop nora if EFFECTIVE MAP recovers
            if nr and ma_eff > MAP_NX and ct_start:
                new_a.append(Action(t_s=t[i], action_type=ActionType.INFUSION_STOP,
                                    drug="noradrenaline", value=0.0, unit="mcg/kg/min"))
                nr = False; nlt = t[i]; current_dose = 0.0

            # Rescue boluses for severe hypotension
            if map_bl[i] < MAP_B and t[i] - lbs >= BI:
                if self.rng.random() > bfp:
                    ch = self.rng.choice(["ephedrine", "phenylephrine"])
                    dcb = float(self.rng.uniform(0.1, 0.3))
                    if ch == "ephedrine":
                        b = 5.0 + self.rng.exponential(5.0)
                        new_a.append(Action(
                            t_s=t[i], action_type=ActionType.VASOACTIVE_BOLUS,
                            drug="ephedrine",
                            value=float(np.clip(b * self.rng.lognormal(0.0, dcb), 2.0, 20.0)),
                            unit="mg", route="IV",
                        ))
                    else:
                        b = 50.0 + self.rng.exponential(50.0)
                        new_a.append(Action(
                            t_s=t[i], action_type=ActionType.VASOACTIVE_BOLUS,
                            drug="phenylephrine",
                            value=float(np.clip(b * self.rng.lognormal(0.0, dcb), 20.0, 250.0)),
                            unit="mcg", route="IV",
                        ))
                lbs = t[i]

        if nr:
            new_a.append(Action(t_s=em_s - CD, action_type=ActionType.INFUSION_STOP,
                                drug="noradrenaline", value=0.0, unit="mcg/kg/min"))
        return sorted(new_a, key=lambda a: a.t_s)

    # ==================================================================
    # PK / PD helpers
    # ==================================================================
    def _apply_pk_iiv(self, model, k_cv=0.20, v_cv=0.15, ke0_cv=0.15):
        kf = float(self.rng_pkpd.lognormal(0.0, k_cv))
        vf = float(self.rng_pkpd.lognormal(0.0, v_cv))
        kef = float(self.rng_pkpd.lognormal(0.0, ke0_cv))
        model.k10 *= kf; model.k12 *= kf; model.k13 *= kf
        model.k21 *= kf; model.k31 *= kf
        model.v1 *= vf; model.v2 *= vf; model.v3 *= vf
        model.ke0 *= kef

    def _actions_to_infusions(self, actions, drug):
        infs = []; cr = 0.0; ls = 0.0
        for a in sorted(actions, key=lambda x: x.t_s):
            if a.drug != drug: continue
            if a.action_type in (ActionType.INFUSION_START, ActionType.INFUSION_CHANGE,
                                 ActionType.VASOACTIVE_INFUSION):
                if cr > 0:
                    infs.append(Infusion(start_s=ls, end_s=a.t_s, rate=cr, drug=drug))
                cr = a.value or 0.0; ls = a.t_s
            elif a.action_type == ActionType.INFUSION_STOP:
                if cr > 0:
                    infs.append(Infusion(start_s=ls, end_s=a.t_s, rate=cr, drug=drug))
                cr = 0.0; ls = a.t_s
        if cr > 0:
            infs.append(Infusion(start_s=ls, end_s=float("inf"), rate=cr, drug=drug))
        return infs

    def _extract_boluses(self, actions, drug, unit_check: str | None = None) -> list[tuple[float, float]]:
        """Return (t_s, dose) for BOLUS actions of this drug.

        Doses are returned in the model's internal unit (mg for propofol,
        mcg for remifentanil). The simulator records mg and mcg explicitly,
        so no conversion is performed here.
        """
        boluses = []
        for a in actions:
            if a.drug != drug: continue
            if a.action_type != ActionType.BOLUS: continue
            if a.value is None: continue
            boluses.append((float(a.t_s), float(a.value)))
        return sorted(boluses, key=lambda x: x[0])

    def _rate_at(self, t, infs):
        for i in infs:
            if i.start_s <= t <= i.end_s or (t >= i.start_s and i.end_s == float("inf")):
                return i.rate
        return 0.0

    def _sevoflurane_at(self, t, events):
        cur = 0.0
        for a in sorted(events, key=lambda x: x.t_s):
            if a.t_s <= t: cur = a.value or 0.0
        return cur

    def _cum_bolus(self, t, actions, drug):
        return sum(
            a.value or 0.0 for a in actions
            if a.drug == drug and a.t_s <= t
            and a.action_type in (ActionType.BOLUS, ActionType.VASOACTIVE_BOLUS)
        )

    def _decayed_bolus(self, t, actions, drug, hl):
        if hl <= 0: return self._cum_bolus(t, actions, drug)
        tau = hl / np.log(2)
        return sum(
            (a.value or 0.0) * np.exp(-(t - a.t_s) / tau)
            for a in actions
            if a.drug == drug and a.t_s <= t
            and a.action_type in (ActionType.BOLUS, ActionType.VASOACTIVE_BOLUS)
        )

    # ==================================================================
    # Signal generation
    # ==================================================================
    def _ema(self, x, tau, dt=0.5):
        from scipy.signal import lfilter
        a = float(np.exp(-dt / tau))
        return lfilter([1.0 - a], [1.0, -a], x).astype(x.dtype)

    @staticmethod
    def _phase_start(timeline, name: str) -> float:
        for p in timeline.phases:
            if p.name == name:
                return float(p.start_s)
        return 0.0

    def _gen_spo2(self, t):
        from scipy.signal import lfilter
        n = len(t); dt = float(np.mean(np.diff(t))) if n > 1 else 0.5
        tau = 30.0; alpha = float(np.exp(-dt / tau))
        r = self.rng_sensor.random()
        if r < 0.55: bl, ss = float(self.rng_sensor.uniform(98.7, 99.1)), 0.10
        elif r < 0.85: bl, ss = float(self.rng_sensor.uniform(99.2, 99.8)), 0.50
        else: bl, ss = float(self.rng_sensor.uniform(97.5, 99.5)), 1.50
        si = float(np.sqrt(1.0 - alpha ** 2) * ss)
        innov = self.rng_sensor.normal((1.0 - alpha) * bl, si, size=n)
        zi = np.array([float(self.rng_sensor.normal(bl, 0.5))])
        base, _ = lfilter([1.0], [1.0, -alpha], innov, zi=zi)
        if self.rng_sensor.random() < 0.12:
            ne = 1 + self.rng_sensor.poisson(0.8)
            for _ in range(ne):
                ts = self.rng_sensor.uniform(t[0] + 120.0, max(t[0] + 121.0, t[-1] - 120.0))
                dur = self.rng_sensor.exponential(90.0) + 30.0
                d = self.rng_sensor.uniform(8.0, 25.0)
                idx = (t >= ts) & (t <= ts + dur)
                if idx.sum() > 0:
                    base[idx] -= d * np.sin(np.pi * (t[idx] - ts) / dur)
        return np.clip(base, 60.0, 100.0)

    def _gen_etco2(self, t):
        from scipy.signal import lfilter
        n = len(t); dt = float(np.mean(np.diff(t))) if n > 1 else 0.5
        base = float(self.rng_sensor.normal(33.0, 2.5))
        sig = np.full(n, base)
        tau = 300.0; ad = float(np.exp(-dt / tau))
        ss = float(self.rng_sensor.lognormal(np.log(4.5), 0.6))
        sd = float(np.sqrt(1.0 - ad ** 2) * ss)
        innov = self.rng_sensor.normal(0.0, sd, size=n)
        zi = np.array([float(self.rng_sensor.normal(0.0, ss * 0.4))])
        drift, _ = lfilter([1.0], [1.0, -ad], innov, zi=zi)
        return np.clip(sig + drift, 10.0, 60.0)

    # ==================================================================
    # Observed track building
    # ==================================================================
    def _build_tracks(self, t, patient, map_vals, hr_vals, bis, spo2, etco2, fio2,
                      ce_prop, cp_prop, prop_st, ppf_rate_arr,
                      ce_rem, cp_rem, remi_st, remi_rate_arr,
                      mac_arr, mac_obs, roc_boluses, vasoactive, nora_inf,
                      peep=None, rr=None, tv=None, pip=None, compliance=None,
                      ppf_boluses=None, remi_boluses=None, phen_inf=None,
                      timeline=None):
        """Build observed tracks with realistic cadence and masking — v3."""
        if self.rng_sensor.random() < 0.55:
            eg, ed = 0.002, float(self.rng_sensor.uniform(15.0, 30.0))
        else:
            eg = float(self.rng_sensor.uniform(0.003, 0.008))
            ed = float(self.rng_sensor.uniform(30.0, 70.0))

        if peep is None: peep = np.full_like(t, 5.0)
        if rr is None: rr = np.full_like(t, 12.0)
        if tv is None: tv = np.full_like(t, 500.0)
        if pip is None: pip = np.full_like(t, 18.0)

        # ── F2 recalibración: variabilidad entre casos de settings/canales que
        # antes eran constantes (separadores triviales real vs sintético).
        _n2o_on = self.rng_sensor.random() < 0.12
        _n2o_pct = float(self.rng_sensor.uniform(35.0, 68.0)) if _n2o_on else 0.0
        _insp_pause = float(self.rng_sensor.uniform(10.0, 30.0)) if self.rng_sensor.random() < 0.35 else 0.0
        _fresh_flow = float(self.rng_sensor.uniform(0.8, 2.6))
        # v7 (C1): el track COMPLIANCE usa la compliance REAL del modelo
        # respiratorio (consistente con PIP), no un valor independiente.
        _compliance = compliance if compliance is not None else float(self.rng_sensor.uniform(20.0, 45.0))
        _insp_tm = float(self.rng_sensor.uniform(0.8, 1.7))
        _set_pip_alarm = float(self.rng_sensor.uniform(30.0, 45.0))
        _vent_set_pcp = float(self.rng_sensor.uniform(12.0, 24.0))
        _vent_leak0 = float(self.rng_sensor.uniform(5.0, 25.0)) if self.rng_sensor.random() < 0.6 else 0.0
        _mawp = peep + 0.4 * (pip - peep)  # presión media en vía aérea (entre PEEP y PIP)

        # Explicit bolus event spikes (plan C13, §5.3): dose at exact timestamp.
        ppf_bolus_t = [e[0] for e in (ppf_boluses or [])]
        ppf_bolus_d = [e[1] for e in (ppf_boluses or [])]
        remi_bolus_t = [e[0] for e in (remi_boluses or [])]
        remi_bolus_d = [e[1] for e in (remi_boluses or [])]
        roc_bolus_t = [a.t_s for a in roc_boluses]
        roc_bolus_d = [a.value or 0.0 for a in roc_boluses]
        phen_bolus_t = [a.t_s for a in vasoactive if a.drug == "phenylephrine"]
        phen_bolus_d = [a.value or 0.0 for a in vasoactive if a.drug == "phenylephrine"]
        eph_bolus_t = [a.t_s for a in vasoactive if a.drug == "ephedrine"]
        eph_bolus_d = [a.value or 0.0 for a in vasoactive if a.drug == "ephedrine"]

        # Body temperature: A4 — deriva de hipotermia amplia. Referencia real
        # (sin artefactos <30°C): p1=33.2, p50=36.0, p99=37.3 (500 casos, semilla
        # 2024). Mayoría enfría poco (0.2-2.0°C); ~15% hipotermia marcada (2.8-4.5°C).
        # NOTA v7 (C3 REVERTIDO): ensanchar _bt_start a normal(36.0, 0.9) empeoró
        # la W1 de BT (0.30 -> 0.93) y reabrió V1 (0.664 -> 0.765). Se restaura el
        # unif(36.5, 37.4) de v6; la corrección correcta debe ajustar la FORMA.
        dt_bt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.5
        tau_bt = self.rng_sensor.uniform(1800.0, 7200.0)
        _bt_start = self.rng_sensor.uniform(36.5, 37.4)
        if self.rng_sensor.random() < 0.15:
            _bt_drop = self.rng_sensor.uniform(2.8, 4.5)
        else:
            _bt_drop = self.rng_sensor.uniform(0.2, 2.0)
        bt_latent = _bt_start - _bt_drop * (1.0 - np.exp(-t / tau_bt))
        bt_latent = bt_latent + self.rng_sensor.normal(0.0, 0.15, len(t))
        art_bt = self.rng_sensor.random(len(t)) < 0.0003
        if art_bt.any():
            bt_latent[art_bt] = self.rng_sensor.uniform(18.0, 30.0, size=int(art_bt.sum()))

        # BIS signal quality: real SQI mean ~73 (often degraded), not ~95.
        sqi_latent = np.full(len(t), float(self.rng_sensor.uniform(75.0, 88.0)))
        for _ in range(self.rng_sensor.integers(6, 16)):
            s = self.rng_sensor.uniform(t[0], t[-1])
            d = self.rng_sensor.uniform(10.0, 180.0)
            m = (t >= s) & (t <= s + d)
            if m.any():
                sqi_latent[m] = np.minimum(sqi_latent[m], self.rng_sensor.uniform(10.0, 70.0))
        # Deriva lenta de SQI (calidad de electrodo)
        sqi_latent = np.clip(sqi_latent + self._ema(self.rng_sensor.normal(0.0, 6.0, len(t)), 120.0), 0.0, 100.0)

        tracks = {
            "time": t,
            "Solar8000/HR": self._obs_hr(t, hr_vals),
            "Solar8000/ART_SBP": self._obs_art(t, map_vals + 35.0, (-100, 350)),
            "Solar8000/ART_MBP": self._obs_art(t, map_vals, (-100, 350)),
            "Solar8000/ART_DBP": self._obs_art(t, map_vals - 20.0, (-100, 350)),
            "Solar8000/NIBP_SBP": self.sensor.observe(t, map_vals + 31.0, 180.0, 4.0, 0.05, (50, 220)),
            "Solar8000/NIBP_MBP": self.sensor.observe(t, map_vals, 180.0, 3.0, 0.05, (30, 180)),
            "Solar8000/NIBP_DBP": self.sensor.observe(t, map_vals - 17.0, 180.0, 3.0, 0.05, (30, 140)),
            "Solar8000/PLETH_HR": self.sensor.observe(t, hr_vals, 1.0, 2.0, 0.10, (35, 150), gap_rate=0.001, gap_duration_s=30.0),
            "Solar8000/PLETH_SPO2": self._obs_spo2(t, spo2, self._phase_start(timeline, "induction")),
            "Solar8000/ETCO2": self.sensor.observe(t, etco2, 2.0, 0.32, 0.45, (0, 80), gap_rate=eg, gap_duration_s=ed),
            "Solar8000/RR_CO2": self.sensor.observe(t, rr, 2.0, 0.15, 0.40, (0, 70), gap_rate=0.0005, gap_duration_s=20.0),
            "Solar8000/BT": self.sensor.observe(t, bt_latent, 60.0, 0.02, 0.20, (10, 50)),
            "BIS/BIS": self._obs_bis(t, bis),
            "BIS/EMG": self._obs_emg(t, bis),
            "BIS/SQI": self.sensor.observe(t, sqi_latent, 1.0, 3.0, 0.45, (0, 100)),
            "BIS/SEF": self.sensor.observe(t, 8.0 + 0.12 * bis, 1.0, 1.2, 0.45, (0, 30)),
            "BIS/SR": self.sensor.observe(t, np.where(bis < 25.0, self.rng_sensor.uniform(5.0, 60.0, size=len(t)), 0.0), 1.0, 2.0, 0.45, (0, 100)),
            "BIS/TOTPOW": self.sensor.observe(t, 45.0 + 0.3 * bis, 1.0, 15.0, 0.45, (0, 100)),
            "Primus/FIO2": self.sensor.observe(t, fio2 * 100.0 - 5.0, 7.0, 0.5, 0.15, (21, 100)),
            "Primus/ETCO2": self.sensor.observe(t, etco2, 7.0, 0.32, 0.15, (0, 80)),
            "Primus/MAC": self._obs_mac(t, mac_obs),
            "Primus/SET_MAC": np.asarray(mac_arr, dtype=np.float64),
            "Primus/EXP_SEVO": self.sensor.observe(t, mac_obs * 2.0, 7.0, 0.05, 0.15, (0.0, 5.0)),
            "Primus/INSP_SEVO": self.sensor.observe(t, mac_obs * 2.2, 7.0, 0.05, 0.15, (0.0, 5.0)),
            "Primus/EXP_DES": self.sensor.observe(t, np.full_like(t, 0.0), 7.0, 0.0, 0.15, (0.0, 10.0)),
            "Primus/INSP_DES": self.sensor.observe(t, np.full_like(t, 0.0), 7.0, 0.0, 0.15, (0.0, 10.0)),
            "Primus/PEEP_MBAR": self.sensor.observe(t, peep, 7.0, 0.2, 0.15, (0, 25)),
            "Primus/PIP_MBAR": self.sensor.observe(t, pip, 7.0, 0.31, 0.15, (0, 70)),
            "Primus/PPLAT_MBAR": self.sensor.observe(t, pip * 0.85, 7.0, 0.31, 0.15, (0, 60)),
            "Primus/MV": self.sensor.observe(t, tv * rr / 1000.0, 7.0, 0.04, 0.15, (0, 25)),
            "Primus/TV": self.sensor.observe(t, tv, 7.0, 15.0, 0.15, (0, 1600)),
            "Primus/RR_CO2": self.sensor.observe(t, rr, 7.0, 0.15, 0.15, (0, 70)),
            "Primus/SET_FIO2": self.sensor.observe(t, fio2 * 100.0, 7.0, 0.5, 0.15, (21, 100)),
            "Primus/SET_RR_IPPV": self.sensor.observe(t, rr, 7.0, 0.15, 0.15, (0, 70)),
            "Primus/SET_TV_L": self.sensor.observe(t, tv / 1000.0, 7.0, 0.01, 0.15, (0, 1.6)),
            "Primus/SET_PIP": self.sensor.observe(t, np.full_like(t, _set_pip_alarm), 7.0, 0.5, 0.15, (5, 60)),
            "Primus/SET_INSP_TM": self.sensor.observe(t, np.full_like(t, _insp_tm), 7.0, 0.05, 0.15, (0.5, 3.0)),
            "Primus/SET_INSP_PAUSE": self.sensor.observe(t, np.full_like(t, _insp_pause), 7.0, 1.0, 0.15, (0, 50)),
            "Primus/SET_INTER_PEEP": self.sensor.observe(t, peep, 7.0, 0.2, 0.15, (0, 25)),
            "Primus/SET_FRESH_FLOW": self.sensor.observe(t, np.full_like(t, _fresh_flow), 7.0, 0.15, 0.15, (0.2, 6.0)),
            "Primus/SET_AGE": self.sensor.observe(t, np.full_like(t, float(patient.age)), 7.0, 0.0, 0.15, (0, 120)),
            "Primus/MAWP_MBAR": self.sensor.observe(t, _mawp, 7.0, 0.8, 0.15, (3, 30)),
            "Primus/PAMB_MBAR": self.sensor.observe(t, np.full_like(t, 1013.0), 7.0, 2.0, 0.15, (940, 1060)),
            "Primus/VENT_LEAK": self.sensor.observe(t, np.full_like(t, _vent_leak0), 7.0, 1.0, 0.15, (0, 30)),
            "Primus/COMPLIANCE": self.sensor.observe(t, np.full_like(t, _compliance), 7.0, 4.0, 0.15, (10, 100)),
            "Primus/FLOW_AIR": self.sensor.observe(t, np.full_like(t, 0.5), 7.0, 0.05, 0.15, (0, 2)),
            "Primus/FLOW_O2": self.sensor.observe(t, np.full_like(t, 0.5), 7.0, 0.05, 0.15, (0, 2)),
            "Primus/FLOW_N2O": self.sensor.observe(t, np.full_like(t, _n2o_pct / 100.0 * 1.5), 7.0, 0.05, 0.15, (0, 3)),
            "Primus/FEN2O": self.sensor.observe(t, np.full_like(t, _n2o_pct), 7.0, 1.0, 0.15, (0, 80)),
            "Primus/FIN2O": self.sensor.observe(t, np.full_like(t, _n2o_pct), 7.0, 1.0, 0.15, (0, 80)),
            "Primus/FEO2": self.sensor.observe(t, fio2 * 100.0 - 6.0, 7.0, 0.5, 0.15, (21, 100)),
            "Primus/INCO2": self.sensor.observe(t, np.full_like(t, 0.6), 7.0, 0.25, 0.15, (0, 3)),
            "Solar8000/VENT_RR": self.sensor.observe(t, rr, 2.0, 0.15, 0.15, (0, 70)),
            "Solar8000/VENT_MAWP": self.sensor.observe(t, _mawp, 2.0, 0.8, 0.15, (3, 30)),
            "Solar8000/VENT_INSP_TM": self.sensor.observe(t, np.full_like(t, _insp_tm), 2.0, 0.05, 0.15, (0.5, 3.0)),
            "Solar8000/VENT_SET_TV": self.sensor.observe(t, tv, 2.0, 15.0, 0.15, (0, 1600)),
            "Solar8000/VENT_SET_FIO2": self.sensor.observe(t, fio2 * 100.0, 2.0, 0.5, 0.15, (21, 100)),
            "Solar8000/FIO2": self.sensor.observe(t, fio2 * 100.0 - 5.0, 2.0, 0.5, 0.15, (21, 100)),
            "Solar8000/FEO2": self.sensor.observe(t, fio2 * 100.0 - 6.0, 2.0, 0.5, 0.15, (21, 100)),
            "Solar8000/INCO2": self.sensor.observe(t, np.full_like(t, 0.6), 2.0, 0.25, 0.15, (0, 3)),
            "Solar8000/RR": self.sensor.observe(t, rr, 2.0, 0.15, 0.15, (0, 70)),
            "Solar8000/ST_II": self.sensor.observe(t, np.full_like(t, 0.0), 2.0, 0.1, 0.15, (-2, 2)),
            "Solar8000/ST_III": self.sensor.observe(t, np.full_like(t, 0.0), 2.0, 0.1, 0.15, (-2, 2)),
            "Solar8000/ST_I": self.sensor.observe(t, np.full_like(t, 0.0), 2.0, 0.1, 0.15, (-2, 2)),
            "Solar8000/ST_AVR": self.sensor.observe(t, np.full_like(t, 0.0), 2.0, 0.1, 0.15, (-2, 2)),
            "Solar8000/ST_AVL": self.sensor.observe(t, np.full_like(t, 0.0), 2.0, 0.1, 0.15, (-2, 2)),
            "Solar8000/ST_AVF": self.sensor.observe(t, np.full_like(t, 0.0), 2.0, 0.1, 0.15, (-2, 2)),
            "Solar8000/GAS2_INSPIRED": self.sensor.observe(t, mac_obs * 2.0, 2.0, 0.05, 0.15, (0.0, 5.0)),
            "Solar8000/GAS2_EXPIRED": self.sensor.observe(t, mac_obs * 2.0, 2.0, 0.05, 0.15, (0.0, 5.0)),
            "Solar8000/CVP": self.sensor.observe(t, np.full_like(t, 8.0), 2.0, 1.0, 0.15, (0, 20)),
            "Solar8000/VENT_SET_PCP": self.sensor.observe(t, np.full_like(t, _vent_set_pcp), 2.0, 1.0, 0.15, (5, 30)),
            "Solar8000/VENT_PIP": self.sensor.observe(t, pip, 2.0, 0.31, 0.15, (0, 70)),
            "Solar8000/VENT_PPLAT": self.sensor.observe(t, pip * 0.85, 2.0, 0.31, 0.15, (0, 60)),
            "Solar8000/VENT_TV": self.sensor.observe(t, tv, 2.0, 15.0, 0.15, (0, 1600)),
            "Solar8000/VENT_MV": self.sensor.observe(t, tv * rr / 1000.0, 2.0, 0.04, 0.15, (0, 25)),
            "Orchestra/PPF20_RATE": self._obs_drug(t, ppf_rate_arr * 3.0, 0.46, (0, 2000), 1.0, (100.0, 500.0)),
            "Orchestra/PPF20_CE": self.sensor.observe(t, ce_prop, 1.0, 0.05, 0.0, (0, 20)),
            "Orchestra/PPF20_CP": self.sensor.observe(t, cp_prop, 1.0, 0.05, 0.0, (0, 20)),
            "Orchestra/PPF20_CT": self.sensor.observe(t, prop_st[:, 2], 1.0, 0.05, 0.0, (0, 20)),
            "Orchestra/PPF20_VOL": np.cumsum(ppf_rate_arr * 3.0 / 3600.0 * self.config.dt_seconds),
            "Orchestra/RFTN20_RATE": self._obs_drug(t, remi_rate_arr * 3.0, 0.46, (0, 1500), 1.0, (20.0, 150.0)),
            "Orchestra/RFTN20_CE": self.sensor.observe(t, ce_rem, 1.0, 0.1, 0.0, (0, 50)),
            "Orchestra/RFTN20_CP": self.sensor.observe(t, cp_rem, 1.0, 0.1, 0.0, (0, 50)),
            "Orchestra/RFTN20_CT": self.sensor.observe(t, remi_st[:, 2], 1.0, 0.1, 0.0, (0, 50)),
            "Orchestra/RFTN20_VOL": np.cumsum(remi_rate_arr * 3.0 / 3600.0 * self.config.dt_seconds),
            "Orchestra/ROC_RATE": self.sensor.observe_discrete_event(
                t, [a.t_s for a in roc_boluses], [a.value or 0 for a in roc_boluses],
            ),
            "Orchestra/ROC_VOL": np.cumsum(np.array([self._cum_bolus(ti, roc_boluses, "rocuronium") for ti in t])),
            "Orchestra/EPH_RATE": self.sensor.observe_discrete_event(
                t, [a.t_s for a in vasoactive if a.drug == "ephedrine"],
                [a.value or 0 for a in vasoactive if a.drug == "ephedrine"],
            ),
            "Orchestra/EPH_VOL": np.cumsum(np.array([self._cum_bolus(ti, vasoactive, "ephedrine") for ti in t])),
            "Orchestra/PHEN_RATE": self.sensor.observe_discrete_event(
                t, [i.start_s for i in (phen_inf or [])], [i.rate for i in (phen_inf or [])],
            ),
            "Orchestra/PHEN_VOL": np.cumsum(np.array([self._rate_at(ti, phen_inf or []) for ti in t]) / 60.0 * self.config.dt_seconds),
            "Orchestra/NEPI_RATE": self.sensor.observe_discrete_event(
                t, [i.start_s for i in nora_inf], [i.rate for i in nora_inf],
            ),
            "Orchestra/NEPI_VOL": np.cumsum(np.array([self._rate_at(ti, nora_inf) for ti in t]) / 60.0 * self.config.dt_seconds),
            "ppf_bolus_mg": self._event_spike(t, ppf_bolus_t, ppf_bolus_d),
            "remi_bolus_ug": self._event_spike(t, remi_bolus_t, remi_bolus_d),
            "roc_bolus_mg": self._event_spike(t, roc_bolus_t, roc_bolus_d),
            "phen_bolus_mcg": self._event_spike(t, phen_bolus_t, phen_bolus_d),
            "eph_bolus_mg": self._event_spike(t, eph_bolus_t, eph_bolus_d),
        }

        # A1: presencia de tracks por fase clínica. Los helpers ya no aplican
        # ventanas arbitrarias de presencia; aquí se enmascara centralmente.
        self._quantize_tracks(tracks)
        self._apply_presence_masks(tracks, t, timeline)
        return tracks

    def _quantize_tracks(self, tracks: dict) -> None:
        """v6 (M6): redondear el VALOR OBSERVADO a la rejilla del monitor real.
        El estado interno (truth) sigue en continuo; solo se cuantiza la salida."""
        for name, step in QUANT_GRID.items():
            arr = tracks.get(name)
            if not isinstance(arr, np.ndarray) or step <= 0:
                continue
            tracks[name] = np.where(
                np.isnan(arr), arr, np.round(arr / step) * step,
            ).astype(arr.dtype, copy=False)

    def _apply_presence_masks(self, tracks: dict, t: np.ndarray, timeline) -> None:
        """A1: encender cada track solo durante su fase clínica.

        HR/SpO2/NIBP/BT/ART desde la inducción (con jitter de encendido);
        BIS desde ~60 s antes de la inducción; ventilación desde la intubación
        hasta la extubación. Control tracks (bolos, tasas, setpoints) no se
        enmascaran. Referencia: presencia real p5/p50 en
        `data/real_reference_v5.json` (rejilla 5 s + forward-fill).
        """
        phases = {p.name: p for p in timeline.phases}
        ind = phases["induction"].start_s
        intub = phases["intubation"].start_s
        extub = phases["extubation"].start_s if "extubation" in phases else phases["emergence"].start_s
        end = timeline.total_duration_s

        # Jitter de encendido de monitores (algunos casos arrancan tarde el
        # monitor, igual que en reales: HR p5=0.85, p50=0.957).
        rng = self.rng_sensor
        if rng.random() < 0.12:
            monitor_on = ind + float(min(rng.exponential(5.0 * 60.0), 20.0 * 60.0))
        else:
            monitor_on = ind + float(min(rng.exponential(1.0 * 60.0), 8.0 * 60.0))
        monitor_on = min(monitor_on, end)

        # BIS se enciende ligeramente antes de la inducción (real p5=0.97).
        bis_on = max(0.0, ind - 60.0)

        monitor_tracks = {
            "Solar8000/HR", "Solar8000/PLETH_HR", "Solar8000/PLETH_SPO2",
            "Solar8000/NIBP_SBP", "Solar8000/NIBP_MBP", "Solar8000/NIBP_DBP",
            "Solar8000/BT", "Solar8000/ART_SBP", "Solar8000/ART_MBP",
            "Solar8000/ART_DBP", "Solar8000/CVP",
            "Solar8000/ST_I", "Solar8000/ST_II", "Solar8000/ST_III",
            "Solar8000/ST_AVR", "Solar8000/ST_AVL", "Solar8000/ST_AVF",
        }
        bis_tracks = {
            "BIS/BIS", "BIS/EMG", "BIS/SQI", "BIS/SEF", "BIS/SR", "BIS/TOTPOW",
        }
        mac_tracks = {
            "Primus/MAC", "Primus/EXP_SEVO", "Primus/INSP_SEVO",
            "Solar8000/GAS2_INSPIRED", "Solar8000/GAS2_EXPIRED",
            "Primus/EXP_DES", "Primus/INSP_DES",
        }
        vent_tracks = {
            "Primus/ETCO2", "Solar8000/ETCO2", "Solar8000/RR_CO2",
            "Primus/PEEP_MBAR", "Primus/PIP_MBAR", "Primus/PPLAT_MBAR",
            "Primus/MV", "Primus/TV", "Primus/RR_CO2",
            "Primus/FIO2", "Primus/FEO2", "Primus/INCO2", "Primus/MAWP_MBAR",
            "Primus/SET_FIO2", "Primus/SET_RR_IPPV", "Primus/SET_TV_L",
            "Primus/SET_PIP", "Primus/SET_INSP_TM", "Primus/SET_INSP_PAUSE",
            "Primus/SET_INTER_PEEP", "Primus/SET_FRESH_FLOW",
            "Solar8000/VENT_RR", "Solar8000/VENT_MAWP", "Solar8000/VENT_INSP_TM",
            "Solar8000/VENT_SET_TV", "Solar8000/VENT_SET_FIO2",
            "Solar8000/VENT_PIP", "Solar8000/VENT_PPLAT", "Solar8000/VENT_TV",
            "Solar8000/VENT_MV", "Solar8000/VENT_SET_PCP",
            "Solar8000/FIO2", "Solar8000/FEO2", "Solar8000/INCO2", "Solar8000/RR",
        }

        def _mask(name: str, on: float, off: float) -> None:
            if name not in tracks:
                return
            arr = tracks[name]
            if not isinstance(arr, np.ndarray):
                return
            m = (t < on) | (t > off)
            tracks[name] = np.where(m, np.nan, arr).astype(arr.dtype, copy=False)

        for name in monitor_tracks:
            _mask(name, monitor_on, end)
        for name in bis_tracks:
            _mask(name, bis_on, end)
        for name in mac_tracks:
            _mask(name, monitor_on, end)
        for name in vent_tracks:
            _mask(name, intub, end)

    @staticmethod
    def _event_spike(t: np.ndarray, event_times: list[float], event_values: list[float]) -> np.ndarray:
        """Return an array with each event's value placed at the nearest grid point."""
        out = np.zeros(len(t), dtype=np.float64)
        if not event_times:
            return out
        times = np.asarray(event_times, dtype=np.float64)
        values = np.asarray(event_values, dtype=np.float64)
        idx = np.argmin(np.abs(t[:, None] - times[None, :]), axis=0)
        np.add.at(out, idx, values)
        return out

    # ==================================================================
    # Observation helpers
    # ==================================================================
    def _obs_spo2(self, t, spo2, induction_start):
        # A4: SpO2 con variabilidad realista (std intracaso p50 real = 1.45):
        # desaturación leve en inducción (apnea) + desaturaciones ocasionales.
        obs = self.sensor.observe(t, spo2, 2.0, 0.3, 0.05, (40, 100), gap_rate=0.002, gap_duration_s=60.0)
        dur = t[-1] - t[0]
        # Desaturación de inducción (apnea pre-intubación): descenso a 86-93
        # durante 90-180 s tras el inicio de la inducción (siempre presente).
        if induction_start < dur:
            s = induction_start
            d = float(self.rng_sensor.uniform(90.0, 180.0))
            target = float(self.rng_sensor.uniform(86.0, 93.0))
            m = (t >= s) & (t <= s + d) & (~np.isnan(obs))
            if m.any():
                obs[m] = np.clip(target + self.rng_sensor.normal(0.0, 0.5, size=m.sum()), 85.0, 100.0)
        # Desaturaciones transitorias leves en mantenimiento.
        n_desat = int(self.rng_sensor.poisson(2.0))
        for _ in range(n_desat):
            s = float(self.rng_sensor.uniform(induction_start + 180.0, max(induction_start + 181.0, dur - 120.0)))
            d = float(self.rng_sensor.uniform(45.0, 120.0))
            target = float(self.rng_sensor.uniform(88.0, 96.0))
            m = (t >= s) & (t <= s + d) & (~np.isnan(obs))
            if m.any():
                obs[m] = np.clip(target + self.rng_sensor.normal(0.0, 0.5, size=m.sum()), 85.0, 100.0)
        return np.where(np.isnan(obs), np.nan, np.round(obs))

    def _obs_art(self, t, latent, vr):
        from scipy.signal import lfilter as _lf
        # A3: ruido de monitor σ≈2.5 mmHg (antes AR con σ efectiva grande).
        co = float(self.rng_sensor.normal(0.0, 2.0))
        ln = latent + self.rng_sensor.normal(0.0, 2.5, len(latent)) + co
        gr, gd = (0.0, 0.0) if self.rng_sensor.random() < 0.75 else (
            float(self.rng_sensor.uniform(0.002, 0.02)), float(self.rng_sensor.uniform(10.0, 60.0)),
        )
        obs = self.sensor.observe(t, self._ema(ln, 5.0), 2.0, 0.0, 0.009, vr, gap_rate=gr, gap_duration_s=gd)
        # Lavado de línea arterial: 1 evento breve y raro (<1% del tiempo).
        if self.rng_sensor.random() < 0.15:
            as_ = float(self.rng_sensor.uniform(t[0], max(t[0] + 1.0, t[-1] - 60.0)))
            ad = float(self.rng_sensor.uniform(2.0, 5.0))
            am = (t >= as_) & (t <= as_ + ad) & (~np.isnan(obs))
            if am.sum() > 0:
                obs[am] = float(-self.rng_sensor.uniform(10, 60))
        # Aplanamiento transitorio por amortiguación: raro y breve.
        nf = int(self.rng_sensor.poisson(0.2))
        for _ in range(nf):
            fs = float(self.rng_sensor.uniform(t[0], max(t[0] + 1.0, t[-1] - 20.0)))
            fd = float(self.rng_sensor.uniform(2.0, 5.0))
            fm = (t >= fs) & (t <= fs + fd) & (~np.isnan(obs))
            if fm.sum() > 0:
                obs[fm] = float(self.rng_sensor.uniform(180.0, 240.0))
        return obs

    def _obs_bis(self, t, bis_vals):
        # A3: ruido de monitor σ≈2 BIS (antes AR si=8 + drops frecuentes).
        bv = np.clip(bis_vals + self.rng_sensor.normal(0.0, 2.0, len(t)), 0.0, 100.0)
        # Electrobisturí: episodios raros y breves que bajan el BIS.
        n_drop = int(self.rng_sensor.poisson(0.3))
        for _ in range(n_drop):
            ds = float(self.rng_sensor.uniform(t[0], max(t[0] + 1.0, t[-1] - 100.0)))
            dd = float(self.rng_sensor.uniform(15.0, 30.0))
            dt = float(self.rng_sensor.uniform(30.0, 42.0))
            dm = (t >= ds) & (t <= ds + dd)
            if dm.sum() > 0:
                bv[dm] = np.minimum(bv[dm], dt)
        eb = self._ema(bv, 3.0)
        gr, gd = (0.0, 5.0) if self.rng_sensor.random() < 0.9 else (
            float(self.rng_sensor.uniform(0.0005, 0.002)), float(self.rng_sensor.uniform(5.0, 15.0)),
        )
        obs = self.sensor.observe(t, eb, 1.0, 0.0, 0.0001, (0, 100), gap_rate=gr, gap_duration_s=gd)
        return np.clip(obs, 0, 100)

    def _obs_emg(self, t, bis_vals):
        # v6 (M1): EMG recalibrado (nivel ~27, asimetría derecha, acoplamiento
        # positivo a BIS). El ruido del monitor baja de 6.0 a 0.8.
        lat = emg_latent(bis_vals, self.rng_sensor)
        return self.sensor.observe(t, lat, 1.0, _EMG_NOISE_SD, 0.45, (0, 100))

    def _obs_hr(self, t, hr_vals):
        from scipy.signal import lfilter as _lf
        # A3: ruido de monitor σ≈1.5 lpm (antes AR con σ efectiva grande + spikes).
        lh = hr_vals + self.rng_sensor.normal(0.0, 1.5, len(t))
        gr, gd = (0.001, 5.0) if self.rng_sensor.random() < 0.85 else (
            float(self.rng_sensor.uniform(0.002, 0.008)), float(self.rng_sensor.uniform(10.0, 40.0)),
        )
        obs = self.sensor.observe(t, self._ema(lh, 5.0), 2.0, 0.0, 0.02, (0, 300), gap_rate=gr, gap_duration_s=gd)
        # Espigas transitorias raras (taquicardia/bradicardia paroxística).
        ns = int(self.rng_sensor.poisson(0.3))
        for _ in range(ns):
            st = self.rng_sensor.uniform(t[0], max(t[0] + 1.0, t[-1] - 60.0))
            d = self.rng_sensor.exponential(30.0) + 5.0
            delta = self.rng_sensor.uniform(15, 40) if self.rng_sensor.random() < 0.8 else -self.rng_sensor.uniform(8, 15)
            m = (t >= st) & (t <= st + d)
            obs[m] += delta * np.exp(-(t[m] - st) / d * 3)
        return np.clip(obs, 0, 300)

    def _obs_mac(self, t, mac_vals):
        dr = float(self.rng_sensor.uniform(0.05, 0.20)) if self.rng_sensor.random() < 0.7 else float(self.rng_sensor.uniform(0.22, 0.40))
        return self.sensor.observe(t, mac_vals, self.rng_sensor.uniform(6.0, 8.0), 0.02, dr, (0.0, 2.5))

    def _obs_drug(self, t, rate_arr, dropout, vr, uf=3.0, br=(500.0, 1500.0)):
        rm = rate_arr * uf
        nz = rm > 0
        if nz.any():
            fn = int(np.where(nz)[0][0])
            bs = int(self.rng_sensor.uniform(10.0, 60.0) / 0.5)
            brt = float(self.rng_sensor.uniform(*br))
            be = min(fn + bs, len(rm))
            rm[fn:be] = np.maximum(rm[fn:be], brt)
        return self.sensor.observe(t, rm, 1.0, 0.0, dropout, vr, hold=False, timing_jitter_frac=0.8)
