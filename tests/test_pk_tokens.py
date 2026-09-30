"""Tests de src/tokens/pk_tokens.py (módulo PK del contrato de tokens v1).

Metodología: tests primero. Este fichero se escribe ANTES que pk_tokens.py; en la
primera ejecución todo debe estar en ROJO (ModuleNotFoundError).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import paths

from tokens import pk_tokens as pk

ROOT = Path(__file__).resolve().parents[1]
LOG2 = np.log(2.0)
DT_S = 5.0
DT_MIN = DT_S / 60.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _grid(n=2000):
    return np.arange(n) * DT_S


# --------------------------------------------------------------------------
# a) Conversión de unidades de cada RATE a la unidad canónica
# --------------------------------------------------------------------------

def test_rate_unit_conversions():
    # propofol: mL/h (20 mg/mL) -> mg/min = raw/3
    assert pk.rate_to_canonical_per_min("propofol", 600.0) == pytest.approx(200.0)
    # remifentanilo: mL/h (20 ug/mL) -> ug/min = raw/3
    assert pk.rate_to_canonical_per_min("remifentanilo", 300.0) == pytest.approx(100.0)
    # fenilefrina: ug/min -> ug/min
    assert pk.rate_to_canonical_per_min("fenilefrina", 50.0) == pytest.approx(50.0)
    # noradrenalina: ug/kg/min -> ug/kg/min
    assert pk.rate_to_canonical_per_min("noradrenalina", 0.1) == pytest.approx(0.1)
    # efedrina: mg/min -> mg/min
    assert pk.rate_to_canonical_per_min("efedrina", 1.0) == pytest.approx(1.0)
    # rocuronio: mL/h (10 mg/mL) -> mg/min = raw*10/60
    assert pk.rate_to_canonical_per_min("rocuronio", 120.0) == pytest.approx(20.0)
    # sevoflurano: MAC -> MAC (sin conversión)
    assert pk.rate_to_canonical_per_min("sevoflurano", 0.7) == pytest.approx(0.7)


# --------------------------------------------------------------------------
# b) Bolo único con decaimiento exponencial: ce(t) = D·exp(-ke0·t)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("drug,ke0_s", [
    ("fenilefrina", LOG2 / 300.0),
    ("efedrina", LOG2 / 600.0),
])
def test_single_bolus_exponential_decay(drug, ke0_s):
    n = 400
    bolus = np.zeros(n)
    bolus[0] = 100.0  # dosis en la primera celda, aplicada en t=0
    rate = np.zeros(n)
    ce = pk.exponential_ce(ke0_s, bolus, rate, dt_s=DT_S)
    t = _grid(n)
    expected = 100.0 * np.exp(-ke0_s * t)
    assert ce[0] == pytest.approx(100.0, abs=1e-6)
    assert np.allclose(ce, expected, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# c) Infusión constante de fenilefrina: ce converge al estado estacionario
# --------------------------------------------------------------------------

def test_phen_constant_infusion_converges_to_steady_state():
    ke0_s = LOG2 / 300.0
    n = 2000
    rate_per_min = np.full(n, 50.0)  # ug/min
    bolus = np.zeros(n)
    ce = pk.exponential_ce(ke0_s, bolus, rate_per_min, dt_s=DT_S)
    # estado estacionario: R_s/ke0_s con R_s = 50/60 ug/s
    ss = (50.0 / 60.0) / ke0_s
    assert ce[-1] == pytest.approx(ss, rel=1e-5)
    # monótona creciente hacia el estado estacionario
    assert np.all(np.diff(ce) >= 0)


# --------------------------------------------------------------------------
# d) Schnider: pico de Ce entre 1.5 y 2.5 min, Ce(0)=0, balance de masa
# --------------------------------------------------------------------------

def test_schnider_bolus_peak_and_mass_balance():
    p = pk.schnider_params(weight=70.0, age=40.0, height=170.0, sex="M")
    n = 2400  # 200 min
    bolus = np.zeros(n)
    bolus[0] = 100.0  # mg
    rate = np.zeros(n)
    states = pk.build_linear_ce(
        k10=p["k10"], k12=p["k12"], k13=p["k13"], k21=p["k21"], k31=p["k31"],
        ke0=p["ke0"], v1=p["v1"], v2=p["v2"], v3=p["v3"],
        rate_per_min=rate, bolus=bolus, dt_s=DT_S,
    )
    ce = states[:, 3]
    assert ce[0] == pytest.approx(0.0, abs=1e-12)
    ipeak = int(np.argmax(ce))
    t_peak = ipeak * DT_S
    assert 90.0 <= t_peak <= 150.0  # 1.5-2.5 min
    # balance de masa: v1*c1+v2*c2+v3*c3 + eliminado == dosis
    c1 = states[:, 0]
    elim = np.concatenate([[0.0], np.cumsum(p["k10"] * p["v1"] * (c1[:-1] + c1[1:]) / 2.0 * DT_MIN)])
    mass = p["v1"] * states[:, 0] + p["v2"] * states[:, 1] + p["v3"] * states[:, 2] + elim
    assert np.allclose(mass, 100.0, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------
# e) Minto: mismo tipo de comprobación que d)
# --------------------------------------------------------------------------

def test_minto_bolus_peak_and_mass_balance():
    p = pk.minto_params(weight=70.0, height=170.0, age=40.0, sex="M")
    n = 2400
    bolus = np.zeros(n)
    bolus[0] = 100.0  # ug
    rate = np.zeros(n)
    states = pk.build_linear_ce(
        k10=p["k10"], k12=p["k12"], k13=p["k13"], k21=p["k21"], k31=p["k31"],
        ke0=p["ke0"], v1=p["v1"], v2=p["v2"], v3=p["v3"],
        rate_per_min=rate, bolus=bolus, dt_s=DT_S,
    )
    ce = states[:, 3]
    assert ce[0] == pytest.approx(0.0, abs=1e-12)
    ipeak = int(np.argmax(ce))
    t_peak = ipeak * DT_S
    # Minto (ke0=0.6) tiene pico más temprano; se comprueba que existe y es razonable
    assert 30.0 <= t_peak <= 180.0
    c1 = states[:, 0]
    elim = np.concatenate([[0.0], np.cumsum(p["k10"] * p["v1"] * (c1[:-1] + c1[1:]) / 2.0 * DT_MIN)])
    mass = p["v1"] * states[:, 0] + p["v2"] * states[:, 1] + p["v3"] * states[:, 2] + elim
    assert np.allclose(mass, 100.0, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------
# f) Wierda: tras un bolo, Ce monótona creciente hasta el pico y luego decreciente
# --------------------------------------------------------------------------

def test_wierda_bolus_monotonic_peak():
    p = pk.wierda_params(weight=70.0)
    n = 2400
    bolus = np.zeros(n)
    bolus[0] = 50.0  # mg
    rate = np.zeros(n)
    states = pk.build_linear_ce(
        k10=p["k10"], k12=p["k12"], k13=p["k13"], k21=p["k21"], k31=p["k31"],
        ke0=p["ke0"], v1=p["v1"], v2=p["v2"], v3=p["v3"],
        rate_per_min=rate, bolus=bolus, dt_s=DT_S,
    )
    ce = states[:, 3]
    ipeak = int(np.argmax(ce))
    assert 60.0 <= ipeak * DT_S <= 600.0
    # creciente hasta el pico, decreciente después (tolerancia por ruido numérico)
    assert np.all(np.diff(ce[: ipeak + 1]) >= -1e-9)
    assert np.all(np.diff(ce[ipeak:]) <= 1e-9)


# --------------------------------------------------------------------------
# g) Noradrenalina: ce == tasa exactamente
# --------------------------------------------------------------------------

def test_noradrenaline_ce_equals_rate():
    n = 100
    rate_raw = np.array([0.0, 0.05, 0.05, 0.1] * (n // 4), dtype=float)
    demo = dict(weight=70.0, age=40.0, height=170.0, sex="M")
    bolus = np.zeros(n)
    ce = pk.compute_ce("noradrenalina", rate_raw, bolus, demo, dt_s=DT_S)
    assert np.allclose(ce, np.nan_to_num(rate_raw, nan=0.0), rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------
# h) dose_cum no decreciente y termina en bolos + integral de infusión
# --------------------------------------------------------------------------

def test_dose_cum_non_decreasing_and_total():
    n = 100
    bolus = np.zeros(n)
    bolus[5] = 20.0
    bolus[30] = 10.0
    rate = np.full(n, 5.0)  # mg/min
    dc = pk.compute_dose_cum("propofol", bolus, rate, dt_s=DT_S)
    assert np.all(np.diff(dc) >= 0)
    total_bolus = 30.0
    total_infusion = rate[0] * (n - 1) * DT_MIN  # celdas 1..n-1
    assert dc[-1] == pytest.approx(total_bolus + total_infusion, rel=1e-9)


# --------------------------------------------------------------------------
# i) Máscaras de tres estados
# --------------------------------------------------------------------------

def test_ce_mask_three_states():
    n = 6
    finite = np.array([True, True, False, True, False, True])
    # columna ausente en la cohorte -> 1 en todas las celdas
    m1 = pk.ce_mask(cohort_has_col=False, case_has_col=False, finite_grid=finite)
    assert np.array_equal(m1, np.ones(n, dtype=np.uint8))
    # columna presente en la cohorte, ausente en el caso -> 2 en todas
    m2 = pk.ce_mask(cohort_has_col=True, case_has_col=False, finite_grid=finite)
    assert np.array_equal(m2, np.full(n, 2, dtype=np.uint8))
    # columna presente en cohorte y caso -> 0 donde hay valor, 2 donde NaN
    m3 = pk.ce_mask(cohort_has_col=True, case_has_col=True, finite_grid=finite)
    expected = np.array([0, 0, 2, 0, 2, 0], dtype=np.uint8)
    assert np.array_equal(m3, expected)


# --------------------------------------------------------------------------
# c2) Corrección 1: forward-fill de tasas sin tope de edad para integración
# --------------------------------------------------------------------------

def test_forward_fill_hold_for_integration():
    """Tasa constante con un hueco de 120 s: Ce idéntica a la tasa sin hueco
    (hold sin tope de edad) y m_ce = 2 solo en las celdas del hueco más allá de 30 s."""
    n = 100
    grid = np.arange(n) * DT_S
    raw_t = grid.copy()
    raw_v = np.full(n, 10.0)
    gap = (grid >= 50.0) & (grid < 170.0)  # hueco de 120 s
    raw_v[gap] = np.nan
    # (1) hold: idéntico a la tasa sin hueco (sin tope de edad)
    hold = pk._forward_fill_hold(raw_t, raw_v, grid)
    assert np.allclose(hold, 10.0, rtol=1e-12)
    # (2) con max_age 30 s: NaN solo más allá de 30 s desde el último valor válido
    aged = pk._forward_fill(raw_t, raw_v, grid, 30.0)
    expect_finite = (grid < 80.0) | (grid >= 170.0)
    assert np.array_equal(np.isfinite(aged), expect_finite)
    # (3) Ce con hold == Ce sin hueco (tolerancia 1e-6)
    demo = dict(weight=70.0, age=40.0, height=170.0, sex="M")
    ce_hold = pk.compute_ce("noradrenalina", hold, np.zeros(n), demo, dt_s=DT_S)
    ce_nogap = pk.compute_ce("noradrenalina", np.full(n, 10.0), np.zeros(n), demo, dt_s=DT_S)
    assert np.allclose(ce_hold, ce_nogap, rtol=1e-6, atol=1e-7)
    # (4) m_ce = 2 solo en las celdas del hueco más allá de 30 s
    m = pk.ce_mask(True, True, np.isfinite(aged))
    expected = np.where(expect_finite, 0, 2).astype(np.uint8)
    assert np.array_equal(m, expected)


# --------------------------------------------------------------------------
# j) Rocuronio: pico de tasa de bomba == bolo equivalente (CAMBIO 1)
# --------------------------------------------------------------------------

def test_rocuronium_rate_spike_equals_equivalent_bolus():
    """Un pico de ROC_RATE de 1200 mL/h durante 10 s sobre tasa 0 produce en Ce
    el mismo efecto que un bolo de la dosis equivalente (tolerancia 5 %)."""
    n = 200
    demo = dict(weight=70.0, age=40.0, height=170.0, sex="M")
    rate_mlh = np.zeros(n)
    rate_mlh[10:12] = 1200.0  # pico de 10 s (2 celdas)
    ce_rate = pk.compute_ce("rocuronio", rate_mlh, np.zeros(n), demo, dt_s=DT_S)
    # dosis equivalente: 1200 mL/h * 10 s * 10 mg/mL = 33.33 mg
    dose_mg = 1200.0 * (10.0 / 3600.0) * 10.0
    bolus_eq = np.zeros(n)
    bolus_eq[10] = dose_mg  # bolo equivalente al inicio de la celda del pico
    ce_bolus = pk.compute_ce("rocuronio", np.zeros(n), bolus_eq, demo, dt_s=DT_S)
    # mismo efecto: el pico de Ce coincide dentro del 5 % (la entrega en 10 s solo
    # difiere en el transitorio inicial; la discretización a 5 s explica la tolerancia)
    assert ce_rate.max() == pytest.approx(ce_bolus.max(), rel=0.05)
    # y la cola (tras el transitorio de entrega) coincide dentro del 5 %
    assert np.allclose(ce_rate[40:], ce_bolus[40:], rtol=0.05, atol=1e-3)
    assert ce_rate.max() > 0


# --------------------------------------------------------------------------
# Integración (sobre disco). Marcadas pero se ejecutan todas.
# --------------------------------------------------------------------------

INTEGRATION = pytest.mark.integration


def _first_partition(source: str, split: str) -> Path:
    p = paths.WINDOWS_DIR / "windows" / f"source={source}" / f"split={split}" / "part-00000.parquet"
    assert p.exists(), p
    return p


@INTEGRATION
def test_k_determinism_same_partition_twice():
    """Dos ejecuciones sobre la misma partición dan bytes idénticos."""
    part = _first_partition("synthetic_v7", "train")
    with tempfile.TemporaryDirectory() as td:
        out1 = Path(td) / "a" / "part-00000.parquet"
        out2 = Path(td) / "b" / "part-00000.parquet"
        pk.process_partition(part, out1, max_cases=3)
        pk.process_partition(part, out2, max_cases=3)
        b1 = out1.read_bytes()
        b2 = out2.read_bytes()
        assert b1 == b2


@INTEGRATION
def test_l_gate2_cf_prefix_and_lever_divergence():
    """5 pares CF: prefijo idéntico hasta split_t, divergencia exacta en la palanca."""
    pairs = pk.load_cf_pairs()[:5]
    assert len(pairs) == 5
    for pr in pairs:
        a, b = pr["caseid_a"], pr["caseid_b"]
        lever = pr["lever"]
        split_t = pr["split_t"]
        ta = pk.process_case_tokens(int(a), "cf_v7")
        tb = pk.process_case_tokens(int(b), "cf_v7")
        assert set(ta.columns) == set(tb.columns)
        cols = [c for c in ta.columns if c not in ("caseid", "t", "source", "split")]
        # alinear por (caseid, t)
        ta = ta.sort_values("t").reset_index(drop=True)
        tb = tb.sort_values("t").reset_index(drop=True)
        assert len(ta) == len(tb)
        pre_a = ta[ta.t <= split_t]
        pre_b = tb[tb.t <= split_t]
        assert len(pre_a) == len(pre_b)
        assert len(pre_a) > 0
        for c in cols:
            va = pre_a[c].to_numpy()
            vb = pre_b[c].to_numpy()
            assert np.array_equal(va, vb), f"columna {c} difiere en el prefijo del par {pr}"
        # a partir de la primera ventana posterior a split_t, solo difiere la palanca
        post_a = ta[ta.t > split_t]
        post_b = tb[tb.t > split_t]
        assert len(post_a) == len(post_b)
        lever_drug = _lever_drug(lever)
        lever_prefix = {f"ce_{lever_drug}", f"dose_cum_{lever_drug}", f"bolus_{lever_drug}"} \
            if lever_drug else set()
        diverged_any = False
        for c in cols:
            va = post_a[c].to_numpy()
            vb = post_b[c].to_numpy()
            if c in lever_prefix:
                if not np.array_equal(va, vb):
                    diverged_any = True
            else:
                assert np.array_equal(va, vb), f"columna {c} difiere fuera de la palanca en {pr}"
        assert diverged_any, f"palanca {lever} no divergió en ninguna columna PK en {pr}"


def _lever_drug(lever: str):
    mapping = {
        "ppf20_rate": "propofol", "ppf_bolus": "propofol", "propofol_bolus": "propofol",
        "remi_bolus": "remifentanilo", "remi_up": "remifentanilo", "rftn20_rate": "remifentanilo",
        "nepi_rate": "noradrenalina", "noradrenaline": "noradrenalina",
        "phen_bolus": "fenilefrina", "phen_rate": "fenilefrina",
        "eph_bolus": "efedrina", "ephedrine": "efedrina",
        "sevo_mac": "sevoflurano", "sevo_up": "sevoflurano",
        # palancas de ventilación: no afectan a los tokens PK
        "fio2_down": None, "peep_down": None, "peep_up": None, "set_fio2": None,
        "set_peep": None, "set_rr": None, "set_tv": None,
    }
    return mapping.get(lever, None)


@INTEGRATION
def test_m_gate3_integrator_equivalence():
    """Gate 3 (contrato v2): equivalencia del integrador.

    Parte 1 (aserción < 1 %): Ce del módulo (RK4 forma cerrada) frente al
    integrador de anessim (ThreeCompartmentModel.simulate, RK4 iterativo) con los
    mismos parámetros poblacionales (SIN IIV) y la misma serie de RATE (hold) y
    bolos en rejilla de 5 s, para propofol y remifentanilo en 20 casos.

    Parte 2 (diagnóstico, sin aserción): error frente a PPF20_CE / RFTN20_CE
    guardados, en celdas de mantenimiento (t > 600 s) con tasa constante (< 1 %)
    los 3 min previos y sin bolo, 200 casos schnider (propofol) y 200 casos (remi).
    Esperado ~20 % por IIV; es documentación, no criterio.
    """
    from anessim.pk.base import Infusion
    from anessim.pk.propofol import PropofolSchnider
    from anessim.pk.remifentanil import RemifentanilMinto

    meta_dir = paths.COHORTS["synthetic_v7"] / "metadata"
    cases = sorted((paths.COHORTS["synthetic_v7"] / "cases").glob("*.parquet"))
    demo = pk.load_clinical_map()["synthetic_v7"]

    def pk_model(cid: int) -> str:
        return json.loads((meta_dir / f"{cid}_meta.json").read_text(encoding="utf-8"))["pk_model"]

    def anessim_ce(model, grid, rate_hold, bolus_grid):
        """Integrador de anessim (RK4 iterativo, SIN IIV) sobre la misma serie.

        `Infusion.rate_at` usa intervalos cerrados (`start_s <= t <= end_s`); si
        dos segmentos consecutivos comparten el límite, el `_ode` suma ambas tasas
        en el punto de cambio. Se resta un épsilon para que cada celda aporte
        exactamente una tasa (la misma serie que integra el módulo).
        """
        rate_min = rate_hold / 3.0  # mL/h -> mg/min (propofol) o ug/min (remi)
        infs = []
        i = 1
        while i < len(grid):
            j = i
            while j + 1 < len(grid) and rate_min[j + 1] == rate_min[i]:
                j += 1
            if rate_min[i] != 0.0:
                infs.append(Infusion(start_s=grid[i - 1], end_s=grid[j] - 1e-9,
                                     rate=rate_min[i], drug=model.drug))
            i = j + 1
        boluses = [(grid[i - 1] if i > 0 else grid[0], float(bolus_grid[i]))
                   for i in np.where(bolus_grid > 0)[0]]
        st = model.simulate(infs, grid, boluses=boluses)
        return st[:, 3]

    # ---- parte 1: equivalencia del integrador (20 casos) ----
    prop_errs: list[float] = []
    remi_errs: list[float] = []
    for case_path in cases[:20]:
        cid = int(case_path.stem)
        row = demo[cid]
        d = dict(weight=float(row["weight"]), age=float(row["age"]),
                 height=float(row["height"]), sex=str(row["sex"]))
        df = pq.read_table(case_path, columns=[
            "time", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE",
            "ppf_bolus_mg", "remi_bolus_ug"]).to_pandas()
        t = df["time"].to_numpy(float)
        grid = np.arange(t[0], t[-1], DT_S)
        if len(grid) < 2:
            continue
        models = [
            ("propofol", "Orchestra/PPF20_RATE", "ppf_bolus_mg",
             PropofolSchnider(weight_kg=d["weight"], age_y=d["age"],
                              height_cm=d["height"], sex=d["sex"])),
            ("remifentanilo", "Orchestra/RFTN20_RATE", "remi_bolus_ug",
             RemifentanilMinto(weight_kg=d["weight"], height_cm=d["height"],
                               age_y=d["age"], sex=d["sex"])),
        ]
        for drug, rate_col, bol_col, model in models:
            rate_hold = pk._forward_fill_hold(t, df[rate_col].to_numpy(float), grid)
            bolus = pk._bolus_on_grid(t, df[bol_col].to_numpy(float), grid)
            ce_my = pk.compute_ce(drug, rate_hold, bolus, d, dt_s=DT_S)
            ce_an = anessim_ce(model, grid, rate_hold, bolus)
            ok = ce_an > 0.1
            if ok.sum() > 10:
                err = np.abs(ce_my[ok] - ce_an[ok]) / ce_an[ok]
                (prop_errs if drug == "propofol" else remi_errs).extend(err.tolist())

    mean_prop = float(np.mean(prop_errs)) if prop_errs else float("nan")
    mean_remi = float(np.mean(remi_errs)) if remi_errs else float("nan")
    print(f"[gate3] equivalencia integrador: propofol err={mean_prop:.5f} (n={len(prop_errs)}) "
          f"| remi err={mean_remi:.5f} (n={len(remi_errs)})")
    assert len(prop_errs) > 0 and len(remi_errs) > 0
    assert mean_prop < 0.01, f"propofol error relativo vs anessim {mean_prop:.5f} >= 1 %"
    assert mean_remi < 0.01, f"remi error relativo vs anessim {mean_remi:.5f} >= 1 %"

    # ---- parte 2: diagnóstico IIV (sin aserción) ----
    schnider_cases = [c for c in cases if pk_model(int(c.stem)) == "schnider"][:200]
    remi_cases = cases[:200]

    def diag(case_list, drug, out):
        for case_path in case_list:
            cid = int(case_path.stem)
            row = demo[cid]
            d = dict(weight=float(row["weight"]), age=float(row["age"]),
                     height=float(row["height"]), sex=str(row["sex"]))
            rate_col = "Orchestra/PPF20_RATE" if drug == "propofol" else "Orchestra/RFTN20_RATE"
            ce_col = "Orchestra/PPF20_CE" if drug == "propofol" else "Orchestra/RFTN20_CE"
            bol_col = "ppf_bolus_mg" if drug == "propofol" else "remi_bolus_ug"
            # v7: esquemas heterogéneos — algunos casos no tienen el track CE
            # del fármaco (sin administración); el diagnóstico los omite.
            if ce_col not in pq.read_schema(case_path).names:
                continue
            df = pq.read_table(case_path, columns=["time", rate_col, ce_col, bol_col]).to_pandas()
            t = df["time"].to_numpy(float)
            grid = np.arange(t[0], t[-1], DT_S)
            if len(grid) < 2:
                continue
            rate = pk._forward_fill_hold(t, df[rate_col].to_numpy(float), grid)
            bolus = pk._bolus_on_grid(t, df[bol_col].to_numpy(float), grid)
            ce_gen = pk._forward_fill_hold(t, df[ce_col].to_numpy(float), grid)
            ce_my = pk.compute_ce(drug, rate, bolus, d, dt_s=DT_S)
            for i in range(len(grid)):
                if grid[i] <= 600.0 or not np.isfinite(ce_gen[i]) or ce_gen[i] <= 0.5:
                    continue
                lo = max(0, i - 36)  # 3 min previos
                if i - lo < 36:
                    continue
                w = rate[lo:i]
                mn, mx = w.min(), w.max()
                mean_r = (mn + mx) / 2.0
                if mean_r <= 0 or (mx - mn) / mean_r >= 0.01:
                    continue
                if bolus[lo:i].sum() > 0:
                    continue
                out.append(float(abs(ce_my[i] - ce_gen[i]) / ce_gen[i]))

    prop_diag: list[float] = []
    remi_diag: list[float] = []
    diag(schnider_cases, "propofol", prop_diag)
    diag(remi_cases, "remifentanilo", remi_diag)

    def _stats(name, arr):
        a = np.array(arr)
        if len(a):
            print(f"[gate3] diag IIV {name}: n={len(a)} media={a.mean():.4f} "
                  f"mediana={np.median(a):.4f} p90={np.percentile(a, 90):.4f}")
        else:
            print(f"[gate3] diag IIV {name}: n=0")

    _stats("propofol(schnider)", prop_diag)
    _stats("remifentanilo", remi_diag)


@INTEGRATION
def test_n_real_no_vol_and_bolus_zero():
    """CAMBIO 1: en real no se leen columnas *_VOL; bolus_*=0 y bolus_obs_*=0
    para todos los fármacos; efedrina inobservable (m_ce=1)."""
    # (a) el lector no solicita ninguna columna *_VOL
    assert not any(c.endswith("_VOL") for c in pk.CASE_COLS)
    part = _first_partition("real", "train")
    out = tempfile.mkdtemp()
    try:
        result = pk.process_partition(part, Path(out) / "part-00000.parquet", max_cases=10)
        tbl = pq.read_table(Path(out) / "part-00000.parquet").to_pandas()
        assert len(tbl) > 0
        # (b) bolus_* == 0 y bolus_obs_* == 0 en real para todos los fármacos
        for d in pk.DRUGS:
            assert (tbl[f"bolus_{d}"] == 0).all(), d
            assert (tbl[f"bolus_obs_{d}"] == 0).all(), d
        # efedrina inobservable en real
        assert (tbl["m_ce_efedrina"] == 1).all()
        assert result is not None
    finally:
        shutil.rmtree(out, ignore_errors=True)
