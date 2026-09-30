"""Tests de la recalibración v6 del generador (M1-M7).

Cubre las distribuciones y la capa de medida recalibradas:
  M1 BIS/EMG (nivel + asimetría + acoplamiento BIS)
  M2 PEEP (masa en 0 + enteros)
  M3 PIP (compliance ~48 + componente resistivo, v7 C1)
  M4 RR (variabilidad entre/within casos + acoplamiento VCO2 + carga)
  M5 ETCO2 (dispersión + rango, v7 C4)
  M6 cuantización de la capa de medida
  M7 topes ensanchados

No toca el AE ni los pares contrafactuales.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anessim import respiratory as resp  # noqa: E402
from anessim import simulate as sim  # noqa: E402


# --------------------------------------------------------------------------
# M1: EMG
# --------------------------------------------------------------------------

def test_a_emg_level_and_skew():
    rng = np.random.default_rng(0)
    bis = np.clip(rng.normal(41.6, 10.3, 300_000), 0, 100)
    emg = sim.emg_latent(bis, rng)
    assert 26.0 < emg.mean() < 29.0          # nivel real ~27.35 (antes ~38)
    assert 3.0 < np.std(emg) < 5.0           # sd real ~3.69 (antes ~6.5)
    assert np.percentile(emg, 50) < emg.mean()  # asimetría derecha


def test_b_emg_couples_to_bis():
    rng = np.random.default_rng(1)
    low = sim.emg_latent(np.full(200_000, 25.0), rng)
    rng = np.random.default_rng(1)
    high = sim.emg_latent(np.full(200_000, 60.0), rng)
    # Más BIS (más ligero) -> más EMG en media.
    assert high.mean() > low.mean()


# --------------------------------------------------------------------------
# M2: PEEP
# --------------------------------------------------------------------------

def test_c_peep_zero_mass_and_levels():
    rng = np.random.default_rng(2)
    vals = np.array([resp.sample_peep(rng) for _ in range(200_000)])
    assert 0.40 < np.mean(vals == 0.0) < 0.52   # masa en 0 ~46%
    assert 2.0 < vals.mean() < 3.4              # nivel real ~2.66 (antes ~5.6)
    assert np.all(vals == np.round(vals))       # enteros
    assert vals.max() <= 25.0


# --------------------------------------------------------------------------
# M3: compliance -> PIP
# --------------------------------------------------------------------------

def test_d_compliance_gives_realistic_pip():
    rng = np.random.default_rng(3)
    models = [resp.RespiratoryModel(rng, weight_kg=70.0) for _ in range(5000)]
    # v7 (C1): PIP = TV/Crs + R·V̇_peak + PEEP (componente resistivo incluido).
    tv = 397.0
    peep = 2.66
    pips = np.array([m.compute_pip(tv, peep) for m in models])
    assert 15.0 < pips.mean() < 20.0
    assert 3.0 < np.std(pips) < 7.0


# --------------------------------------------------------------------------
# M4: RR
# --------------------------------------------------------------------------

def test_e_rr_variability():
    # Camino de producción: la RR se sortea con el acoplamiento VCO2, la carga
    # respiratoria compartida y el peso (v7 C4).
    rng = np.random.default_rng(4)
    vals = []
    for _ in range(100_000):
        w = float(np.clip(rng.normal(70.0, 15.0), 35, 140))
        vco2 = resp.vco2_reference(w) * float(rng.lognormal(0.0, 0.12))
        load = float(rng.lognormal(0.0, 0.25))
        vals.append(resp.sample_rr(rng, vco2=vco2,
                                   vco2_ref=resp.vco2_reference(w), weight_kg=w,
                                   resp_load=load))
    vals = np.array(vals)
    assert 2.2 < np.std(vals) < 3.8            # sd real ~2.77 (antes ~0.55)
    assert 13.0 < vals.mean() < 16.0
    assert np.all(vals == np.round(vals))


def test_f_rr_compensates_vco2():
    rng = np.random.default_rng(5)
    hi = np.array([resp.sample_rr(rng, vco2=250.0, vco2_ref=165.0)
                   for _ in range(50_000)])
    rng = np.random.default_rng(5)
    lo = np.array([resp.sample_rr(rng, vco2=110.0, vco2_ref=165.0)
                   for _ in range(50_000)])
    assert hi.mean() > lo.mean()                # más CO2 -> más RR


# --------------------------------------------------------------------------
# M5: ETCO2
# --------------------------------------------------------------------------

def test_g_etco2_dispersion_and_range():
    rng = np.random.default_rng(6)
    ets = []
    for _ in range(3000):
        w = float(np.clip(rng.normal(70.0, 15.0), 35, 140))
        m = resp.RespiratoryModel(rng, weight_kg=w)
        rr = resp.sample_rr(rng, vco2=m.vco2,
                            vco2_ref=resp.vco2_reference(w), weight_kg=w,
                            resp_load=m.resp_load)
        et, _ = m.compute_etco2(6.3 * w, rr)
        ets.append(et)
    ets = np.array(ets)
    assert 30.0 < ets.mean() < 39.0
    assert 2.5 < np.std(ets) < 5.5            # sd real ~3.77 (antes ~6.26)
    assert ets.min() >= 0.0
    assert ets.max() <= 90.0 * 0.95


# --------------------------------------------------------------------------
# M6: cuantización
# --------------------------------------------------------------------------

def test_h_quantize_tracks_rounds_to_grid():
    sim_ = sim.CaseSimulator()
    tracks = {
        "Solar8000/HR": np.array([72.4, 72.7, np.nan, 73.1]),
        "BIS/BIS": np.array([40.14, 40.19, np.nan, 40.22]),
        "BIS/EMG": np.array([26.544, 26.551, np.nan, 26.556]),
        "time": np.array([0.0, 1.0, 2.0, 3.0]),  # no cuantizado
    }
    sim_._quantize_tracks(tracks)
    assert np.allclose(tracks["Solar8000/HR"][:2], [72.0, 73.0])
    assert np.isnan(tracks["Solar8000/HR"][2])
    assert np.allclose(tracks["BIS/BIS"][:2], [40.1, 40.2])
    assert np.allclose(tracks["BIS/EMG"][:2], [26.54, 26.55])
    assert tracks["time"][0] == 0.0


def test_i_quant_grid_covers_image_tracks():
    for t in ["BIS/BIS", "Solar8000/HR", "Solar8000/PLETH_SPO2",
              "Primus/ETCO2", "Primus/PEEP_MBAR", "Primus/PIP_MBAR",
              "Primus/MV", "Primus/TV", "Primus/RR_CO2", "Solar8000/BT",
              "Solar8000/ART_MBP", "Solar8000/ART_SBP", "Solar8000/ART_DBP",
              "BIS/EMG"]:
        assert t in sim.QUANT_GRID
    # Pasos: enteros para vitales, 0.1 para BIS/MV/BT, 0.01 para EMG.
    assert sim.QUANT_GRID["Solar8000/HR"] == 1.0
    assert sim.QUANT_GRID["BIS/BIS"] == 0.1
    assert sim.QUANT_GRID["Primus/MV"] == 0.1
    assert sim.QUANT_GRID["Solar8000/BT"] == 0.1
    assert sim.QUANT_GRID["BIS/EMG"] == 0.01


# --------------------------------------------------------------------------
# M7: topes
# --------------------------------------------------------------------------

def test_j_vco2_reference_allometric():
    assert resp.vco2_reference(70.0) == pytest.approx(165.0)
    assert resp.vco2_reference(35.0) < resp.vco2_reference(70.0)
    assert resp.vco2_reference(140.0) > resp.vco2_reference(70.0)
