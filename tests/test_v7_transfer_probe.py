"""Tests de src/diagnostics/v7_transfer_probe.py (sonda reparada, PASO 1).

Unitarios (a-e) sin disco; integración (f-h) sobre la cache generada con
`python -m diagnostics.v7_transfer_probe run --cohort v7`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diagnostics import v7_transfer_probe as tp  # noqa: E402


# --------------------------------------------------------------------------
# Unitarios (sin disco)
# --------------------------------------------------------------------------

def test_a_scaler_fit_transform_inverse():
    X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    sc = tp.Scaler.fit(X)
    assert sc.std.shape == (2,)
    assert np.allclose(sc.transform(X).mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(sc.inverse_transform(sc.transform(X)), X, atol=1e-8)


def test_b_scaler_zero_std_becomes_one():
    X = np.ones((5, 2))
    sc = tp.Scaler.fit(X)
    assert sc.std[0] == 1.0  # std 0 -> 1.0 (sin división por cero)


def test_c_ratios_vs_persistence():
    mae = {"a": 1.5, "b": 0.5}
    base = {"a": 1.0, "b": 1.0}
    r = tp._ratios(mae, base)
    assert r["a"] == pytest.approx(1.5)
    assert r["b"] == pytest.approx(0.5)


def test_d_residual_prediction_matches_contract():
    """La predicción final es x_t + delta (contrato z_{t+1} = z_t + f(·))."""
    torch = pytest.importorskip("torch")
    model = tp.MLP(tp.cg.N_VARS + tp.N_CONTROL, (8,), tp.cg.N_VARS)
    # Pesos a cero + bias cero -> delta predicho 0 -> predicción = x_t.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    sx = tp.Scaler(np.zeros(tp.cg.N_VARS + tp.N_CONTROL),
                   np.ones(tp.cg.N_VARS + tp.N_CONTROL))
    sy = tp.Scaler(np.zeros(tp.cg.N_VARS), np.ones(tp.cg.N_VARS))
    x14 = np.full(tp.cg.N_VARS, 7.0)
    ctrl = np.zeros(tp.N_CONTROL)
    pred = tp._predict_single(model, x14, ctrl, sx, sy)
    assert np.allclose(pred, x14, atol=1e-5)  # delta 0 -> persistencia


def test_e_sanity_variables_subset_of_moving():
    assert set(tp.SANITY_VARS) <= set(tp.MOVING_VARS)
    assert tp.HORIZONS == [1, 12, 60]
    assert tp.MOVING_VARS == ["Solar8000/HR", "Solar8000/ART_MBP",
                              "Solar8000/ART_SBP", "BIS/BIS",
                              "Primus/ETCO2"]


# --------------------------------------------------------------------------
# Integración (sobre cache v7)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results():
    if not tp.CACHE_PATH.exists():
        pytest.skip("cache de la sonda no generada (falta run)")
    return json.loads(tp.CACHE_PATH.read_text(encoding="utf-8"))


@pytest.mark.integration
def test_f_finetuning_starts_from_d1_weights(results):
    # 1.4: los pesos iniciales de D3 son los finales de D1.
    assert results["d3_init_max_abs_diff_vs_d1"] == pytest.approx(0.0, abs=1e-9)
    assert results["d3_init_params_norm"] == pytest.approx(
        results["d1_final_params_norm"], rel=1e-9)


@pytest.mark.integration
def test_g_epoch0_loss_matches_d1(results):
    # 1.4: la pérdida de D3 en la época 0 coincide con la de D1 sobre calib.
    assert results["d1_loss_on_calib_val"] == pytest.approx(
        results["d3_epoch0_loss_on_calib_val"], abs=1e-6)


@pytest.mark.integration
def test_h_sanity_and_report(results):
    # 1.5: control de cordura presente y con las variables exigidas.
    assert set(results["sanity"]["per_var"]) == {"Solar8000/HR",
                                                  "Solar8000/ART_MBP", "BIS/BIS"}
    assert "passed" in results["sanity"]
    assert tp.REPORT_PATH.exists()
    txt = tp.REPORT_PATH.read_text(encoding="utf-8")
    assert "CONTROL DE CORDURA" in txt


@pytest.mark.integration
def test_i_v9_conditions_present(results):
    # v9: D2' (control de presupuesto) y D3' (escalador objetivo real).
    assert "d2prime_mae" in results
    assert "d3prime_mae" in results
    assert "d2prime_training" in results and "d3prime_training" in results
    dec = results["decision"]
    assert "transfer_confirmed" in dec
    assert "d3_wins_vs_d2prime" in dec
    # D2' usa el mismo presupuesto que el fine-tune de D3 (lr 1e-4, tope 56).
    assert results["d2prime_training"]["lr"] == tp.LR_D2_PRIME


@pytest.mark.integration
def test_j_v9_report_has_transfer_verdict(results):
    txt = tp.REPORT_PATH.read_text(encoding="utf-8")
    assert "TRANSFERENCIA CONFIRMADA" in txt
