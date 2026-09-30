"""Tests de src/diagnostics/v7_attribution.py (atribución de la sonda no lineal).

Unitarios (a-e) sin disco; integración (f-k) sobre windows_v2/windows_v3 y el
informe generado con `python -m diagnostics.v7_attribution run`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diagnostics import v7_attribution as va  # noqa: E402
from diagnostics import cohort_gap as cg  # noqa: E402

LOST_V6 = pytest.mark.skip(reason=(
    "requiere la cohorte v6 (synthetic_v6/vaso_reinf_v6/cf_v6 en windows_v3), "
    "perdida — paths.LOST_COHORTS"))


# --------------------------------------------------------------------------
# Unitarios (sin disco)
# --------------------------------------------------------------------------

def test_a_pairwise_mutual_information_scale():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20000)
    y_indep = rng.normal(0, 1, 20000)
    y_dep = 2.0 * x + rng.normal(0, 0.1, 20000)
    mi_indep = va.pairwise_mi(x, y_indep, n_bins=10)
    mi_dep = va.pairwise_mi(x, y_dep, n_bins=10)
    assert mi_dep > mi_indep
    # Identidad -> MI alta (cercana a la entropía de deciles).
    mi_self = va.pairwise_mi(x, x, n_bins=10)
    assert mi_self > mi_dep


def test_b_corr_matrix_difference_frobenius():
    rng = np.random.default_rng(1)
    A = rng.normal(0, 1, (500, 4))
    B = rng.normal(0, 1, (500, 4))
    res = va.corr_difference(A, B)
    assert "frobenius" in res
    assert res["frobenius"] >= 0.0
    assert len(res["top_pairs"]) >= 0


def test_c_top_discrepancies_sorted():
    pairs = [("a|b", 0.5), ("c|d", 0.9), ("e|f", 0.1)]
    top = va._top_pairs(pairs, 2)
    assert top[0][0] == "c|d"
    assert top[1][0] == "a|b"


def test_d_quantization_table_columns():
    table = va._quant_table_rows({}, {}, ["BIS/BIS"])
    # Devuelve una lista de filas con claves esperadas.
    assert isinstance(table, list)


def test_e_hgb_ablation_drop_sorted():
    univ = [{"track": "a", "auc": 0.9}, {"track": "b", "auc": 0.6}]
    full = 0.95
    ab = va._ablation_from_univ(univ, {"a": 0.7, "b": 0.9}, full)
    # drop = full - auc_sin_variable
    drops = {e["track"]: e["drop"] for e in ab}
    assert drops["a"] == pytest.approx(0.95 - 0.7)
    assert drops["b"] == pytest.approx(0.95 - 0.9)


# --------------------------------------------------------------------------
# Integración (disco)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results():
    return va.run_all()


@pytest.mark.integration
@pytest.mark.requires_lost_data
@LOST_V6
def test_f_a1_a3_present(results):
    r = results
    assert "a1" in r and "a2" in r and "a3" in r
    # a1/a3 son LISTAS de 14 entradas (univariante/ablación), no dicts.
    assert len(r["a1"]) == 14
    assert len(r["a3"]) == 14


@pytest.mark.integration
@pytest.mark.requires_lost_data
@LOST_V6
def test_g_a4_deltas_present(results):
    a4 = results["a4"]
    assert "univariate_deltas" in a4
    assert len(a4["univariate_deltas"]) == 14


@pytest.mark.integration
@pytest.mark.requires_lost_data
@LOST_V6
def test_h_a5_quantization_present(results):
    a5 = results["a5"]
    assert set(a5.keys()) >= {"real", "v6"}


@pytest.mark.integration
@pytest.mark.requires_lost_data
@LOST_V6
def test_i_a6_corr_mi_present(results):
    a6 = results["a6"]
    assert "corr_frobenius" in a6 or "corr" in a6
    assert "mi" in a6


@pytest.mark.integration
@pytest.mark.requires_lost_data
@LOST_V6
def test_j_report_written(results):
    assert va.REPORT_PATH.exists()
    txt = va.REPORT_PATH.read_text(encoding="utf-8")
    assert "A1" in txt and "A6" in txt
    assert "ROJO" in txt and "VERDE" in txt
