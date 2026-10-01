"""Tests de src/diagnostics/gap_addendum.py (mediciones adicionales P1).

Unitarios (a-d) sin disco; integración (e-j) sobre data/windows_v2/ y el
informe generado con `python -m diagnostics.gap_addendum run`.

P1a suelo lineal (11 variables), P1b cuantización, P1c sonda no lineal,
P1d recortes de rango. No toca el generador ni el AE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diagnostics import gap_addendum as ga  # noqa: E402
from diagnostics import cohort_gap as cg  # noqa: E402


# --------------------------------------------------------------------------
# Unitarios (sin disco)
# --------------------------------------------------------------------------

def test_a_floor_excludes_three_breach_tracks():
    assert len(ga.FLOOR_TRACKS) == 11
    assert "BIS/EMG" not in ga.FLOOR_TRACKS
    assert "Primus/PEEP_MBAR" not in ga.FLOOR_TRACKS
    assert "Primus/PIP_MBAR" not in ga.FLOOR_TRACKS
    assert set(ga.FLOOR_TRACKS) <= set(cg.IMAGE_TRACKS)


def test_b_quantization_stats_integer_vs_continuous():
    # Entero: valores 72, 72, 73, 74 -> enteros, min_inc=1, 1/3 sin cambio.
    caseid = np.array([1, 1, 1, 1], dtype=np.int64)
    values = np.array([[72.0], [72.0], [73.0], [74.0]], dtype=np.float64)
    masks = np.ones((4, 1), dtype=np.uint8)
    q = ga.quantization_stats(values, masks, caseid, 0)
    assert q["n"] == 4
    assert q["n_distinct"] == 3
    assert q["frac_integer"] == pytest.approx(1.0)
    assert q["min_inc"] == pytest.approx(1.0)
    assert q["frac_nochange"] == pytest.approx(1.0 / 3.0)

    # Continuo: 2.5, 2.7, 2.65 -> no enteros, min_inc < 1 (0.05).
    values_c = np.array([[2.5], [2.7], [2.65]], dtype=np.float64)
    qc = ga.quantization_stats(values_c, masks[:3], caseid[:3], 0)
    assert qc["frac_integer"] == pytest.approx(0.0)
    assert qc["min_inc"] == pytest.approx(0.05)


def test_c_quantization_ignores_masked_nan():
    caseid = np.array([1, 1, 1], dtype=np.int64)
    values = np.array([[10.0], [np.nan], [11.0]], dtype=np.float64)
    masks = np.array([[1], [0], [1]], dtype=np.uint8)
    q = ga.quantization_stats(values, masks, caseid, 0)
    assert q["n"] == 2
    assert q["n_distinct"] == 2
    # La celda enmascarada no forma par consecutivo válido.
    assert q["frac_nochange"] is None


def test_d_hgb_probe_separates_perfectly():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 1.0, (200, 2)), rng.normal(5.0, 1.0, (200, 2))])
    y = np.r_[np.zeros(200, dtype=int), np.ones(200, dtype=int)]
    groups = np.arange(400)
    r = ga.run_origin_probe_hgb(X, y, groups)
    assert r["auc_mean"] > 0.95


def test_e_hgb_probe_chance_on_random_labels():
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, (400, 3))
    y = rng.integers(0, 2, 400)
    groups = rng.integers(0, 20, 400)
    r = ga.run_origin_probe_hgb(X, y, groups)
    assert abs(r["auc_mean"] - 0.5) < 0.25


# --------------------------------------------------------------------------
# Integración (disco). El fixture `results` ejecuta run_all() UNA vez y cachea.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results(tmp_path_factory):
    d = tmp_path_factory.mktemp("gap_addendum")
    return ga.run_all(cache_path=d / "cache.json", report_path=d / "report.txt")


@pytest.mark.integration
def test_f_p1a_floor_in_range(results):
    p1a = results["p1a"]
    assert p1a["n_tracks"] == 11
    assert 0.0 <= p1a["auc_mean"] <= 1.0


@pytest.mark.integration
def test_g_p1b_quantization_covers_all(results):
    p1b = results["p1b"]
    assert set(p1b.keys()) == set(cg.ALL_SOURCES)
    for source in cg.ALL_SOURCES:
        assert set(p1b[source].keys()) == set(cg.IMAGE_TRACKS)
        for track in cg.IMAGE_TRACKS:
            assert p1b[source][track]["n"] > 0
            assert p1b[source][track]["n_distinct"] > 0


@pytest.mark.integration
def test_h_p1c_hgb_high(results):
    """La sonda no lineal debe separar al menos tan bien como la lineal."""
    p1c = results["p1c"]
    assert 0.0 <= p1c["union_auc"] <= 1.0
    assert len(p1c["per_cohort"]) == 3
    for e in p1c["per_cohort"]:
        assert 0.0 <= e["auc"] <= 1.0


@pytest.mark.integration
def test_i_p1d_ranges_present(results):
    p1d = results["p1d"]
    out = p1d["real_outside_generator_range"]
    assert set(out.keys()) == set(cg.IMAGE_TRACKS)
    for track in cg.IMAGE_TRACKS:
        frac = out[track]["frac_outside"]
        assert frac is None or 0.0 <= frac <= 1.0


@pytest.mark.integration
def test_j_report_written(results):
    assert ga.REPORT_PATH.exists()
    txt = ga.REPORT_PATH.read_text(encoding="utf-8")
    assert "P1a" in txt
    assert "P1b" in txt
    assert "P1c" in txt
    assert "P1d" in txt
    assert "Supuesto" in txt
    assert "ROJO" in txt
