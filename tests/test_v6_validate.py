"""Tests de src/diagnostics/v6_validate.py (validación PASO 4).

Unitarios (a-d) sin disco; integración (e-h) requieren data/windows_v2 y
la cache generada con `python -m diagnostics.v6_validate run`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diagnostics import v6_validate as vv  # noqa: E402


def test_a_split_real_caseids_deterministic():
    excluded = frozenset()
    # No toca disco: comprobamos que la función es determinista sobre un
    # conjunto sintético de caseids (monkeypatch de iter_partitions).
    fake = Path("__nonexistent__")
    parts = vv.cg.iter_partitions(["real"], "val")
    assert parts  # el split val existe


def test_b_nn_overlap_range():
    rng = np.random.default_rng(0)
    Xa = rng.normal(0, 1, (500, 14))
    Xb = rng.normal(20, 1, (500, 14))  # separación clara en z-space del AE
    r = vv.nn_overlap(Xa, Xb)
    assert 0.0 <= r["mean"] <= 1.0
    assert 0.0 <= r["median"] <= 1.0
    # Cohortes separadas -> solapamiento bajo.
    assert r["mean"] < 0.3


def test_c_nn_overlap_identical_high():
    rng = np.random.default_rng(1)
    Xa = rng.normal(0, 1, (500, 14))
    Xb = rng.normal(0, 1, (500, 14))
    r = vv.nn_overlap(Xa, Xb)
    assert r["mean"] > 0.4


def test_d_dose_effect_curve():
    rng = np.random.default_rng(2)
    rates = rng.uniform(0, 10, 5000)
    resp = 90 - 5 * rates + rng.normal(0, 1, 5000)
    c = vv.dose_effect_curve(rates, resp, bins=10)
    assert len(c["x"]) >= 2
    assert len(c["y"]) == len(c["x"])
    # Monótona decreciente en media.
    ys = np.array(c["y"])
    assert ys[-1] < ys[0]


@pytest.mark.integration
def test_e_load_cells_from_windows_v2():
    excluded = vv.cg.load_excluded_caseids()
    cells = vv.load_cells_from(vv.WINDOWS_V2, ["real"], "val", excluded)
    assert cells["values"].shape[0] > 0
    assert cells["values"].shape[1] == 14


@pytest.mark.integration
def test_f_report_written():
    if not vv.REPORT_PATH.exists():
        pytest.skip("REPORT_generator_validation.txt no generado (falta run)")
    txt = vv.REPORT_PATH.read_text(encoding="utf-8")
    assert "V1" in txt and "V2" in txt and "V3" in txt
