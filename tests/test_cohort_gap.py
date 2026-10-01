"""Tests de src/diagnostics/cohort_gap.py (caracterización de la brecha
real/sintético).

Unitarios (a-f) sin disco; integración (g-m) sobre data/windows_v2/ y el
informe generado con `python -m diagnostics.cohort_gap run`.

El sondeo replica el protocolo del gate 6 del AE (submuestra de 200 000 celdas
con semilla 9876, GroupKFold 5 por caseid, estandarización por fold,
LogisticRegression), pero sobre la ENTRADA CRUDA. No toca el AE.
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

from diagnostics import cohort_gap as cg  # noqa: E402

import paths  # noqa: E402


# --------------------------------------------------------------------------
# Unitarios (sin disco)
# --------------------------------------------------------------------------

def test_a_column_means_ignores_masked():
    values = np.array([[1.0, 10.0], [3.0, 30.0], [np.nan, 50.0]], dtype=np.float64)
    masks = np.array([[1, 1], [1, 1], [0, 1]], dtype=np.uint8)
    means = cg.column_means(values, masks)
    assert means[0] == pytest.approx(2.0)   # (1+3)/2
    assert means[1] == pytest.approx(30.0)  # (10+30+50)/3


def test_b_impute_with_fills_masked():
    values = np.array([[1.0], [np.nan], [3.0]], dtype=np.float64)
    masks = np.array([[1], [0], [1]], dtype=np.uint8)
    out = cg.impute_with(values, masks, np.array([2.0]))
    assert out[1, 0] == pytest.approx(2.0)
    assert not np.isnan(out).any()


def test_c_sample_two_groups_balanced():
    n = 10_000
    caseid = np.repeat(np.arange(100), 100)
    is_a = np.zeros(n, dtype=bool)
    is_a[: n // 2] = True
    universe = np.ones(n, dtype=bool)
    idx, y, groups = cg.sample_two_groups(is_a, universe, caseid, half=500, seed=9876)
    assert len(idx) == 1000
    assert int(y.sum()) == 500
    assert int((y == 0).sum()) == 500
    assert np.array_equal(groups, caseid[idx])


def test_d_deltas_first_cell_imputed():
    caseid = np.array([1, 1, 1, 2, 2], dtype=np.int64)
    values = np.array([[10.0], [20.0], [30.0], [100.0], [105.0]], dtype=np.float64)
    masks = np.ones((5, 1), dtype=np.uint8)
    dmeans = cg.delta_means(values, masks, caseid)
    # deltas válidos: (20-10)=10, (30-20)=10, (105-100)=5 -> media 25/3
    assert dmeans[0] == pytest.approx(25.0 / 3.0)
    d = cg.deltas_for_indices(np.array([1, 3]), values, masks, caseid, dmeans)
    assert d[0, 0] == pytest.approx(10.0)          # celda 1: tiene celda anterior
    assert d[1, 0] == pytest.approx(25.0 / 3.0)    # celda 3: primera de su caso -> imputada


def test_e_probe_separates_perfectly():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 1.0, (200, 2)), rng.normal(5.0, 1.0, (200, 2))])
    y = np.r_[np.zeros(200, dtype=int), np.ones(200, dtype=int)]
    groups = np.arange(400)
    r = cg.run_origin_probe(X, y, groups)
    assert r["auc_mean"] > 0.95


def test_f_probe_chance_on_random_labels():
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, (400, 3))
    y = rng.integers(0, 2, 400)
    groups = rng.integers(0, 20, 400)
    r = cg.run_origin_probe(X, y, groups)
    assert abs(r["auc_mean"] - 0.5) < 0.2


# --------------------------------------------------------------------------
# Integración (disco). El fixture `results` ejecuta run_all() UNA vez y cachea.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results(tmp_path_factory):
    d = tmp_path_factory.mktemp("cohort_gap")
    return cg.run_all(cache_path=d / "cache.json", report_path=d / "report.txt")


@pytest.mark.integration
def test_g_load_cells_reads_val():
    excluded = cg.load_excluded_caseids()
    assert len(excluded) >= 36
    parts = cg.iter_partitions(["real"], "val")
    assert parts
    df = cg.read_partition(parts[0], cg.READ_COLS)
    df = cg.filter_cells(df, excluded)
    assert set(cg.VALUE_COLS) <= set(df.columns)
    assert set(cg.MASK_COLS) <= set(df.columns)


@pytest.mark.integration
def test_h_d3_controls_near_chance(results):
    """Los controles de cordura DEBEN dar AUC ~0.5; si no, el protocolo está roto."""
    d3 = results["d3"]
    assert d3["a_train_vs_val"]["auc_mean"] < 0.6
    assert d3["b_half_split"]["auc_mean"] < 0.6


@pytest.mark.integration
def test_i_d1_full14_reproduces_manifest(results):
    """Reproducción: una ejecución nueva reproduce el AUC de 14 variables
    registrado en manifests/cohort_gap_v7_results.json con |diff| < 1e-6, y la
    estructura de D1 (univariante, acumulativo, ablación) está completa.

    NOTA: el 0.9734 de v5 (protocolo cohort_gap, val completo) y el 0.7654 de
    v7 (V1, real_holdout) son protocolos DISTINTOS y no se comparan.
    """
    ref = json.loads((paths.MANIFESTS_DIR / "cohort_gap_v7_results.json")
                     .read_text(encoding="utf-8"))
    d1 = results["d1"]
    assert abs(d1["cumulative"][-1]["auc"] - ref["d1"]["cumulative"][-1]["auc"]) < 1e-6
    assert len(d1["univariate"]) == 14
    assert len(d1["cumulative"]) == 14
    assert len(d1["ablation"]) == 14
    assert "full14_auc" in d1


@pytest.mark.integration
def test_j_d2_three_cohorts(results):
    d2 = results["d2"]
    assert len(d2["real_vs_cohort"]) == 3
    assert len(d2["synth_pairs"]) == 3
    for e in d2["real_vs_cohort"] + d2["synth_pairs"]:
        assert 0.0 <= e["auc"] <= 1.0


@pytest.mark.integration
def test_k_d4_wasserstein_finite(results):
    d4 = results["d4"]
    ws = d4["wasserstein"]
    assert set(ws.keys()) == set(cg.SYNTH_SOURCES)
    for cohort in ws:
        assert set(ws[cohort].keys()) == set(cg.IMAGE_TRACKS)
        for track in cg.IMAGE_TRACKS:
            w = ws[cohort][track]["w1_norm"]
            assert np.isfinite(w) and w >= 0.0


@pytest.mark.integration
def test_l_d5_deltas_present(results):
    d5 = results["d5"]
    assert 0.0 <= d5["deltas_only"]["auc_mean"] <= 1.0
    assert 0.0 <= d5["values_and_deltas"]["auc_mean"] <= 1.0


@pytest.mark.integration
def test_m_report_written(results):
    assert cg.REPORT_PATH.exists()
    txt = cg.REPORT_PATH.read_text(encoding="utf-8")
    assert "D6" in txt
    assert "Supuesto" in txt
    assert "ROJO" in txt
