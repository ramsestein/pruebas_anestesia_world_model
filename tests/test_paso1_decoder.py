"""Tests del PASO 1 — decoder congelado sobre v7 (FASES A y B).

Fijan la reproducción de B1 (diag 5b real, k=13) y de A1/B2 contra
manifests/paso1_decoder_v7.json con |diff| < 1e-6, para que cualquier cambio
futuro en la evaluación del AE falle en rojo.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import paths

# IMPORTANTE (Windows): importar ae.physio_ae (pyarrow) ANTES de torch para no
# romper pq.read_table.
from ae import physio_ae as pa  # noqa: E402

import torch  # noqa: E402  (ya importado por pa en el orden correcto)

CLINICAL = ["Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS", "Primus/ETCO2"]
V7 = ["synthetic_v7", "vaso_reinf_v7", "cf_v7"]


@pytest.fixture(scope="module")
def frozen():
    p = paths.MANIFESTS_DIR / "paso1_decoder_v7.json"
    assert p.is_file(), "ejecuta scripts/paso1_decoder_v7.py primero"
    return json.loads(p.read_text(encoding="utf-8"))


def _model_and_stats():
    device = torch.device("cpu")
    model = pa.load_model("ae_bal", "cpu").to(device)
    model.eval()
    norm_stats = pa.load_norm_stats("ae_bal")
    return model, norm_stats, device


@pytest.mark.integration
def test_b1_diag5b_real_reproduces(frozen):
    """B1: diag 5b real reproduce el manifest ae_bal (k_clinico = 13, +-0.0005)."""
    model, norm_stats, device = _model_and_stats()
    excluded = pa.excluded_caseids()
    val = pa.load_cells(["real"], "val", excluded)
    gates = pa.evaluate(model, norm_stats, val, device)
    d5b = gates["diag5b"]["cohorts"]["real"]
    assert d5b["k_clinico"] == 13

    ref = json.loads((paths.AE_DIR / "ae_bal" / "manifest_ae.json").read_text(
        encoding="utf-8"))["gates"]["diag5b"]["cohorts"]["real"]
    for v in pa.DIAG5B_VARS:
        a = np.asarray(d5b["per_variable"][v], dtype=np.float64)
        b = np.asarray(ref["per_variable"][v], dtype=np.float64)
        assert np.allclose(a, b, atol=0.0005, equal_nan=True), f"diag5b {v} no reproduce"

    assert frozen["B1_diag5b_real_reproduction"]["k_clinico"] == 13
    assert frozen["B1_diag5b_real_reproduction"]["reproduced"] is True


@pytest.mark.integration
def test_a1_gate2_and_b2_diag5b_reproduce(frozen):
    """A1 (gate2 v7) y B2 (diag5b v7 por cohorte y unión) reproducen el manifest."""
    model, norm_stats, device = _model_and_stats()
    excluded = pa.excluded_caseids()

    val = pa.load_cells(paths.ALL_COHORTS, "val", excluded)
    gates = pa.evaluate(model, norm_stats, val, device)

    # A1: gate 2, 14 variables x 4 cohortes
    for c in paths.ALL_COHORTS:
        for t in pa.IMAGE_TRACKS:
            got = float(gates["gate2"][c][t])
            ref = float(frozen["A1_gate2_v7"][c][t])
            assert abs(got - ref) < 1e-6, f"A1 {c}|{t}: {got} vs {ref}"

    # B2 unión (sintético v7)
    union = gates["diag5b"]["cohorts"]["synthetic"]
    refu = frozen["B2_diag5b_v7"]["union"]
    assert union["k_clinico"] == refu["k_clinico"]
    for v in pa.DIAG5B_VARS:
        a = np.asarray(union["per_variable"][v], dtype=np.float64)
        b = np.asarray(refu["per_variable"][v], dtype=np.float64)
        assert np.allclose(a, b, atol=1e-6), f"B2 union {v} no reproduce"

    # B2 por cohorte v7
    for c in V7:
        vc = pa.load_cells([c], "val", excluded)
        gc = pa.evaluate(model, norm_stats, vc, device)
        d5b = gc["diag5b"]["cohorts"]["synthetic"]
        refc = frozen["B2_diag5b_v7"]["cohorts"][c]
        assert d5b["k_clinico"] == refc["k_clinico"], f"{c} k_clinico"
        for v in pa.DIAG5B_VARS:
            a = np.asarray(d5b["per_variable"][v], dtype=np.float64)
            b = np.asarray(refc["per_variable"][v], dtype=np.float64)
            assert np.allclose(a, b, atol=1e-6), f"B2 {c}|{v} no reproduce"
