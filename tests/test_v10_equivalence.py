"""Tests de cierre de integridad (v10): equivalencia por contenido.

Verifica que la reejecución de los gates 2/3 del ae_bal congelado sobre la
cohorte real de windows_v4 y los marginales split=val reproducen los valores
registrados (manifest ae_bal y cache cohort_gap). No abre diagnósticos nuevos:
solo valida los artefactos JSON generados por scripts/v10_gate_recheck.py y
scripts/v10_marginales_val.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA = Path("data/diagnostics")


def test_a_gate2_real_matches_manifest():
    chk = json.load(open(DATA / "v10_gate_recheck.json", encoding="utf-8"))
    m = json.load(open("data/ae_v1/ae_bal/manifest_ae.json", encoding="utf-8"))
    ref = m["gates"]["gate2"]["real"]
    for t in ["Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS", "Primus/ETCO2"]:
        assert chk["gate2_real"][t] == pytest.approx(ref[t], abs=1e-3)


def test_b_gate3_matches_manifest():
    chk = json.load(open(DATA / "v10_gate_recheck.json", encoding="utf-8"))
    m = json.load(open("data/ae_v1/ae_bal/manifest_ae.json", encoding="utf-8"))
    ref = m["gates"]["gate3"]
    assert chk["gate3"]["n_cases"] == ref["n_cases"]
    assert chk["gate3"]["n_cells"] == ref["n_cells"]
    assert chk["gate3"]["err_hr_with_art"] == pytest.approx(
        ref["err_hr_with_art"], abs=1e-3)
    assert chk["gate3"]["err_hr_without_art"] == pytest.approx(
        ref["err_hr_without_art"], abs=1e-3)


def test_c_marginales_val_match_d4():
    chk = json.load(open(DATA / "v10_marginales_val.json", encoding="utf-8"))
    assert chk["max_rel_dstd"] < 1e-4
    assert chk["max_abs_dmean"] < 1e-4


def test_d_exclusiones_son_36():
    m = json.load(open("data/tokens_v1/manifest_tokens.json", encoding="utf-8"))
    s = m["sin_celdas_caseids"]
    assert s == [4476]
    assert len(s) == 1
    assert len(m["excluded_no_phase_marks_caseids"]) + len(s) == 36
