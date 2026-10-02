"""PASO 3 / C1 — test fijado contra manifests/tokens_v2_reproduccion_real.json.

El JSON lo produce scripts/paso3_c1_reproduce.py (reproducción a escala completa
de la cohorte real con los estadísticos congelados de tokens_v1). Este test
bloquea los criterios C1.
"""

from __future__ import annotations

import json

import pytest

import paths


def _load():
    p = paths.MANIFESTS_DIR / "tokens_v2_reproduccion_real.json"
    if not p.exists():
        pytest.skip("manifests/tokens_v2_reproduccion_real.json no existe "
                    "(ejecuta scripts/paso3_c1_reproduce.py)")
    return json.loads(p.read_text(encoding="utf-8"))


def test_c1_counts_real():
    r = _load()
    tr = r["splits"]["train"]
    va = r["splits"]["val"]
    assert tr["n_rows_old"] == 734_663
    assert va["n_rows_old"] == 1_501_733
    assert va["n_dense_old"] == 1_376_544
    assert r["verdict"]["counts_match"]


def test_c1_exact_columns_identical():
    r = _load()
    assert r["verdict"]["exact_ok"], (
        "columnas exactas con |diff| >= 1e-6")
    for split in ("train", "val"):
        assert r["splits"][split]["max_abs_diff_exact"] < 1e-6


def test_c1_ctx_binary_identical():
    r = _load()
    assert r["verdict"]["ctx_binary_ok"]
    assert r["splits"]["train"]["ctx_binary_mismatch"] == 0
    assert r["splits"]["val"]["ctx_binary_mismatch"] == 0


def test_c1_ctx_continuous_affine():
    r = _load()
    assert r["verdict"]["ctx_affine_ok"]
    for split in ("train", "val"):
        aff = r["splits"][split]["ctx_continuous_affine"]
        for item in ("age", "height", "weight", "bmi"):
            a = aff[item]
            assert a["n"] > 0, f"{item}: sin datos"
            assert a["max_residual"] < 1e-6, f"{item}: residuo {a['max_residual']}"
            # una sola escala y un solo desplazamiento (afín, no trivial)
            assert abs(a["scale"]) > 1e-9, f"{item}: escala degenerada"
