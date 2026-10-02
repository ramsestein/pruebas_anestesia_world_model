"""PASO 3 / C3 — test fijado contra manifests/tokens_v2_cf_pairs.json.

El JSON lo produce scripts/paso3_c3_cf_pairs.py. Este test bloquea:
  - prefijo idéntico en el 100 % de los pares utilizables;
  - efecto de palanca >= 99 % POR PALANCA.
"""

from __future__ import annotations

import json

import pytest

import paths


def _load():
    p = paths.MANIFESTS_DIR / "tokens_v2_cf_pairs.json"
    if not p.exists():
        pytest.skip("manifests/tokens_v2_cf_pairs.json no existe "
                    "(ejecuta scripts/paso3_c3_cf_pairs.py)")
    return json.loads(p.read_text(encoding="utf-8"))


def test_c3_n_pairs():
    r = _load()
    assert r["n_pairs"] == 6_345


def test_c3_prefix_identical_all_usable_pairs():
    r = _load()
    assert r["prefix_identical_all_pairs"], (
        f"{r['n_prefix_failures']} pares con prefijo divergente")


def test_c3_lever_effect_ge_99pct_per_lever():
    r = _load()
    assert r["effect_ge_99pct_all_levers"]
    table = r["per_lever"]
    assert len(table) == 21
    for row in table:
        assert row["pct"] >= 99.0, (
            f"palanca {row['lever']}: efecto {row['pct']:.2f} % < 99 %")
