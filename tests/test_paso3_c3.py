"""PASO 3 / C3 — test fijado contra manifests/tokens_v2_cf_pairs.json.

El JSON lo produce scripts/paso3_c3_cf_pairs.py. Este test bloquea:
  - prefijo idéntico en el 100 % de los pares utilizables;
  - REPRODUCCIÓN de la tabla histórica por palanca del paso 3.

Nota (paso 3b): el criterio «la palanca tiene efecto» del paso 3 medía el
efecto en la ventana que contiene ``split_t`` (post_split = t1 > split_t),
cuando el generador aplica las intervenciones farmacológicas 10 s después
(``t_action = split_t + 10``). Ese criterio queda SUSTITUIDO por el del paso
3b (``post_action = t1 > t_action``, ventana que contiene ``t_action`` o la
siguiente), verificado sobre ``data/tokens_v2/pairs_annotated.parquet`` en
``tests/test_paso3b_criterios.py`` (C1-C4). Aquí sólo se congela, como
REPRODUCCIÓN, el artefacto histórico del paso 3 para que no cambie.
"""

from __future__ import annotations

import json

import pytest

import paths

# Tabla histórica del paso 3 (criterio antiguo, post_split = t1 > split_t):
# lever -> (usable, effect, pct, effect_any, pct_any). Congelada para
# reproducir el artefacto; NO es el criterio vigente (ver docstring).
PER_LEVER_PASO3 = {
    "eph_bolus": (375, 345, 92.0, 375, 100.0),
    "ephedrine": (120, 106, 88.3333, 120, 100.0),
    "fio2_down": (150, 144, 96.0, 150, 100.0),
    "nepi_rate": (375, 319, 85.0667, 375, 100.0),
    "noradrenaline": (250, 205, 82.0, 250, 100.0),
    "peep_down": (150, 67, 44.6667, 93, 62.0),
    "peep_up": (150, 133, 88.6667, 150, 100.0),
    "phen_bolus": (375, 348, 92.8, 375, 100.0),
    "phen_rate": (375, 349, 93.0667, 375, 100.0),
    "ppf20_rate": (375, 308, 82.1333, 371, 98.9333),
    "ppf_bolus": (375, 310, 82.6667, 375, 100.0),
    "propofol_bolus": (250, 203, 81.2, 250, 100.0),
    "remi_bolus": (375, 324, 86.4, 375, 100.0),
    "remi_up": (250, 200, 80.0, 247, 98.8),
    "rftn20_rate": (375, 298, 79.4667, 367, 97.8667),
    "set_fio2": (375, 355, 94.6667, 375, 100.0),
    "set_peep": (375, 231, 61.6, 291, 77.6),
    "set_rr": (375, 317, 84.5333, 363, 96.8),
    "set_tv": (375, 335, 89.3333, 375, 100.0),
    "sevo_mac": (375, 290, 77.3333, 375, 100.0),
    "sevo_up": (150, 117, 78.0, 150, 100.0),
}


def _load():
    p = paths.MANIFESTS_DIR / "tokens_v2_cf_pairs.json"
    if not p.exists():
        pytest.skip("manifests/tokens_v2_cf_pairs.json no existe "
                    "(ejecuta scripts/paso3_c3_cf_pairs.py)")
    return json.loads(p.read_text(encoding="utf-8"))


def test_c3_n_pairs():
    r = _load()
    assert r["n_pairs"] == 6_345
    assert r["n_usable"] == 6_345
    assert r["n_unusable"] == 0


def test_c3_prefix_identical_all_usable_pairs():
    r = _load()
    assert r["prefix_identical_all_pairs"], (
        f"{r['n_prefix_failures']} pares con prefijo divergente")


def test_c3_lever_effect_reproduccion_historica():
    """Reproduce la tabla por palanca del paso 3 (criterio antiguo).

    Congela el artefacto para detectar cualquier reescritura del JSON. El
    criterio vigente es C1-C4 de tests/test_paso3b_criterios.py.
    """
    r = _load()
    table = r["per_lever"]
    assert len(table) == 21
    assert {row["lever"] for row in table} == set(PER_LEVER_PASO3)
    for row in table:
        esperado = PER_LEVER_PASO3[row["lever"]]
        obtenido = (row["usable"], row["effect"], row["pct"],
                    row["effect_any"], row["pct_any"])
        assert obtenido == esperado, (
            f"palanca {row['lever']}: {obtenido} != {esperado}")
    # el criterio antiguo NO se cumplía: queda registrado tal cual
    assert r["effect_ge_99pct_all_levers"] is False
