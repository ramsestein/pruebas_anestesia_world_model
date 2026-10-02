"""PASO 3b — criterios de los pares contrafactuales (C1-C4) y adopción v2.

El manifiesto lo produce scripts/paso3b_annotate_cf_pairs.py y resume la
anotación de data/tokens_v2/pairs_annotated.parquet. Aquí se congelan los
criterios:

  C1  |t_divergence_raw - t_action| <= 30 s en todos los pares efectivos.
  C2  el efecto aparece en la ventana que contiene t_action o en la siguiente
      (effect_lag_windows <= 1) en >= 99 % de los pares efectivos.
  C3  ningún par con null_cause = "unexplained".
  C4  las ventanas anteriores a la que contiene t_action son idénticas entre
      ramas en TODAS las features y máscaras (salvo cf_role).

ALCANCE (documentado, no un atajo): C1 y C2 se evalúan sobre las ACCIONES
PUNTUALES (``one_shot``: eventos pharma/learning/sevo, con t_action en
``intervention_a``). Las palancas de CONSIGNA PERSISTENTE (los overrides de
ventilación: ``simulate.py`` aplica un delta acumulativo, ``peep_arr[post] +=
delta``) no cambian la señal observada en t_action sino cuando el plan base
cambia la consigna, de modo que la ventana de t_action no puede contener el
efecto por construcción. Su estadístico se reporta aparte en
``criteria.*.persistent_consigna`` y se verifica aquí que sigue la clase
correcta (n>0 y efecto presente).

Además se comprueba la adopción por artefacto (pk_v2, context_v2, tokens_v2) y
que pairs.parquet NO cambia de sha (R2).
"""

from __future__ import annotations

import hashlib
import json

import pytest

import paths
from tokens import cf_pairs as cp

NULL_CAUSES = {"clip_bound", "below_resolution", "no_change_requested",
               "case_ends", "unexplained"}
N_PAIRS = 6_345


def _load():
    p = paths.MANIFESTS_DIR / "tokens_v2_cf_pairs_annotation.json"
    if not p.exists():
        pytest.skip("manifests/tokens_v2_cf_pairs_annotation.json no existe "
                    "(ejecuta scripts/paso3b_annotate_cf_pairs.py)")
    return json.loads(p.read_text(encoding="utf-8"))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# estructura del manifiesto
# ---------------------------------------------------------------------------

def test_paso3b_n_pairs():
    r = _load()
    assert r["n_pairs"] == N_PAIRS
    assert r["n_effective"] + r["n_null"] == N_PAIRS


def test_paso3b_pairs_parquet_intacto():
    """R2: pairs.parquet no se reescribe; su sha debe coincidir con el anotado."""
    r = _load()
    assert r["pairs_parquet_sha256"] == _sha256(cp.PAIRS_PARQUET)


def test_paso3b_todas_las_palancas_en_el_manifiesto():
    from tokens import tokenize as tk
    r = _load()
    levers = {row["lever"] for row in r["per_lever"]}
    assert levers == set(tk.LEVER_GROUP)


# ---------------------------------------------------------------------------
# criterios C1-C4
# ---------------------------------------------------------------------------

def test_paso3b_c1_divergencia_cruda_cerca_de_t_action():
    r = _load()
    c1 = r["criteria"]["C1_divergencia_cruda_cerca_de_t_action"]
    assert c1["umbral_s"] == 30.0
    assert c1["n_effective_one_shot"] > 0
    assert c1["n_failures"] == 0, c1["failures"][:5]
    assert c1["ok"] is True
    # las consignas persistentes quedan fuera del alcance y se reportan
    pc = c1["persistent_consigna"]
    assert pc["n_effective"] > 0
    assert pc["n_failures"] > 0, (
        "si no hubiera fallos, C1 aplicaría también a las consignas "
        "persistentes y habría que revisar el alcance")


def test_paso3b_c2_efecto_en_ventana_de_t_action_o_siguiente():
    r = _load()
    c2 = r["criteria"]["C2_efecto_en_ventana_de_t_action_o_siguiente"]
    assert c2["umbral_lag"] == 1
    assert c2["n_effective_one_shot"] > 0
    assert c2["pct_lag_le_1"] >= 99.0, (
        f"sólo {c2['pct_lag_le_1']:.4f} % de las acciones puntuales tienen el "
        f"efecto en la ventana de t_action o la siguiente")
    assert c2["ok"] is True
    assert c2["persistent_consigna"]["n_effective"] > 0
    assert c2["persistent_consigna"]["pct_lag_le_1"] < c2["pct_lag_le_1"]


def test_paso3b_c3_sin_nulos_sin_explicar():
    r = _load()
    c3 = r["criteria"]["C3_sin_nulos_sin_explicar"]
    assert c3["n_unexplained"] == 0
    assert c3["ok"] is True
    assert r["null_causes"].get("unexplained", 0) == 0


def test_paso3b_c4_prefijo_identico_hasta_t_action():
    r = _load()
    c4 = r["criteria"]["C4_prefijo_identico_hasta_t_action"]
    assert c4["n_failures"] == 0, c4["failures"][:5]
    assert c4["ok"] is True


# ---------------------------------------------------------------------------
# nulos
# ---------------------------------------------------------------------------

def test_paso3b_causas_de_nulo_validas():
    r = _load()
    causas = set(r["null_causes"])
    assert causas <= NULL_CAUSES, f"causas desconocidas: {sorted(causas - NULL_CAUSES)}"
    assert sum(r["null_causes"].values()) == r["n_null"]


def test_paso3b_nulos_por_palanca_suman():
    r = _load()
    total = sum(sum(row["nulls"].values()) for row in r["per_lever"])
    assert total == r["n_null"]


def test_paso3b_nulos_no_peep_documentados():
    """Los nulos que no son recorte de PEEP quedan documentados con evidencia."""
    r = _load()
    nn = r["nulls_non_peep"]
    assert r["n_nulls_non_peep"] == len(nn)
    for e in nn:
        assert e["null_cause"] in NULL_CAUSES
        assert e["null_cause"] != "clip_bound"
        # evidencia mínima por par
        assert "requested_change" in e and e["requested_change"]
        assert "truth_diff_after_action" in e
        assert e["t_action"] is not None


def test_paso3b_clip_bound_es_peeP_con_base_baja():
    """clip_bound es, por construcción, PEEP con base 0 (clip [0,25])."""
    r = _load()
    por_palanca = {row["lever"]: row["nulls"] for row in r["per_lever"]}
    clip = {lv: d.get("clip_bound", 0) for lv, d in por_palanca.items()
            if d.get("clip_bound", 0)}
    assert set(clip) <= {"peep_down", "set_peep", "peep_up", "ppf20_rate"}, clip
    assert clip.get("peep_down", 0) > 0 and clip.get("set_peep", 0) > 0
    assert r["null_causes"].get("clip_bound", 0) == sum(clip.values())


def test_paso3b_below_resolution_es_tasa_o_cuantizacion():
    """below_resolution: tasas sobrescritas y setpoints por debajo del escalón."""
    r = _load()
    por_palanca = {row["lever"]: row["nulls"] for row in r["per_lever"]}
    br = {lv: d.get("below_resolution", 0) for lv, d in por_palanca.items()
          if d.get("below_resolution", 0)}
    assert set(br) <= {"rftn20_rate", "ppf20_rate", "remi_up", "set_rr",
                       "set_peep", "set_tv"}, br
    assert r["null_causes"]["below_resolution"] == sum(br.values())
    for lv in ("rftn20_rate", "ppf20_rate", "remi_up", "set_rr"):
        assert br.get(lv, 0) > 0, lv
    # todos los nulos no-PEEP son below_resolution (verdad divergente) o
    # no_change_requested (petición ≈ base)
    for e in r["nulls_non_peep"]:
        assert e["null_cause"] in ("below_resolution", "no_change_requested"), e
        if e["null_cause"] == "below_resolution":
            assert e["truth_diff_after_action"] is True, e


# ---------------------------------------------------------------------------
# adopción por artefacto
# ---------------------------------------------------------------------------

def test_paso3b_adopcion_pk_context_tokens_v2():
    assert paths.PK_DIR == paths.PK_V2_DIR
    assert paths.CONTEXT_DIR == paths.CONTEXT_V2_DIR
    assert paths.TOKENS_DIR == paths.TOKENS_V2_DIR


def test_paso3b_conjunto_cf_son_los_pares_efectivos():
    r = _load()
    if not cp.PAIRS_ANNOTATED_PARQUET.exists():
        pytest.skip("pairs_annotated.parquet no existe")
    df = cp.load_pairs_annotated()
    ids = cp.effective_pair_ids(df)
    assert len(ids) == r["n_effective"]
    assert len(df) - len(ids) == r["n_null"]
