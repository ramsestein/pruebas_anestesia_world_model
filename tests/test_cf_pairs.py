"""PASO 3b — tests de src/tokens/cf_pairs.py (módulo de anotación CF).

Cubren las piezas puras (t_action, post_action, conjunto CF efectivo) y la
coherencia del artefacto ``data/tokens_v2/pairs_annotated.parquet``.
"""

from __future__ import annotations

import pytest

import paths
from tokens import cf_pairs as cp

REQUIRED_COLS = [
    "t_action", "action_source", "t_divergence_raw", "requested_change",
    "base_value_at_action", "effect_t1", "effect_lag_windows",
    "lever_effective", "null_cause",
]
NULL_CAUSES = {"clip_bound", "below_resolution", "no_change_requested",
               "case_ends", "unexplained"}


# ---------------------------------------------------------------------------
# t_action
# ---------------------------------------------------------------------------

def test_parse_action_time():
    assert cp.parse_action_time("t=4648.7s INFUSION_CHANGE remifentanil 12.4mcg/min") == 4648.7
    assert cp.parse_action_time("t=12345.0s BOLUS propofol 80.0mg") == 12345.0
    with pytest.raises(ValueError):
        cp.parse_action_time("sin instante")


def test_action_time_metadata_para_farmaco():
    meta = {"intervention_a": ["t=14364.7s INFUSION_CHANGE remifentanil 16.1mcg/min"]}
    t, src = cp.action_time(meta, "remi_up", split_t=14354.7)
    assert t == pytest.approx(14364.7)
    assert src == "metadata"
    # el generador aplica la acción 10 s después del split
    assert t - 14354.7 == pytest.approx(10.0)


def test_action_time_codigo_para_override_ventilatorio():
    # peep_down no registra intervention_a: t_action = split_t (simulate.py).
    meta = {"intervention_a": [], "vent_override_a": {"peep_delta": -5.0}}
    t, src = cp.action_time(meta, "peep_down", split_t=1234.5)
    assert t == pytest.approx(1234.5)
    assert src == cp.VENT_ACTION_SOURCE
    assert "simulate.py" in src


# ---------------------------------------------------------------------------
# post_action (criterio corregido, refinado a la rejilla de tokens)
# ---------------------------------------------------------------------------

def test_action_grid_time_baja_a_la_rejilla_de_5s():
    assert cp.GRID_S == 5.0
    assert cp.action_grid_time(14364.7) == 14360.0
    assert cp.action_grid_time(6781.7) == 6780.0
    assert cp.action_grid_time(6780.0) == 6780.0


def test_post_action_la_primera_ventana_es_la_del_punto_de_rejilla():
    # t_action = 100.2 s -> punto de rejilla 100.0 s
    assert cp.post_action(100.0, 100.2) is True    # cierra en el punto de rejilla
    assert cp.post_action(95.0, 100.2) is False    # cierra en el punto anterior
    assert cp.post_action(160.0, 100.2) is True


def test_post_action_la_ventana_que_contiene_t_action_cuenta():
    # ventana (t0=60, t1=120) que contiene t_action=100: es post-acción
    assert cp.post_action(120.0, 100.0) is True
    # la anterior (t1=60) queda en el prefijo
    assert cp.post_action(60.0, 100.0) is False


def test_post_action_excluye_la_divergencia_anticipada_de_pk_v2():
    """pk_v2 cuantiza a 5 s: una acción en 6781.7 aparece ya en la rejilla 6780.

    La ventana que cierra en 6780 debe contar como post-acción (si no, su
    divergencia quedaría en el prefijo y rompería C4).
    """
    t_act = 6781.7
    assert cp.post_action(6780.0, t_act) is True
    assert cp.post_action(6775.0, t_act) is False


def test_pre_action_margen_de_dos_celdas():
    """El prefijo de C4 se queda 2 celdas por detrás (atribución temprana)."""
    assert cp.PREFIX_MARGIN_CELLS == 2
    t_act = 3125.2                                        # grid = 3125
    assert cp.pre_action(3120.0, t_act) is False          # 3120 > 3115
    assert cp.pre_action(3115.0, t_act) is True           # borde: 3115 <= 3115
    assert cp.pre_action(3060.0, t_act) is True


def test_pre_action_y_post_action_cubren_la_accion():
    t_act = 6781.7                                        # grid = 6780
    assert cp.pre_action(6775.0, t_act) is False          # fuera del prefijo
    assert cp.post_action(6780.0, t_act) is True          # dentro de post-acción


def test_is_persistent():
    assert cp.is_persistent("peep_down")
    assert cp.is_persistent("set_peep")
    assert not cp.is_persistent("rftn20_rate")
    assert not cp.is_persistent("sevo_mac")
    assert cp.PERSISTENT_LEVERS == cp.VENT_OVERRIDE_LEVERS


# ---------------------------------------------------------------------------
# cobertura de mapas
# ---------------------------------------------------------------------------

def test_lever_tracks_cubre_los_grupos_de_lever_group():
    from tokens import tokenize as tk
    grupos = set(tk.LEVER_GROUP.values())
    assert grupos <= set(cp.LEVER_TRACKS), (
        f"grupos sin tracks: {sorted(grupos - set(cp.LEVER_TRACKS))}")


def test_lever_features_cubre_los_grupos():
    from tokens import tokenize as tk
    grupos = set(tk.LEVER_GROUP.values())
    assert grupos <= set(cp.LEVER_FEATURES)


def test_vent_override_levers_son_palancas_conocidas():
    from tokens import tokenize as tk
    assert cp.VENT_OVERRIDE_LEVERS <= set(tk.LEVER_GROUP)


# ---------------------------------------------------------------------------
# artefacto pairs_annotated
# ---------------------------------------------------------------------------

def _annotated():
    p = cp.PAIRS_ANNOTATED_PARQUET
    if not p.exists():
        pytest.skip("data/tokens_v2/pairs_annotated.parquet no existe "
                    "(ejecuta scripts/paso3b_annotate_cf_pairs.py)")
    return cp.load_pairs_annotated()


def test_pairs_annotated_columnas_requeridas():
    df = _annotated()
    faltan = [c for c in REQUIRED_COLS if c not in df.columns]
    assert not faltan, f"columnas ausentes: {faltan}"


def test_pairs_annotated_una_fila_por_par():
    df = _annotated()
    assert len(df) == 6345
    assert df.pair_id.nunique() == 6345


def test_pairs_annotated_null_cause_valida():
    df = _annotated()
    causas = set(df.null_cause.dropna().unique())
    assert causas <= NULL_CAUSES, f"causas desconocidas: {sorted(causas - NULL_CAUSES)}"
    # los pares efectivos no llevan causa
    assert df.loc[df.lever_effective.astype(bool), "null_cause"].isna().all()


def test_effective_pair_ids_coincide_con_la_columna():
    df = _annotated()
    ids = cp.effective_pair_ids(df)
    assert isinstance(ids, frozenset)
    esperado = frozenset(int(x) for x in df.loc[df.lever_effective.astype(bool), "pair_id"])
    assert ids == esperado
    assert len(ids) > 0
    assert len(ids) < len(df)


def test_pair_ids_efectivos_incluidos_en_pairs_parquet():
    import pyarrow.parquet as pq
    df = _annotated()
    pairs = pq.read_table(cp.PAIRS_PARQUET).to_pandas()
    assert set(df.pair_id) == set(pairs.pair_id)
