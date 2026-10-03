"""PASO 3c — criterios y medidas de la anotación corregida de pares CF.

Congelan:

* la **convención 0.1** de tiempos (celda ``(g-5, g]``, ``t1 = t0 + 60``) y la
  frontera ``t_boundary`` con ``pre_action <=> t1 < t_boundary``,
  ``post_action <=> t1 >= t_boundary``;
* ``t_effective`` (``t_action`` para acciones puntuales, ``t_divergence_raw``
  para consignas persistentes efectivas, NaN para los nulos);
* las medidas de las fases B1, B2, C (adelanto y C5), D y E1 sobre los 6345
  pares, leídas del manifiesto v2 (no se recalculan: son literales medidos);
* y que el pipeline de entrenamiento sigue anclado a la anotación de 3b porque
  B1 FALLA (fase F no adoptada).

Ningún caso vuelve a medir el corpus: todos leen artefactos ya escritos (el
manifiesto y el parquet de la anotación v2) o comprueban funciones puras.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tokens import cf_pairs as cp

ROOT = Path(__file__).resolve().parents[1]
V2_PARQUET = ROOT / "data" / "tokens_v2" / "pairs_annotated_v2.parquet"
V2_JSON = ROOT / "manifests" / "tokens_v2_cf_pairs_annotation_v2.json"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def v2() -> pd.DataFrame:
    return pd.read_parquet(V2_PARQUET)


@pytest.fixture(scope="module")
def man() -> dict:
    return json.loads(V2_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fase 0 — convención de tiempos y frontera
# ---------------------------------------------------------------------------

def test_convencion_prefijo_estricto_y_post_contiene_la_celda_frontera():
    """La celda (g-5, g] cierra en g y g le pertenece.

    ``t1 == t_boundary`` es la ventana que CONTIENE la celda de la frontera, y
    por tanto es post-acción; el prefijo exige cierre estrictamente anterior.
    """
    tb = 5220.0
    assert cp.post_action_bounded(5220.0, tb) is True    # contiene la celda de tb
    assert cp.post_action_bounded(5215.0, tb) is False   # cierra antes
    assert cp.pre_action_bounded(5215.0, tb) is True
    assert cp.pre_action_bounded(5220.0, tb) is False    # ya ve la intervención
    assert cp.pre_action_bounded(5100.0, tb) is True


def test_prefijo_y_post_cubren_todas_las_ventanas_sin_solape():
    tb = 8004.0
    t1s = [t0 + 60 for t0 in range(7800, 8200, 60)]
    pre = [t for t in t1s if cp.pre_action_bounded(t, tb)]
    post = [t for t in t1s if cp.post_action_bounded(t, tb)]
    assert len(pre) + len(post) == len(t1s)
    assert set(pre).isdisjoint(post)
    assert max(pre) == 7980.0
    assert min(post) == 8040.0


def test_frontera_no_usa_margen_de_celdas():
    """El margen de 2 celdas de 3b está retirado de la definición corregida."""
    assert cp.pre_action_bounded(5215.0, 5220.0) is True   # 1 celda antes: prefijo
    assert cp.pre_action(5215.0, 5220.0) is False          # 3b: margen de 2 celdas
    assert cp.PREFIX_MARGIN_CELLS == 2                     # 3b intacto (no adoptado)


def test_t_boundary_es_min_de_la_divergencia_de_rejilla_y_la_rejilla_de_t_effective():
    assert cp.t_boundary_of(5220.0, 5220.2, 5.0) == 5220.0
    assert cp.t_boundary_of(3120.0, 3125.2, 5.0) == 3120.0
    assert cp.t_boundary_of(7000.0, 6996.8, 5.0) == 6995.0
    assert cp.t_boundary_of(None, 5220.2, 5.0) == 5220.0
    assert np.isnan(cp.t_boundary_of(3120.0, float("nan"), 5.0))
    assert np.isnan(cp.t_boundary_of(None, float("nan"), 5.0))


# ---------------------------------------------------------------------------
# Fase A — t_effective
# ---------------------------------------------------------------------------

def test_t_effective_es_t_action_en_las_acciones_puntuales(v2):
    one = v2[v2.a2_class == "one_shot"]
    assert len(one) == 4380
    assert np.allclose(one.t_effective.to_numpy(np.float64),
                       one.t_action.to_numpy(np.float64))


def test_t_effective_es_la_divergencia_cruda_en_las_consignas_persistentes(v2):
    per = v2[v2.a2_class.isin(["inmediato", "retrasado"])]
    assert set(per.a2_class) == {"inmediato", "retrasado"}
    assert np.allclose(per.t_effective.to_numpy(np.float64),
                       per.t_divergence_raw.to_numpy(np.float64))
    # las persistentes son exactamente las 7 palancas de override de ventilación
    assert set(per.lever) == {"peep_up", "peep_down", "fio2_down", "set_fio2",
                              "set_rr", "set_tv", "set_peep"}


def test_los_pares_nulos_no_tienen_t_effective(v2):
    nul = v2[v2.a2_class == "nulo"]
    assert len(nul) == 168
    assert nul.t_effective.isna().all()
    assert not nul.lever_effective.any()


def test_reparto_de_clases_y_retraso_de_las_consignas(v2, man):
    assert int((v2.a2_class == "one_shot").sum()) == 4380
    assert int((v2.a2_class == "inmediato").sum()) == 1719
    assert int((v2.a2_class == "retrasado").sum()) == 78
    assert man["A"]["retraso_mediano_s"] == pytest.approx(484.245)
    assert man["A"]["retraso_max_s"] == pytest.approx(12531.81)
    # las retrasadas son todas de ventilación y el recorte se documenta con evidencia
    r = man["A"]["retrasados"]
    assert len(r) == 78
    assert all(x["lever"].startswith(("set_", "peep_", "fio2_")) for x in r)
    assert all("base_setpoint_at_action" in x for x in r)


# ---------------------------------------------------------------------------
# Fase B — prefijo fisiológico (B1, bloqueante) y B2 (informativo)
# ---------------------------------------------------------------------------

def test_b1_falla_y_el_fallo_queda_congelado(man):
    b1 = man["B1"]
    assert b1["ok"] is False
    assert b1["n_failures"] == 877
    assert b1["n_efectivos"] == 6177
    assert b1["pct_fallo"] == pytest.approx(14.198, abs=0.01)


def test_b1_por_palanca_y_por_clase(man):
    b1 = man["B1"]
    assert b1["por_palanca"] == {"set_fio2": 247, "set_rr": 145, "set_tv": 131,
                                 "set_peep": 126, "fio2_down": 96, "peep_up": 88,
                                 "peep_down": 35, "sevo_mac": 5, "eph_bolus": 3,
                                 "phen_bolus": 1}
    assert b1["clases"] == {"boundary_t_igual_t_eff": 329, "gt_0_le_5": 337,
                            "gt_5_le_10": 138, "gt_10": 73}
    assert sum(b1["clases"].values()) == b1["n_failures"]


def test_b1_la_imagen_nunca_se_adelanta_al_acto_del_generador(man):
    """El fallo no es una fuga hacia el futuro: es t_effective demasiado tardío.

    En los 877 pares la primera celda divergente de la imagen cierra en o
    después de ``t_action`` (el acto que ejecuta el generador).
    """
    b1 = man["B1"]
    assert b1["n_imagen_antes_de_t_action"] == 0
    assert b1["min_imagen_menos_t_action_s"] == pytest.approx(0.0)
    assert all(x["imagen_menos_t_action"] >= 0 for x in b1["failures"])


def test_b2_es_informativo_y_tambien_falla(man):
    assert man["B2"]["n_failures"] == 1773
    assert man["B2"]["ok_raw_prefix"] == 4572


# ---------------------------------------------------------------------------
# Fase C — adelanto de tokens, frontera y prefijo de tokens (C5)
# ---------------------------------------------------------------------------

def test_adelanto_maximo_de_tokens_es_una_celda(man):
    c = man["C"]
    assert c["lead_max_s"] == pytest.approx(5.2, abs=0.001)
    assert c["lead_p95_s"] == pytest.approx(4.02, abs=0.01)
    assert c["clases"] == {"lt_5": 6100, "llamado_5_10": 77, "gt_10": 0}
    assert c["lead_mayor_10"] == []


def test_adelanto_solo_en_las_palancas_de_bolo_estampado(man):
    """Sólo los bolos vasoactivos/pk superan 5 s: es el estampado a 0.5 s."""
    por = man["C"]["por_palanca"]
    altos = {k for k, v in por.items() if v["lead_max_s"] > 5.0}
    assert altos == {"eph_bolus", "ephedrine", "phen_bolus", "phen_rate"}
    assert por["ppf_bolus"]["lead_max_s"] < 1.0
    assert por["set_rr"]["lead_max_s"] == pytest.approx(0.0)


def test_c5_el_prefijo_de_tokens_es_identico_en_todos_los_pares(man):
    assert man["C"]["c5_ok"] is True
    assert man["C"]["c5_failures"] == []


def test_la_frontera_congelada_es_min_de_rejilla(v2):
    eff = v2[v2.lever_effective.fillna(False)]
    con = eff[eff.t_div_grid.notna() & eff.t_effective.notna()]
    tb = np.minimum(con.t_div_grid.to_numpy(np.float64),
                    np.floor(con.t_effective.to_numpy(np.float64) / 5.0) * 5.0)
    assert np.allclose(con.t_boundary.to_numpy(np.float64), tb)
    assert len(con) >= 6100
    # sin divergencia de rejilla la frontera es la rejilla de t_effective
    sin = eff[eff.t_div_grid.isna() & eff.t_effective.notna()]
    assert np.allclose(sin.t_boundary.to_numpy(np.float64),
                       np.floor(sin.t_effective.to_numpy(np.float64) / 5.0) * 5.0)
    # pares sin t_effective (nulos) no tienen frontera
    assert eff[eff.t_effective.isna()].t_boundary.isna().all()


# ---------------------------------------------------------------------------
# Fase D — el efecto en la ventana de la frontera o la siguiente
# ---------------------------------------------------------------------------

def test_lag_cero_o_uno_al_menos_del_99_por_ciento_en_todo_menos_ventilacion(man):
    pct = man["D"]["cumple_lag_0_1_pct_por_palanca"]
    fallan = set(man["D"]["palancas_que_fallan_99_pct"])
    assert fallan == {"set_rr", "set_tv", "set_peep"}
    assert all(pct[k] >= 99.0 for k in pct if k not in fallan)
    assert pct["sevo_mac"] == pytest.approx(99.2, abs=0.01)
    assert pct["set_rr"] == pytest.approx(93.94, abs=0.01)
    assert pct["set_tv"] == pytest.approx(97.87, abs=0.01)
    assert pct["set_peep"] == pytest.approx(94.5, abs=0.01)
    assert man["D"]["lag_ge_2"] == 49


def test_lag_mayor_que_1_solo_con_consignas_persistentes_y_mac(man):
    pares = man["D"]["lag_ge_2_pares"]
    assert {p["lever"] for p in pares} <= {"set_rr", "set_tv", "set_peep", "sevo_mac"}
    # la frontera la fija la consigna cruda (sparse) que se retiene en la rejilla
    assert {p["t_div_grid_col"] for p in pares} <= {
        "Primus/SET_RR_IPPV", "Primus/SET_TV_L", "Primus/SET_INTER_PEEP",
        "ce_sevoflurano"}


# ---------------------------------------------------------------------------
# Fase E — reclasificación, unidades y manifiesto v2
# ---------------------------------------------------------------------------

def test_e1_los_trece_pares_bajo_resolucion_se_reclasifican(man):
    e1 = man["E1"]
    assert e1["n"] == 13
    assert {(p["lever"], p["pair_id"]) for p in e1["pares"]} == {
        ("remi_up", 171181), ("remi_up", 171217), ("remi_up", 171473),
        ("ppf20_rate", 190207), ("ppf20_rate", 190739),
        ("rftn20_rate", 190811), ("rftn20_rate", 190827),
        ("rftn20_rate", 190913), ("rftn20_rate", 190975),
        ("rftn20_rate", 191359), ("rftn20_rate", 191381),
        ("rftn20_rate", 191423), ("rftn20_rate", 191477)}
    for p in e1["pares"]:
        # el cambio pedido llega a los 7.1 s del acto y el plan base lo repone
        assert 0.0 <= p["delta_s"] <= 7.1
        if p["series_identicas_desde_t_action"]:
            assert p["mantenimiento_s"] is None      # la serie nunca difiere
        else:
            assert 0.0 < p["mantenimiento_s"] <= 4.0  # pulso de 2-4 s (< 1 celda)


def test_e2_unidades_y_los_dos_parcs_no_change_requested(v2):
    """Orchestra/*_RATE va en mL/h; la petición va en mg/min y PPF20 = 20 mg/mL.

    Por eso 7.43 mg/min equivalen a 22.29 mL/h frente a los 22.181519 mL/h
    observados (0.49 %) y 4.31 mg/min a 12.93 mL/h frente a 13.190063 (1.97 %):
    ninguna de las dos peticiones cambia la consigna de forma apreciable.
    """
    for pid, req_mg_min, base_ml_h in ((190231, 7.43, 22.181519),
                                       (190567, 4.31, 13.190063)):
        req_ml_h = req_mg_min * 3.0            # mg/min -> mL/h con 20 mg/mL
        assert abs(req_ml_h - base_ml_h) / base_ml_h <= 0.05
        row = v2[v2.pair_id == pid].iloc[0]
        assert bool(row.lever_effective) is False


def test_el_manifiesto_v2_registra_lo_decidido_post_hoc(man):
    d = man["decided_post_hoc"]
    assert set(d) == {"post_3b", "post_3c_A", "post_3c_C", "post_3c_alcance"}
    assert "split_t" in d["post_3b"]["motivo"]
    assert "t_boundary" in d["post_3c_C"]["decision"]
    assert "3b restringió" in d["post_3c_alcance"]["decision"]


def test_los_artefactos_de_3b_no_se_han_tocado(man):
    """Ni ``pairs.parquet`` ni ``pairs_annotated.parquet`` cambian (R2).

    ``data/`` está en ``.gitignore``, así que el ancla independiente son los
    sha256 que dejó escritos el informe de 3b (fichero versionado).
    """
    txt = (ROOT / "reports" / "REPORT_paso3b_pares_cf.txt").read_text(encoding="utf-8")
    shas = re.findall(r"sha256 = ([0-9a-f]{64})", txt)
    assert len(shas) >= 2
    assert _sha(cp.PAIRS_PARQUET) == shas[0] == man["pairs_parquet_sha256"]
    assert _sha(cp.PAIRS_ANNOTATED_PARQUET) == shas[1] == man["pairs_annotated_3b_sha256"]
    assert _sha(V2_PARQUET) == man["pairs_annotated_v2_sha256"]
    assert V2_PARQUET.name != cp.PAIRS_ANNOTATED_PARQUET.name
    assert V2_PARQUET.exists()


# ---------------------------------------------------------------------------
# Fase F — la anotación v2 NO se adopta
# ---------------------------------------------------------------------------

def test_el_pipeline_sigue_apuntando_a_la_anotacion_de_3b():
    assert cp.PAIRS_ANNOTATED_PARQUET.name == "pairs_annotated.parquet"
    assert cp.PAIRS_ANNOTATED_V2_PARQUET.name == "pairs_annotated_v2.parquet"
    assert cp.PAIRS_ANNOTATED_PARQUET != cp.PAIRS_ANNOTATED_V2_PARQUET


def test_el_conjunto_efectivo_sigue_siendo_el_de_3b():
    ann = cp.load_pairs_annotated()
    assert len(ann) == 6345
    assert cp.effective_pair_ids_count(ann) == 6177
    assert "t_effective" not in ann.columns      # la anotación 3b no la tenía
    v2 = pd.read_parquet(V2_PARQUET)
    assert "t_effective" in v2.columns
