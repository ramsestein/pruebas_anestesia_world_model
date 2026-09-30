"""Tests de src/tokens/context_vocab.py (módulo de contexto del contrato de tokens v1).

Metodología: tests primero. Este fichero se escribe ANTES que context_vocab.py; en
la primera ejecución todo debe estar en ROJO (ImportError/AttributeError).

Cubre los tests mínimos a)..o) de la tarea. Los unitarios (a..i) usan datos
construidos en el test; los de integración (j..o) usan el dataset en disco.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from tokens import context_vocab as cv

ROOT = Path(__file__).resolve().parents[1]

INTEGRATION = pytest.mark.integration


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _df(caseids, **overrides):
    """DataFrame clínico mínimo con todas las columnas de contexto en NaN."""
    data: dict = {"caseid": list(caseids)}
    for c in cv.CONTEXT_COLUMNS:
        data[c] = [np.nan] * len(list(caseids))
    for k, vals in overrides.items():
        data[k] = list(vals)
    return pd.DataFrame(data)


def _vocab_lookup(vocab):
    return {it["item_id"]: it for it in vocab}


# --------------------------------------------------------------------------
# a) Deduplicación INSPIRE
# --------------------------------------------------------------------------

def test_a_dedup_inspire_most_nonnull_then_first():
    df = pd.DataFrame({
        "caseid": [10, 10, 20, 20, 30],
        "a": [1.0, np.nan, 1.0, np.nan, np.nan],
        "b": [2.0, np.nan, 2.0, 3.0, np.nan],
    })
    out = cv.dedupe_inspire(df)
    assert sorted(out.caseid.tolist()) == [10, 20, 30]
    # 10: fila 0 (2 no-null) gana a fila 1 (0 no-null)
    r10 = out[out.caseid == 10].iloc[0]
    assert r10["a"] == 1.0 and r10["b"] == 2.0
    # 20: empate a 2 no-null -> primera (fila 2)
    r20 = out[out.caseid == 20].iloc[0]
    assert r20["b"] == 2.0
    # 30: una sola fila (0 no-null) se conserva
    assert len(out[out.caseid == 30]) == 1


# --------------------------------------------------------------------------
# b) Columnas excluidas: no se leen como feature ni aparecen en el vocabulario
# --------------------------------------------------------------------------

@pytest.mark.parametrize("col", cv.EXCLUDED_COLUMNS)
def test_b_excluded_column_never_read_nor_emitted(col):
    # 1) solicitar la columna como feature debe lanzar error
    with pytest.raises(ValueError):
        cv.assert_feature_columns([col])
    # 2) no es una columna de contexto y no puede aparecer como item_id
    assert col not in cv.CONTEXT_COLUMNS
    # 3) ningún item_id estático (continuos + binarios) usa la columna excluida;
    #    los categóricos derivan solo de CATEGORICAL_COLUMNS ⊂ CONTEXT_COLUMNS
    for item_id in cv.STATIC_ITEM_IDS:
        assert not item_id.startswith(f"{col}:"), f"{col} aparece en {item_id}"
        assert item_id != col, f"{col} aparece como item_id"
    assert not any(c.startswith(f"{col}:") for c in cv.CATEGORICAL_COLUMNS)


# --------------------------------------------------------------------------
# c) Normalización solo sobre train
# --------------------------------------------------------------------------

def test_c_stats_computed_on_train_only():
    df = _df(
        [1, 2, 3, 4],
        split=["train", "train", "train", "val"],
        age=[10.0, 12.0, 14.0, 1000.0],
    )
    stats = cv.compute_train_stats(df, cv.CONTINUOUS_COLUMNS)
    mean, std = stats["age"]
    assert mean == pytest.approx(12.0, rel=1e-9)
    # desviación poblacional (ddof=0) sobre {10,12,14}
    assert std == pytest.approx(np.sqrt(8.0 / 3.0), rel=1e-6)


# --------------------------------------------------------------------------
# d) Agrupación en "otros"
# --------------------------------------------------------------------------

def test_d_otros_grouping_threshold_50_train_only():
    train = pd.Series(["common"] * 50 + ["rare"] * 49)
    mapping, has_otros = cv.build_category_mapping(train, threshold=50)
    assert mapping["common"] == "common"
    assert mapping["rare"] == "otros"
    assert has_otros is True
    # sin val: aunque val tuviera 1000 casos de "rare", la decisión es de train
    train2 = pd.Series(["common"] * 50 + ["rare"] * 50)
    mapping2, has_otros2 = cv.build_category_mapping(train2, threshold=50)
    assert mapping2["rare"] == "rare"
    assert has_otros2 is False


# --------------------------------------------------------------------------
# e) sex -> sex:F solo cuando femenino
# --------------------------------------------------------------------------

def test_e_sex_female_only():
    df = _df([1, 2, 3, 4], sex=["F", "M", None, "F"])
    vocab = cv.build_vocab(df)
    toks = cv.emit_tokens(df, vocab)
    sex_ids = set(toks[toks.item_id == "sex:F"].caseid.tolist())
    assert sex_ids == {1, 4}
    # ningún token sex:M existe
    assert not (toks.item_id.str.startswith("sex:") & (toks.item_id != "sex:F")).any()


# --------------------------------------------------------------------------
# f) preop_ecg / preop_pft anormal según regla; desconocido -> no emitido
# --------------------------------------------------------------------------

def test_f_ecg_pft_abnormal_rule_and_unknown():
    df = _df(
        [1, 2, 3, 4, 5],
        preop_ecg=["Normal Sinus Rhythm", "Atrial fibrillation", None, "", "Banana"],
        preop_pft=["Normal", "Mild obstructive", None, "", "Banana"],
    )
    vocab = cv.build_vocab(df)
    toks = cv.emit_tokens(df, vocab)
    assert set(toks[toks.item_id == "preop_ecg:anormal"].caseid.tolist()) == {2}
    assert set(toks[toks.item_id == "preop_pft:anormal"].caseid.tolist()) == {2}


# --------------------------------------------------------------------------
# g) Nulos en continuos -> sin fila
# --------------------------------------------------------------------------

def test_g_null_continuous_no_row():
    df = _df([1, 2, 3], age=[30.0, np.nan, 50.0])
    vocab = cv.build_vocab(df)
    toks = cv.emit_tokens(df, vocab)
    age_toks = toks[toks.item_id == "age"]
    assert set(age_toks.caseid.tolist()) == {1, 3}
    # el valor queda normalizado: stats train = {30,50} -> mean 40, std 10 (ddof=0)
    v1 = float(age_toks[age_toks.caseid == 1].value.iloc[0])
    assert v1 == pytest.approx((30.0 - 40.0) / 10.0, rel=1e-6)


# --------------------------------------------------------------------------
# h) Normalización de item_id
# --------------------------------------------------------------------------

def test_h_item_id_normalization():
    assert cv.normalize_item_value("Colorectal Surgery") == "colorectal_surgery"
    assert cv.normalize_item_value("Left lateral decubitus") == "left_lateral_decubitus"
    # floats integrales -> sin ".0"
    assert cv.normalize_item_value(2.0) == "2"
    assert cv.normalize_item_value(2) == "2"


# --------------------------------------------------------------------------
# i) Determinismo: dos ejecuciones -> bytes idénticos
# --------------------------------------------------------------------------

def test_i_determinism_identical_bytes(tmp_path):
    df = _df(
        list(range(1, 21)),
        age=np.linspace(20, 80, 20),
        sex=["F"] * 10 + ["M"] * 10,
        optype=["Colorectal Surgery"] * 10 + ["General"] * 10,
        preop_ecg=["Normal Sinus Rhythm"] * 20,
        preop_pft=["Normal"] * 20,
    )
    df["split"] = "train"
    df["source"] = "synthetic_v5"
    vocab = cv.build_vocab(df)
    toks1 = cv.emit_tokens(df, vocab)
    toks2 = cv.emit_tokens(df, vocab)
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    cv.write_tokens(toks1, p1)
    cv.write_tokens(toks2, p2)
    assert p1.read_bytes() == p2.read_bytes()


# --------------------------------------------------------------------------
# Integración (sobre disco)
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def built():
    """Ejecuta el build completo una vez y devuelve el resumen + datos cargados."""
    summary = cv.run()
    tokens = pq.read_table(cv.OUT_DIR / "tokens.parquet").to_pandas()
    coverage = pq.read_table(cv.OUT_DIR / "coverage.parquet").to_pandas()
    vocab = json.loads((cv.OUT_DIR / "vocab.json").read_text(encoding="utf-8"))
    cases = cv.load_cases()
    return dict(summary=summary, tokens=tokens, coverage=coverage, vocab=vocab, cases=cases)


@INTEGRATION
def test_j_gate4_columns_read_do_not_intersect_excluded(built):
    cols = set(built["summary"]["gate4"]["columns_requested"])
    excl = set(cv.EXCLUDED_COLUMNS) - {"caseid"}  # caseid es la clave, no feature
    assert cols & excl == set(), cols & excl
    # y las columnas leídas son exactamente las de contexto + caseid
    assert cols == set(cv.CONTEXT_COLUMNS) | {"caseid"}


@INTEGRATION
def test_k_gate8_no_category_below_50_in_train(built):
    toks = built["tokens"]
    cases = built["cases"]
    train_caseids = set(cases[cases.split == "train"].caseid.tolist())
    # recuenta en train por item_id
    train_counts = toks[toks.caseid.isin(train_caseids)].groupby("item_id").size()
    vocab = {it["item_id"]: it for it in built["vocab"]["items"]}
    for item_id, it in vocab.items():
        if it["tipo"] != "categorico" or it.get("scope") != "v1":
            continue
        if item_id.endswith(":otros"):
            continue
        n = int(train_counts.get(item_id, 0))
        assert n >= 50, f"categoría {item_id} con {n} casos en train (< 50)"


@INTEGRATION
def test_l_gate9_stats_match_train_recomputation(built):
    vocab = {it["item_id"]: it for it in built["vocab"]["items"]}
    for item_id, it in vocab.items():
        if it["tipo"] != "continuo":
            continue
        col = it["columna"]
        # recomputa independientemente sobre las filas train del dataset clínico
        mean, std = cv.recompute_continuo_stats(col)
        assert mean == pytest.approx(it["mean"], rel=1e-6, abs=1e-6)
        assert std == pytest.approx(it["std"], rel=1e-6, abs=1e-6)


@INTEGRATION
def test_m_gate6_source_probe(built):
    g6 = built["summary"]["gate6"]
    assert g6["accuracy"] <= 0.60, (
        f"gate 6 supera 0.60 (accuracy={g6['accuracy']:.4f}); "
        f"top-1={g6['top_coefs'][0]['item_id']}"
    )
    # el sondeo de solo-presencia, los coeficientes y la ablación top-1 se reportan
    assert "presence_accuracy" in g6
    assert len(g6["top_coefs"]) == g6["n_items"]
    assert "top1" in g6["ablation"]
    assert set(g6["pairwise"].keys()) == {
        "real_vs_synthetic_v5", "real_vs_cf_v5",
        "real_vs_vaso_reinf_v5", "synthetic_v5_vs_cf_v5",
    }


@INTEGRATION
def test_n_gate7_ks_tvd_table(built):
    g7 = built["summary"]["gate7"]
    assert len(g7) > 0
    # todas las filas llevan estadístico y n
    for row in g7:
        assert "statistic" in row and row["statistic"] is not None


@INTEGRATION
def test_o_coherence_table(built):
    coh = built["summary"]["coherence"]
    assert "bmi" in coh and "ranges" in coh
    assert len(coh["bmi"]) > 0


# --------------------------------------------------------------------------
# Iteración 2: scope v1/v2
# --------------------------------------------------------------------------

@INTEGRATION
def test_v2_a_no_v2_items_in_tokens(built):
    vocab = {it["item_id"]: it for it in built["vocab"]["items"]}
    v2_ids = {iid for iid, it in vocab.items() if it["scope"] == "v2"}
    assert len(v2_ids) > 0
    assert set(built["tokens"].item_id.unique()) & v2_ids == set()


@INTEGRATION
def test_v2_b_all_v1_items_present(built):
    vocab = {it["item_id"]: it for it in built["vocab"]["items"]}
    v1_ids = {iid for iid, it in vocab.items() if it["scope"] == "v1"}
    emitted = set(built["tokens"].item_id.unique())
    assert v1_ids == emitted, v1_ids ^ emitted
    # los 7 item_id base de v1 contribuyen cada uno con al menos un item
    for col in cv.V1_BASE_COLUMNS:
        assert any(iid == col or iid.startswith(f"{col}:") for iid in v1_ids), col


@INTEGRATION
def test_v2_c_vocab_has_86_with_scope(built):
    vocab = built["vocab"]
    assert vocab["n_items"] == 86
    v1 = [it for it in vocab["items"] if it["scope"] == "v1"]
    v2 = [it for it in vocab["items"] if it["scope"] == "v2"]
    assert len(v1) + len(v2) == 86
    assert len(v1) == 12 and len(v2) == 74
    # los estadísticos de v2 se calculan y guardan aunque no se emitan
    v2_cont = [it for it in v2 if it["tipo"] == "continuo"]
    assert len(v2_cont) == 19
    for it in v2_cont:
        assert it["mean"] is not None and it["std"] is not None
