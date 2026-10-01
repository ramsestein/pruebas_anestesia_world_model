"""Tests del PASO 2 — reentrenamiento del AE sobre real + v7 (ae_v2).

Cubren:
  - R1: data/ae_v1/ no se toca (sha256 de encoder y decoder de ae_bal).
  - El manifest de ae_v2 registra el sha de physio_ae.py (A2), la semilla, el
    motivo de parada y que no fue el tope de épocas (B1).
  - Reproducción de los gates C1-C4 y C6 contra manifests/ae_v2_manifest.json
    con |diff| < 1e-6.
  - Gates 4, 5, 7 y 8 (determinismo/congelación, decode_state, sin fuga
    temporal, estadísticos solo de train) ejecutados contra ae_v2.
  - Si se acepta: paths.AE_DIR apunta a ae_v2 y provenance_gaps.json refleja
    el cambio de estado.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

import paths

# IMPORTANTE (Windows): importar ae.physio_ae (pyarrow) ANTES de torch.
from ae import physio_ae as pa  # noqa: E402
import torch  # noqa: E402  (ya importado por pa en el orden correcto)

CLINICAL = ["Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS", "Primus/ETCO2"]
V7 = ["synthetic_v7", "vaso_reinf_v7", "cf_v7"]

AE_BAL_ENCODER_SHA = "8921d71986ffa0ac7b50a94b1b0bbaf4c6089c0824531ce975a1d91d15f7b1ef"
AE_BAL_DECODER_SHA = "c462fdbf0413672bb7e89a89242e226597844b9cd25c40b4ae1bdc6ac0331578"


def _ae_v2_dir() -> Path:
    return paths.DATA_ROOT / "ae_v2"


def _load_ae_v2() -> pa.PhysioAE:
    d = _ae_v2_dir()
    model = pa.make_model(pa.SEED)
    model.encoder.load_state_dict(torch.load(d / "encoder.pt", map_location="cpu"))
    model.decoder.load_state_dict(torch.load(d / "decoder.pt", map_location="cpu"))
    return model.eval()


def _norm_stats_v2() -> dict:
    return json.loads((_ae_v2_dir() / "norm_stats.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest():
    p = paths.MANIFESTS_DIR / "ae_v2_manifest.json"
    assert p.is_file(), "ejecuta scripts/paso2_train_ae_v2.py primero"
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# R1 — ae_v1 no se toca
# ---------------------------------------------------------------------------

def test_ae_v1_untouched_encoder_decoder_sha():
    """Los sha256 de encoder y decoder de ae_bal no han cambiado (R1)."""
    enc = pa.sha256(paths.AE_V1_DIR / "ae_bal" / "encoder.pt")
    dec = pa.sha256(paths.AE_V1_DIR / "ae_bal" / "decoder.pt")
    assert enc == AE_BAL_ENCODER_SHA, f"encoder ae_bal cambiado: {enc}"
    assert dec == AE_BAL_DECODER_SHA, f"decoder ae_bal cambiado: {dec}"


# ---------------------------------------------------------------------------
# B1 / A2 — campos de procedencia del manifest
# ---------------------------------------------------------------------------

def test_manifest_records_provenance_fields(manifest):
    """El manifest registra el sha de physio_ae.py (A2), la semilla, el motivo
    de parada y que no fue el tope de épocas (B1)."""
    sha_now = pa.sha256(Path(pa.__file__).resolve())
    assert manifest["sha256_physio_ae_py"] == sha_now, "sha physio_ae.py no coincide"
    assert len(manifest["sha256_train_script"]) == 64
    assert manifest["seed"] == 42
    tr = manifest["training"]
    assert tr["stop_reason"] in ("patience", "min_lr"), \
        f"paró por {tr['stop_reason']}"
    assert tr["stop_reason"] != "max_epochs"
    assert tr["epochs_run"] < manifest["hyperparameters"]["max_epochs"]


def test_manifest_has_paso2_gates(manifest):
    pg = manifest["paso2_gates"]
    for key in ("C1", "C2", "C3", "C4"):
        assert key in pg["blocking"], f"falta {key} en blocking"
    assert "k_def" in pg
    assert pg["k_def"] is None or 1 <= pg["k_def"] <= 32


# ---------------------------------------------------------------------------
# Gates 4, 5, 7 y 8 contra ae_v2 (C5)
# ---------------------------------------------------------------------------

def test_gate4_frozen_and_deterministic(manifest):
    """Gate 4: pesos congelados (sha = manifest) y decoder determinista."""
    enc = pa.sha256(_ae_v2_dir() / "encoder.pt")
    dec = pa.sha256(_ae_v2_dir() / "decoder.pt")
    assert enc == manifest["sha256_encoder"]
    assert dec == manifest["sha256_decoder"]

    model = _load_ae_v2()
    z = torch.randn(16, 32)
    with torch.no_grad():
        o1 = model.decode(z)
        o2 = model.decode(z)
    assert o1.numpy().tobytes() == o2.numpy().tobytes()


def test_gate5_decode_state_token1_equals_full(manifest):
    """Gate 5: reconstrucción desde el latente completo == token de estado 1."""
    model = _load_ae_v2()
    z = torch.randn(8, 32)
    full = model.decode(z)
    state = torch.cat([z, torch.zeros(8, 96)], dim=1)
    tok = model.decode_state(state)
    assert torch.allclose(full, tok, atol=1e-4)


@pytest.mark.integration
def test_gate7_read_columns_only(monkeypatch):
    """Gate 7: el módulo no lee columnas fuera de las 14 + máscaras + meta."""
    allowed = pa.ALLOWED_READ_COLS
    calls: list[set] = []
    real_rt = pq.read_table
    real_pf = pq.ParquetFile.read

    def spy_rt(source, columns=None, **kw):
        calls.append(None if columns is None else set(columns))
        return real_rt(source, columns=columns, **kw)

    def spy_pf(self, columns=None, **kw):
        calls.append(None if columns is None else set(columns))
        return real_pf(self, columns=columns, **kw)

    monkeypatch.setattr(pq, "read_table", spy_rt)
    monkeypatch.setattr(pq.ParquetFile, "read", spy_pf)

    parts = sorted((pa.WINDOWS_DIR / "source=real" / "split=train").glob("part-*.parquet"))
    pa.compute_norm_stats(parts[:1], pa.excluded_caseids())

    assert calls, "no se registró ninguna lectura"
    for cols in calls:
        assert cols is not None, "lectura sin columns="
        assert cols <= allowed, f"columnas fuera de lo permitido: {cols - allowed}"


@pytest.mark.slow
@pytest.mark.integration
def test_gate8_stats_only_train(manifest):
    """Gate 8: los estadísticos de ae_v2 coinciden con los recalculados sobre
    train (real + sintéticas v7), igual que los guarda el entrenamiento."""
    m_stats = manifest["norm_stats"]
    excluded = pa.excluded_caseids()
    parts = (pa.iter_partitions(["real"], "train")
             + pa.iter_partitions(pa.SYNTH_SOURCES, "train"))
    stats = pa.compute_norm_stats(parts, excluded)
    for t in pa.IMAGE_TRACKS:
        assert stats[t]["n"] == m_stats[t]["n"], f"n distinto en {t}"
        assert np.isclose(stats[t]["mean"], m_stats[t]["mean"], rtol=1e-6, atol=1e-9), t
        assert np.isclose(stats[t]["std"], m_stats[t]["std"], rtol=1e-6, atol=1e-9), t


# ---------------------------------------------------------------------------
# Reproducción de C1-C4 y C6 contra el manifest (|diff| < 1e-6)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
def test_gates_c1_c4_c6_reproduce(manifest):
    """C1-C4 y C6: una ejecución nueva reproduce el manifest ae_v2 con
    |diff| < 1e-6 (evaluación determinista sobre val)."""
    device = torch.device("cpu")
    model = _load_ae_v2().to(device)
    norm_stats = _norm_stats_v2()
    excluded = pa.excluded_caseids()

    val = pa.load_cells(paths.ALL_COHORTS, "val", excluded)
    gates = pa.evaluate(model, norm_stats, val, device)

    # C1: gate 2 real, variables clínicas.
    for t in CLINICAL:
        got = float(gates["gate2"]["real"][t])
        ref = float(manifest["gates"]["gate2"]["real"][t])
        assert abs(got - ref) < 1e-6, f"C1 {t}: {got} vs {ref}"

    # C2: gate 2 de cada cohorte v7, variables clínicas.
    for c in V7:
        for t in CLINICAL:
            got = float(gates["gate2"][c][t])
            ref = float(manifest["gates"]["gate2"][c][t])
            assert abs(got - ref) < 1e-6, f"C2 {c}|{t}: {got} vs {ref}"

    # C3: gate 1, perfil de orden (monotonía + fracción k=8), real y v7.
    for g in ("real", "synthetic"):
        p = gates["gate1"]["profile"][g]
        r = manifest["gates"]["gate1"]["profile"][g]
        assert p["monotone"] == r["monotone"], f"C3 {g} monotonía"
        assert abs(p["k8_fraction"] - r["k8_fraction"]) < 1e-6, f"C3 {g} k8"

    # C4: gate 3 (independencia de máscara ART).
    g3 = gates["gate3"]
    r3 = manifest["gates"]["gate3"]
    for k in ("err_hr_with_art", "err_hr_without_art", "hr_recon_diff"):
        assert abs(g3[k] - r3[k]) < 1e-6, f"C4 {k}: {g3[k]} vs {r3[k]}"
    assert g3["n_cases"] == r3["n_cases"]

    # C6: diag 5b por cohorte (real y cada v7).
    for cohort in ["real"] + V7:
        vc = pa.load_cells([cohort], "val", excluded)
        gc = pa.evaluate(model, norm_stats, vc, device)
        group = "real" if cohort == "real" else "synthetic"
        d5b = gc["diag5b"]["cohorts"][group]
        ref = manifest["diag5b_by_cohort"][cohort]
        assert d5b["k_clinico"] == ref["k_clinico"], f"C6 {cohort} k_clinico"
        for v in pa.DIAG5B_VARS:
            a = np.asarray(d5b["per_variable"][v], dtype=np.float64)
            b = np.asarray(ref["per_variable"][v], dtype=np.float64)
            assert np.allclose(a, b, atol=1e-6), f"C6 {cohort}|{v} no reproduce"


# ---------------------------------------------------------------------------
# Fase D — decisión aplicada
# ---------------------------------------------------------------------------

def test_decision_applied(manifest):
    """Fase D: el manifest declara la decisión y sus consecuencias son
    coherentes (paths.AE_DIR y provenance_gaps.json)."""
    decision = manifest.get("decision")
    assert decision in ("ACCEPTED", "REJECTED"), \
        "falta el campo 'decision' en ae_v2_manifest.json (fase D)"
    gaps = json.loads((paths.MANIFESTS_DIR / "provenance_gaps.json")
                      .read_text(encoding="utf-8"))
    entries = gaps["entries"]

    if decision == "ACCEPTED":
        assert paths.AE_DIR.name == "ae_v2", \
            "paths.AE_DIR debería apuntar a data/ae_v2/"
        by_artifact = {e["artifact"]: e for e in entries}
        e = by_artifact.get(
            "physio_ae.py en el sha 3f2817c9 (código con el que se ENTRENÓ ae_bal)")
        assert e is not None and e["status"] == "superado por ae_v2", \
            "provenance_gaps no refleja 'superado por ae_v2'"
        assert manifest["paso2_gates"]["k_def"] == manifest["k_def_final"]
    else:
        assert paths.AE_DIR.name == "ae_v1", \
            "paths.AE_DIR no debería cambiar si ae_v2 se rechaza"
        assert manifest["k_def_final"] == 13
