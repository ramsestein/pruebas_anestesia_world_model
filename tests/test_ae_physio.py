"""Tests del autoencoder de fisiología v1 (src/ae/physio_ae.py).

Unitarios (a-f) sin disco; integración (g-j) sobre data/windows_v2/ y
data/ae_v1/ (generado con `python -m ae.physio_ae run`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

import paths

from ae import physio_ae as pae

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_DIR = paths.WINDOWS_DIR / "windows"
AE_DIR = paths.AE_DIR


# --------------------------------------------------------------------------
# Unitarios (sin disco)
# --------------------------------------------------------------------------

def test_a_nested_dropout_zeroes_tail():
    """Con k fijo, las dimensiones k+1..32 del latente que llega al decoder son
    exactamente 0; con k=32 no se anula ninguna."""
    model = pae.make_model(seed=0)
    model.eval()
    x = torch.randn(8, 28)
    captured: list[torch.Tensor] = []
    orig_decode = model.decode

    def spy_decode(z):
        captured.append(z.detach().clone())
        return orig_decode(z)

    model.decode = spy_decode  # type: ignore[method-assign]
    k = torch.full((8,), 5, dtype=torch.long)
    model.forward(x, k=k)
    z = captured[0]
    assert z.shape[-1] == 32
    assert torch.all(z[:, 5:] == 0.0)
    assert torch.any(z[:, :5] != 0.0)  # el prefijo no está anulado

    # k=32: nada anulado, el latente llega íntegro al decoder
    captured.clear()
    model.forward(x, k=32)
    z32 = captured[0]
    assert torch.equal(z32, model.encode(x))


def test_b_loss_ignores_masked():
    """Cambiar el valor de una celda enmascarada no cambia la pérdida."""
    recon = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[2.0, 4.0, 6.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    l0 = pae.masked_variable_loss(recon, target, mask)

    target2 = target.clone()
    target2[0, 1] = 1000.0
    assert torch.equal(l0, pae.masked_variable_loss(recon, target2, mask))

    recon2 = recon.clone()
    recon2[0, 1] = -500.0
    assert torch.equal(l0, pae.masked_variable_loss(recon2, target, mask))


def test_c_per_variable_averaging():
    """Duplicar las celdas de una variable no cambia su peso relativo: la
    pérdida promedia primero por variable."""
    recon1 = torch.arange(1.0, 15.0).reshape(1, 14)
    tgt1 = torch.arange(2.0, 30.0, 2.0).reshape(1, 14)
    mask1 = torch.ones(1, 14)
    l1 = pae.masked_variable_loss(recon1, tgt1, mask1)

    # la variable 0 pasa a tener 2 celdas idénticas; el resto, 1 celda
    recon2 = torch.zeros(2, 14)
    recon2[0] = recon1[0]
    recon2[1, 0] = recon1[0, 0]
    tgt2 = torch.zeros(2, 14)
    tgt2[0] = tgt1[0]
    tgt2[1, 0] = tgt1[0, 0]
    mask2 = torch.zeros(2, 14)
    mask2[0] = 1.0
    mask2[1, 0] = 1.0
    l2 = pae.masked_variable_loss(recon2, tgt2, mask2)
    assert torch.allclose(l1, l2, atol=1e-6)


def test_d_normalize_roundtrip():
    """z-score con los estadísticos guardados y su inversa reconstruyen el
    valor físico (tolerancia 1e-6); std=0 -> 1.0."""
    tracks = pae.IMAGE_TRACKS
    stats = {t: {"mean": 10.0 + i, "std": 3.0 + 0.1 * i} for i, t in enumerate(tracks)}
    x = np.arange(1.0, 15.0, dtype=np.float64).reshape(1, 14) * 3.7
    z = pae.normalize(x, stats)
    y = pae.denormalize(z, stats)
    assert np.allclose(x, y, atol=1e-6)

    stats2 = {t: {"mean": 5.0, "std": 0.0} for t in tracks}
    z2 = pae.normalize(np.full((2, 14), 5.0, dtype=np.float64), stats2)
    assert np.all(z2 == 0.0)
    y2 = pae.denormalize(z2, stats2)
    assert np.allclose(y2, 5.0, atol=1e-6)


def test_e_determinism_50_steps():
    """Dos entrenamientos de 50 pasos con la misma semilla dan exactamente los
    mismos pesos."""
    s1 = pae.train_50_steps(seed=42, device="cpu")
    s2 = pae.train_50_steps(seed=42, device="cpu")
    assert set(s1) == set(s2)
    for k in s1:
        assert torch.equal(s1[k], s2[k]), f"peso distinto en {k}"


def test_f_decoder_14_outputs_no_masks():
    """El decoder produce 14 salidas y no intenta reconstruir máscaras."""
    model = pae.make_model(seed=0)
    model.eval()
    z = torch.randn(4, 32)
    assert model.decode(z).shape == (4, 14)
    x = torch.randn(4, 28)
    assert model.forward(x, k=32).shape == (4, 14)


# --------------------------------------------------------------------------
# Integración (sobre disco)
# --------------------------------------------------------------------------

@pytest.mark.integration
def test_g_gate7_read_columns(monkeypatch):
    """Gate 7: el módulo no lee ninguna columna fuera de las 14 + sus máscaras
    + caseid/t/source/split/phase (monkeypatch sobre pq.read_table)."""
    allowed = pae.ALLOWED_READ_COLS
    calls: list[tuple] = []
    real_rt = pq.read_table
    real_pf = pq.ParquetFile.read

    def spy_rt(source, columns=None, **kw):
        calls.append(("rt", None if columns is None else set(columns)))
        return real_rt(source, columns=columns, **kw)

    def spy_pf(self, columns=None, **kw):
        calls.append(("pf", None if columns is None else set(columns)))
        return real_pf(self, columns=columns, **kw)

    monkeypatch.setattr(pq, "read_table", spy_rt)
    monkeypatch.setattr(pq.ParquetFile, "read", spy_pf)

    parts = sorted((WINDOWS_DIR / "source=real" / "split=train").glob("part-*.parquet"))
    excluded = pae.excluded_caseids()
    pae.compute_norm_stats(parts[:1], excluded)

    assert calls, "no se registró ninguna lectura"
    for kind, cols in calls:
        assert cols is not None, f"lectura sin columns= ({kind})"
        assert cols <= allowed, f"columnas fuera de lo permitido: {cols - allowed}"


@pytest.mark.integration
def test_h_gate8_stats_match_manifest():
    """Gate 8: los estadísticos del manifest coinciden con los recalculados
    sobre train (real)."""
    manifest_path = AE_DIR / "ae_real" / "manifest_ae.json"
    if not manifest_path.exists():
        raise AssertionError("data/ae_v1/ae_real no existe: ejecuta 'python -m ae.physio_ae run'")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats_m = m["norm_stats"]
    excluded = pae.excluded_caseids()
    parts = sorted((WINDOWS_DIR / "source=real" / "split=train").glob("part-*.parquet"))
    stats = pae.compute_norm_stats(parts, excluded)
    for t in pae.IMAGE_TRACKS:
        assert stats[t]["n"] == stats_m[t]["n"], f"n distinto en {t}"
        assert np.isclose(stats[t]["mean"], stats_m[t]["mean"], rtol=1e-6, atol=1e-9), t
        assert np.isclose(stats[t]["std"], stats_m[t]["std"], rtol=1e-6, atol=1e-9), t


@pytest.mark.integration
def test_i_gate5_state_token1_equals_full():
    """Gate 5: reconstrucción desde el latente completo == reconstrucción
    tomando el latente como token de estado 1 (±1e-4)."""
    if not (AE_DIR / "ae_real" / "encoder.pt").exists():
        raise AssertionError("data/ae_v1/ae_real no existe: ejecuta 'python -m ae.physio_ae run'")
    model = pae.load_model("ae_real", device="cpu")
    model.eval()
    z = torch.randn(8, 32)
    full = model.decode(z)
    state = torch.cat([z, torch.zeros(8, 96)], dim=1)
    tok = model.decode_state(state)
    assert torch.allclose(full, tok, atol=1e-4)


@pytest.mark.integration
def test_j_gate4_decoder_bytes_identical():
    """Gate 4: dos pasadas del decoder sobre el mismo latente dan bytes
    idénticos."""
    if not (AE_DIR / "ae_real" / "decoder.pt").exists():
        raise AssertionError("data/ae_v1/ae_real no existe: ejecuta 'python -m ae.physio_ae run'")
    model = pae.load_model("ae_real", device="cpu")
    model.eval()
    z = torch.randn(16, 32)
    with torch.no_grad():
        o1 = model.decode(z)
        o2 = model.decode(z)
    assert o1.detach().numpy().tobytes() == o2.detach().numpy().tobytes()


# --------------------------------------------------------------------------
# Iteración 2: submuestra de val fija y gates corregidos
# --------------------------------------------------------------------------

def test_k_fixed_sample_indices_deterministic():
    """La elección de celdas de val para la parada temprana es determinista y
    usa la misma muestra para todas las épocas y las dos variantes."""
    a = pae._sample_fixed_indices(1_000_000, 3_000_000, 65_536, 65_536, seed=1234)
    b = pae._sample_fixed_indices(1_000_000, 3_000_000, 65_536, 65_536, seed=1234)
    ra, sa = a
    rb, sb = b
    assert np.array_equal(ra, rb)
    assert np.array_equal(sa, sb)
    assert len(ra) == 65_536 and len(sa) == 65_536
    assert ra.max() < 1_000_000 and sa.max() < 3_000_000


@pytest.mark.integration
def test_l_gate3_new_threshold():
    """Gate 3 (iter 2): reporta error de HR contra el HR verdadero con y sin
    ART, y la diferencia entre reconstrucciones; el veredicto se basa en
    err_hr_without_art < 2 lpm."""
    m = json.loads((AE_DIR / "ae_bal" / "manifest_ae.json").read_text(encoding="utf-8"))
    g3 = m["gates"]["gate3"]
    for k in ("err_hr_with_art", "err_hr_without_art", "hr_recon_diff"):
        assert k in g3, f"falta {k} en gate3"
        assert np.isfinite(g3[k]), f"{k} no finito"
    assert g3["ok"] == bool(g3["err_hr_without_art"] < 2.0)


@pytest.mark.integration
def test_m_gate6_per_cell():
    """Gate 6 (iter 2): sondeo por caso y por celda, ambos para 32 y 8 dims."""
    m = json.loads((AE_DIR / "ae_bal" / "manifest_ae.json").read_text(encoding="utf-8"))
    g6 = m["gates"]["gate6"]
    for scope in ("per_case", "per_cell"):
        assert scope in g6, f"falta {scope} en gate6"
        for dims in ("32", "8"):
            assert "auc_mean" in g6[scope][dims], f"falta auc_mean en {scope}/{dims}"
            assert 0.0 <= g6[scope][dims]["auc_mean"] <= 1.0
    assert g6["per_cell"]["n_cells"] == 200_000


@pytest.mark.integration
def test_n_diag5_present():
    """Diagnóstico 5: k mínimo tal que err <= 1.1*err(32), en val real y
    sintético (1..32)."""
    m = json.loads((AE_DIR / "ae_bal" / "manifest_ae.json").read_text(encoding="utf-8"))
    d5 = m["gates"]["diag5"]
    for g in ("real", "synthetic"):
        assert g in d5, f"falta {g} en diag5"
        assert isinstance(d5[g], int) and 1 <= d5[g] <= 32, f"diag5[{g}] inválido"


@pytest.mark.integration
def test_o_stopped_by_patience():
    """Corrección 1: el entrenamiento para por paciencia o por min_lr, no por
    el tope de épocas (ambas variantes)."""
    for variant in ("ae_real", "ae_bal"):
        m = json.loads((AE_DIR / variant / "manifest_ae.json").read_text(encoding="utf-8"))
        tr = m["training"]
        assert tr["stop_reason"] in ("patience", "min_lr"), \
            f"{variant}: paró por {tr['stop_reason']}"
        assert len(tr["val_losses"]) == tr["epochs_run"]


@pytest.mark.integration
def test_p_scheduler_fields():
    """Corrección 6: el manifest declara el planificador plateau y los campos
    de auditoría del recocido."""
    for variant in ("ae_real", "ae_bal"):
        m = json.loads((AE_DIR / variant / "manifest_ae.json").read_text(encoding="utf-8"))
        assert m["scheduler"] == "warmup_linear+reduce_on_plateau", variant
        assert isinstance(m["scheduler_reductions"], int) and m["scheduler_reductions"] >= 1
        assert m["final_lr"] > 0 and m["final_lr"] < pae.LR
        assert isinstance(m["best_epoch"], int) and m["best_epoch"] >= 1


@pytest.mark.integration
def test_q_best_epoch_1indexed():
    """Corrección 5: best_epoch es 1-indexado y coherente en todo el manifest."""
    for variant in ("ae_real", "ae_bal"):
        m = json.loads((AE_DIR / variant / "manifest_ae.json").read_text(encoding="utf-8"))
        tr = m["training"]
        assert tr["best_epoch"] == m["best_epoch"], variant
        assert 1 <= m["best_epoch"] <= tr["epochs_run"], variant


@pytest.mark.integration
def test_r_v3_not_worse_than_v1_real():
    """Requisito 3: la reconstrucción real de la v3 no debe ser peor que la de
    la v1 (HR y ART_MBP). La tolerancia se lee del campo 'acceptance' del
    manifest (no de una constante literal; iter 4, corrección D)."""
    snap = json.loads((AE_DIR / "v1_snapshot.json").read_text(encoding="utf-8"))
    for variant in ("ae_real", "ae_bal"):
        m = json.loads((AE_DIR / variant / "manifest_ae.json").read_text(encoding="utf-8"))
        acc = m["acceptance"]
        tol_lpm = acc["regression_tolerance_lpm"]
        tol_mmhg = acc["regression_tolerance_mmhg"]
        for label, track in (("HR", "Solar8000/HR"), ("ART_MBP", "Solar8000/ART_MBP")):
            v1 = snap[variant]["gates"]["gate2"]["real"][track]
            v3 = m["gates"]["gate2"]["real"][track]
            tol = tol_lpm if label == "HR" else tol_mmhg
            assert v3 <= v1 + tol, \
                f"{variant} {label} real: v3 {v3:.4f} > v1 {v1:.4f} + {tol}"


@pytest.mark.integration
def test_s_lr_history_present():
    """Requisito 2: historial de LR, train_loss y reducciones por época."""
    for variant in ("ae_real", "ae_bal"):
        m = json.loads((AE_DIR / variant / "manifest_ae.json").read_text(encoding="utf-8"))
        tr = m["training"]
        n = tr["epochs_run"]
        assert len(tr["lr_history"]) == n
        assert len(tr["train_losses"]) == n
        assert len(tr["reductions_history"]) == n
        assert tr["reductions_history"][-1] == m["scheduler_reductions"]


@pytest.mark.integration
def test_t_provenance_shas():
    """Corrección C (iter 4): REPORT_tokens_v4.txt es original y se queda en
    entradas ascendentes; REPORT_window_v2.txt es missing_original, va en
    provenance_gaps y NO aparece en entradas ascendentes."""
    m = json.loads((AE_DIR / "ae_bal" / "manifest_ae.json").read_text(encoding="utf-8"))
    # tokens_v4: entrada ascendente (original)
    assert len(m["sha256_report_tokens_v4"]) == 64
    # window_v2: AUSENTE de entradas ascendentes
    assert "sha256_report_window_v2" not in m, "window_v2 no debe ser entrada ascendente"
    # provenance_gaps presente y bien formada
    gaps = m["provenance_gaps"]
    assert isinstance(gaps, list) and len(gaps) == 1
    g = gaps[0]
    assert g["artifact"] == "REPORT_window_v2.txt"
    assert g["status"] == "missing_original"
    assert "note" in g and "no constituye procedencia" in g["note"]
    assert len(g["sha256_reconstruction"]) == 64
    # el fichero reconstruido existe y lleva la advertencia en mayúsculas
    p = ROOT / "reports" / "REPORT_window_v2_RECONSTRUIDO.txt"
    assert p.exists()
    first = p.read_text(encoding="utf-8").splitlines()[0]
    assert "RECONSTRUCCIÓN RETROSPECTIVA" in first.upper()
    # tokens_v4 original sigue existiendo
    assert (ROOT / "reports" / "REPORT_tokens_v4.txt").exists()


@pytest.mark.integration
def test_u_acceptance_field():
    """Corrección D (iter 4): el manifest declara el campo 'acceptance' con
    decided_post_hoc=true intacto y rationale."""
    for variant in ("ae_real", "ae_bal"):
        m = json.loads((AE_DIR / variant / "manifest_ae.json").read_text(encoding="utf-8"))
        acc = m["acceptance"]
        assert acc["regression_tolerance_lpm"] == 0.05
        assert acc["regression_tolerance_mmhg"] == 0.05
        assert acc["introduced_in"] == "iter_3"
        assert acc["decided_post_hoc"] is True
        assert "rationale" in acc and acc["rationale"]


@pytest.mark.integration
def test_v_diag5b_present():
    """Diagnóstico 5b (iter 4): tabla por k de 4 variables clínicas y k_clinico."""
    m = json.loads((AE_DIR / "ae_bal" / "manifest_ae.json").read_text(encoding="utf-8"))
    d5b = m["gates"]["diag5b"]
    assert set(d5b["variables"]) == {"Solar8000/HR", "Solar8000/ART_MBP",
                                     "BIS/BIS", "Primus/ETCO2"}
    for cohort in ("real", "synthetic"):
        c = d5b["cohorts"][cohort]
        assert c["k_clinico"] is None or 1 <= c["k_clinico"] <= 32
        for var in d5b["variables"]:
            assert len(c["per_variable"][var]) == 32


@pytest.mark.integration
def test_w_diag6_present():
    """Diagnóstico 6 (iter 4): A1-A4 presentes y con estructura esperada."""
    m = json.loads((AE_DIR / "ae_bal" / "manifest_ae.json").read_text(encoding="utf-8"))
    d6 = m["diagnostics"]["diag6"]
    a1 = d6["A1_support_overlap"]
    assert "real_cells_fraction_synthetic_neighbors" in a1
    assert "synthetic_cells_fraction_real_neighbors" in a1
    a2 = d6["A2_relative_separation"]
    assert a2["ratio_32"] > 0 and a2["ratio_8"] > 0
    a3 = d6["A3_separation_direction"]
    assert len(a3) == 32 and all("dim" in x and "weight" in x for x in a3)
    a4 = d6["A4_mechanism_control"]
    for key in ("masks_only", "values_only", "masks_and_values"):
        assert 0.0 <= a4[key]["auc_mean"] <= 1.0
