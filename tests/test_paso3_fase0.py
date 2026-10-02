"""Tests de la FASE 0 del PASO 3 (tokens_v2).

- R1: los manifiestos de las versiones v1 (pk_v1/context_v1/tokens_v1) no
  cambian de sha (nada de data/pk_v1/, data/context_v1/, data/tokens_v1/ se
  toca en el paso 3).
- 0.1d: integridad de las copias versionadas de manifests/ frente a sus
  originales en data/ (byte a byte). Con skip explícito si data/ no está.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import paths

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = paths.MANIFESTS_DIR
DATA_ROOT = paths.DATA_ROOT


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# R1 — manifiestos v1 congelados (no se tocan)
# ---------------------------------------------------------------------------

V1_MANIFEST_SHAS = {
    "data/pk_v1/manifest_pk.json":
        "1db873b51f8537c008e7f40fdae1475392f6513dce280e73f84bb6d2d7b6a8b2",
    "data/context_v1/vocab.json":
        "0ddcff917aa3cc2aeef148e7eeed7efdb661fe2b875fe30b82da2b0cd58cf57d",
    "data/tokens_v1/manifest_tokens.json":
        "58d2923181bbdc53f74c9260861901bb245129caf76415621c8bd30f5b88942f",
}


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="data/ no presente (clon sin datos)")
def test_v1_manifests_unchanged():
    for rel, expected in V1_MANIFEST_SHAS.items():
        p = ROOT / rel
        assert p.exists(), f"falta {rel}"
        assert _sha256(p) == expected, f"sha de {rel} ha cambiado"


# ---------------------------------------------------------------------------
# 0.1d — integridad de las copias versionadas de manifests/
# ---------------------------------------------------------------------------

# Copias que deben coincidir BYTE A BYTE con su original en data/.
COPY_ORIGINALS = {
    "windows_v4_manifest.json": "data/windows_v4/manifest.json",
    "windows_v4_split.parquet": "data/windows_v4/split.parquet",
    "pk_v1_manifest.json": "data/pk_v1/manifest_pk.json",
    "pk_v2_manifest.json": "data/pk_v2/manifest_pk.json",
    "context_v2_vocab.json": "data/context_v2/vocab.json",
    "tokens_v1_manifest.json": "data/tokens_v1/manifest_tokens.json",
    "tokens_v2_manifest.json": "data/tokens_v2/manifest_tokens.json",
    "ae_v2_manifest.json": "data/ae_v2/manifest_ae.json",
    "ae_v2_norm_stats.json": "data/ae_v2/norm_stats.json",
    "ae_bal_manifest.json": "data/ae_v1/ae_bal/manifest_ae.json",
    "ae_bal_norm_stats.json": "data/ae_v1/ae_bal/norm_stats.json",
    "ae_real_manifest.json": "data/ae_v1/ae_real/manifest_ae.json",
    "ae_v1_snapshot.json": "data/ae_v1/v1_snapshot.json",
    "ae_v2_snapshot.json": "data/ae_v1/v2_snapshot.json",
    "cf_v7_counterfactual_manifest.json": "data/cf_v7/metadata/counterfactual_manifest.json",
    "paso1_decoder_v7.json": "data/diagnostics/paso1_decoder_v7/paso1_decoder_v7.json",
    "v10_gate_recheck.json": "data/diagnostics/v10_gate_recheck.json",
    "v10_marginales_val.json": "data/diagnostics/v10_marginales_val.json",
    "v10_real_null_placeholder_audit.json": "data/diagnostics/v10_real_null_placeholder_audit.json",
    "v6_validate_results.json": "data/diagnostics/v6_validate_results.json",
    "v7_transfer_probe_results.json": "data/diagnostics/v7_transfer_probe_results.json",
    "v7_validate_results.json": "data/diagnostics/v7_validate_results.json",
    "v9_manifest_global.json": "data/diagnostics/v9_manifest_global.json",
    "v9_real_equivalence.json": "data/diagnostics/v9_real_equivalence.json",
    "v9_reconstruction_check.json": "data/diagnostics/v9_reconstruction_check.json",
}

# Copias cuyo original en data/ se regenera (o se perdió): se conserva la
# copia y se documenta en provenance_gaps.json. Aquí se fija el sha de la
# COPIA para detectar que la copia misma no cambie.
MUTABLE_SNAPSHOT_SHAS = {
    "cohort_gap_results.json":
        "ee61c3fd43a6536e3b406627d3f97f2249c80a67648cde78bd10aa635a03d284",
    "gap_addendum_results.json":
        "766f7be70a6ea298d1a459f0dfd096ca9ecb51d162a72fe702bd6f7fdaac5276",
    "v7_attribution_results.json":
        "604c58d6452f81f24ee8e6d356281d7eefb41ee379f79e623c1099df2a87a324",
    "cohort_gap_v7_results.json":
        "a63fb7a38d2d2a17f4adc368eea1f36700f5692f5a08861aa5a18e321855e3af",
}


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="data/ no presente (clon sin datos)")
def test_manifest_copies_match_originals():
    missing = []
    for cop, orig in COPY_ORIGINALS.items():
        c = MANIFESTS / cop
        o = ROOT / orig
        assert c.exists(), f"copia ausente: {cop}"
        if not o.exists():
            missing.append(orig)
            continue
        assert c.read_bytes() == o.read_bytes(), f"{cop} difiere de {orig}"
    assert not missing, f"originales ausentes en data/: {missing}"


def test_mutable_snapshot_copies_frozen():
    """Las copias de resultados regenerables no cambian (su original sí)."""
    for cop, expected in MUTABLE_SNAPSHOT_SHAS.items():
        c = MANIFESTS / cop
        assert c.exists(), f"copia ausente: {cop}"
        assert _sha256(c) == expected, f"sha de la copia {cop} ha cambiado"
