"""Tests de la FASE 0 del PASO 1 (cierres del paso 0)."""

from __future__ import annotations

import json

import paths


def test_gitattributes_present_crlf():
    p = paths.REPO_ROOT / ".gitattributes"
    assert p.is_file(), ".gitattributes no existe en la raíz"
    txt = p.read_text(encoding="utf-8")
    assert "*.py" in txt and "text eol=crlf" in txt
    assert "*.md" in txt and "*.txt" in txt and "*.json" in txt
    assert "*.parquet binary" in txt and "*.pt      binary" in txt


def test_provenance_gaps_json_valid():
    p = paths.MANIFESTS_DIR / "provenance_gaps.json"
    assert p.is_file(), "manifests/provenance_gaps.json no existe"
    d = json.loads(p.read_text(encoding="utf-8"))
    entries = d.get("entries")
    assert isinstance(entries, list) and len(entries) >= 8
    for e in entries:
        for k in ("artifact", "status", "lost", "substitute"):
            assert k in e and e[k], f"entrada sin campo {k}: {e}"
    artifacts = {e["artifact"] for e in entries}
    for a in ("windows_v2", "windows_v3",
              "cohortes v5 (synthetic_v5, vaso_reinf_v5, cf_v5)",
              "REPORT_window_v2.txt (original)", "v6_validate.py (original)",
              "physio_ae.py en el sha 3f2817c9 (código con el que se ENTRENÓ ae_bal)",
              "context_v1/vocab.json original (sha 945b7884…)"):
        assert a in artifacts, f"falta la entrada mínima {a}"
