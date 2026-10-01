"""PASO 2 — FASE D: decisión sobre ae_v2 y aplicación de consecuencias.

Lee manifests/ae_v2_manifest.json (ya con los veredictos C1-C4 y k_def),
recibe el resultado de C5 (gates 4/5/7/8, vía tests) por argumento y aplica
la regla FIJADA en el enunciado:

  - Si C1-C5 pasan: ae_v2 se ACEPTA. paths.AE_DIR -> data/ae_v2/, k_def
    DEFINITIVO = k_def medido sobre ae_v2, y en provenance_gaps.json las
    entradas de "physio_ae.py en 3f2817c9" y de "cohortes v5" pasan a estado
    "superado por ae_v2" con el sha de entrenamiento de A2.
  - Si falla cualquiera de C1-C5: ae_v2 se RECHAZA, ae_bal sigue vigente,
    k_def = 13 DEFINITIVO, y en provenance_gaps.json se registra en la entrada
    de ae_bal su irreproducibilidad como RIESGO ACEPTADO con el gate que
    falló. paths.AE_DIR no se toca.

Uso:
    python scripts/paso2_decide_ae_v2.py --c5 pass|fail

Salidas:
    Actualiza manifests/ae_v2_manifest.json y data/ae_v2/manifest_ae.json
    (campo 'decision' y 'k_def_final'), manifests/provenance_gaps.json y,
    si se acepta, src/paths.py (AE_DIR -> ae_v2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import paths  # noqa: E402

MANIFEST_COPY = paths.MANIFESTS_DIR / "ae_v2_manifest.json"
MANIFEST_DATA = paths.DATA_ROOT / "ae_v2" / "manifest_ae.json"
PROVENANCE = paths.MANIFESTS_DIR / "provenance_gaps.json"
PATHS_PY = paths.REPO_ROOT / "src" / "paths.py"


def _load_manifest() -> dict:
    return json.loads(MANIFEST_COPY.read_text(encoding="utf-8"))


def _save_manifest(m: dict) -> None:
    txt = json.dumps(m, indent=2, ensure_ascii=False)
    MANIFEST_COPY.write_text(txt, encoding="utf-8")
    if MANIFEST_DATA.exists():
        MANIFEST_DATA.write_text(txt, encoding="utf-8")


def _update_provenance_accept(sha_physio: str) -> None:
    gaps = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    for e in gaps["entries"]:
        if e["artifact"].startswith("physio_ae.py en el sha 3f2817c9"):
            e["status"] = "superado por ae_v2"
            e["superseded_by"] = "ae_v2 entrenado con physio_ae.py sha256 " + sha_physio
        elif e["artifact"].startswith("cohortes v5 (synthetic_v5"):
            e["status"] = "superado por ae_v2"
            e["superseded_by"] = ("ae_v2 entrenado sobre real + v7 "
                                  "(synthetic_v7, vaso_reinf_v7, cf_v7)")
    gaps["generated_at"] = "2026-10-01"
    PROVENANCE.write_text(json.dumps(gaps, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def _update_provenance_reject(failed_gate: str) -> None:
    gaps = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    for e in gaps["entries"]:
        if e["artifact"].startswith("physio_ae.py en el sha 3f2817c9"):
            e["risk_accepted"] = (
                "La irreproducibilidad del entrenamiento de ae_bal queda como "
                "RIESGO ACEPTADO: ae_v2 fue rechazado en el paso 2 (gate que "
                "falló: " + failed_gate + ").")
    PROVENANCE.write_text(json.dumps(gaps, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def _repoint_ae_dir() -> None:
    txt = PATHS_PY.read_text(encoding="utf-8")
    old = 'AE_DIR = DATA_ROOT / "ae_v1"'
    new = 'AE_DIR = DATA_ROOT / "ae_v2"'
    assert old in txt, "no se encontró la línea AE_DIR en paths.py"
    txt = txt.replace(old, new, 1)
    PATHS_PY.write_text(txt, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c5", choices=["pass", "fail"], required=True,
                    help="resultado de C5 (gates 4/5/7/8 vía tests)")
    args = ap.parse_args()

    m = _load_manifest()
    blocking = m["paso2_gates"]["blocking"]
    c1 = bool(blocking["C1"])
    c2 = bool(blocking["C2"])
    c3 = bool(blocking["C3"])
    c4 = bool(blocking["C4"])
    c5 = args.c5 == "pass"
    k_def = m["paso2_gates"]["k_def"]

    accepted = c1 and c2 and c3 and c4 and c5

    m["decision"] = "ACCEPTED" if accepted else "REJECTED"
    m["k_def_final"] = k_def if accepted else 13

    if accepted:
        _update_provenance_accept(m["sha256_physio_ae_py"])
        _repoint_ae_dir()
    else:
        failed = []
        if not c1:
            failed.append("C1")
        if not c2:
            failed.append("C2")
        if not c3:
            failed.append("C3")
        if not c4:
            failed.append("C4")
        if not c5:
            failed.append("C5")
        m["rejected_gates"] = failed
        _update_provenance_reject(", ".join(failed))

    _save_manifest(m)

    print("\n===== PASO 2 — FASE D =====", flush=True)
    print(f"  C1 {'PASA' if c1 else 'FALLA'}", flush=True)
    print(f"  C2 {'PASA' if c2 else 'FALLA'}", flush=True)
    print(f"  C3 {'PASA' if c3 else 'FALLA'}", flush=True)
    print(f"  C4 {'PASA' if c4 else 'FALLA'}", flush=True)
    print(f"  C5 {'PASA' if c5 else 'FALLA'}", flush=True)
    print(f"  ae_v2: {'ACEPTADO' if accepted else 'RECHAZADO'}", flush=True)
    print(f"  k_def: {m['k_def_final']} (DEFINITIVO)", flush=True)
    if accepted:
        print("  paths.AE_DIR -> data/ae_v2/ (commit propio)", flush=True)
    else:
        print("  paths.AE_DIR sin cambios; ae_bal sigue vigente", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
