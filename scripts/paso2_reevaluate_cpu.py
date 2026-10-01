"""PASO 2 — re-evaluación de ae_v2 en CPU (para reproducción bit a bit).

El entrenamiento evaluó los gates en GPU; la re-evaluación en CPU da valores
con ruido de coma flotante ~1e-6 (diferencia GPU vs CPU en los matmul). Para
que el test de reproducción (que corre en CPU) reproduzca el manifest con
|diff| < 1e-6, se RE-EVALUA en CPU (sin reentrenar: los pesos no cambian) y se
actualizan en el manifest los campos de gates/diagnósticos (manteniendo la
sección de entrenamiento, los sha y la procedencia intactos).

Uso:
    python scripts/paso2_reevaluate_cpu.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import paths  # noqa: E402

from ae import physio_ae as pa  # noqa: E402
import torch  # noqa: E402

MANIFEST_COPY = paths.MANIFESTS_DIR / "ae_v2_manifest.json"
MANIFEST_DATA = paths.DATA_ROOT / "ae_v2" / "manifest_ae.json"
AE_V2_DIR = paths.DATA_ROOT / "ae_v2"


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "paso2_train_ae_v2", SRC.parent / "scripts" / "paso2_train_ae_v2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    tm = _load_train_module()

    device = torch.device("cpu")
    model = pa.make_model(pa.SEED)
    model.encoder.load_state_dict(torch.load(AE_V2_DIR / "encoder.pt",
                                             map_location="cpu"))
    model.decoder.load_state_dict(torch.load(AE_V2_DIR / "decoder.pt",
                                             map_location="cpu"))
    model = model.to(device).eval()
    norm_stats = json.loads(
        (AE_V2_DIR / "norm_stats.json").read_text(encoding="utf-8"))
    excluded = pa.excluded_caseids()

    m = json.loads(MANIFEST_COPY.read_text(encoding="utf-8"))

    print("[reeval] cargando val completo (real + v7) ...", flush=True)
    val = pa.load_cells(paths.ALL_COHORTS, "val", excluded)
    print("[reeval] evaluando gates ...", flush=True)
    gates = pa.evaluate(model, norm_stats, val, device)
    print("[reeval] diagnóstico 6 ...", flush=True)
    diagnostics = pa.diagnose(model, norm_stats, val, device)
    print("[reeval] diag5b por cohorte ...", flush=True)
    d5b = tm.diag5b_by_cohort(model, norm_stats, excluded, device)
    print("[reeval] colas ...", flush=True)
    tails_ = tm.tails(model, norm_stats, val, device)
    print("[reeval] veredictos C1-C9 ...", flush=True)
    ae_bal_manifest = json.loads(
        (paths.AE_V1_DIR / "ae_bal" / "manifest_ae.json").read_text(encoding="utf-8"))
    paso2 = tm.compute_paso2_gates(gates, diagnostics, d5b, tails_,
                                   ae_bal_manifest)

    m["gates"] = gates
    m["diagnostics"] = diagnostics
    m["diag5b_by_cohort"] = d5b
    m["tails"] = tails_
    m["paso2_gates"] = paso2
    m["n_val_cells_by_source"] = val["n_by_source"]
    m["device"] = "cpu"
    m["determinism"] = {
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "vram_max_observed_mib": 0.0,
        "note": "re-evaluado en CPU para reproducción bit a bit; el "
                "entrenamiento fue en GPU (vram 120.3 MiB)",
    }

    txt = json.dumps(m, indent=2, ensure_ascii=False)
    MANIFEST_COPY.write_text(txt, encoding="utf-8")
    MANIFEST_DATA.write_text(txt, encoding="utf-8")

    print("\n===== REEVALUACIÓN CPU =====", flush=True)
    print("  blocking:", paso2["blocking"], flush=True)
    print("  k_def:", paso2["k_def"], flush=True)
    print("  gate2 real HR=%.4f ART_MBP=%.4f BIS=%.4f ETCO2=%.4f" % (
        gates["gate2"]["real"]["Solar8000/HR"],
        gates["gate2"]["real"]["Solar8000/ART_MBP"],
        gates["gate2"]["real"]["BIS/BIS"],
        gates["gate2"]["real"]["Primus/ETCO2"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
