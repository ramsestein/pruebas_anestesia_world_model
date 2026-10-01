"""PASO 2 — entrena ae_v2 (variante 50/50, la B) sobre real + v7.

Entrena el autoencoder de fisiología con los hiperparámetros IDÉNTICOS a
ae_bal (contrato_ae_v1.md, iter 3) pero con las cohortes sintéticas v7
(synthetic_v7, vaso_reinf_v7, cf_v7) vía paths.SYNTH_COHORTS, y escribe los
artefactos en data/ae_v2/ (layout plano: encoder.pt, decoder.pt,
norm_stats.json, manifest_ae.json).

NO modifica src/ae/physio_ae.py: usa pa.train("ae_bal"), pa.evaluate y
pa.diagnose tal cual. Los criterios de aceptación de la fase C están fijados
en el enunciado del paso 2; este script los COMPUTA y los GUARDA, no los
ajusta.

Uso:
    python scripts/paso2_train_ae_v2.py [--device cuda|cpu]

Salidas:
    data/ae_v2/                  encoder.pt, decoder.pt, norm_stats.json,
                                 manifest_ae.json, train_log.txt
    manifests/ae_v2_manifest.json     copia versionada del manifest
    manifests/ae_v2_norm_stats.json   copia versionada de norm_stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time as _time
from datetime import datetime
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import paths  # noqa: E402

# IMPORTANTE (Windows): importar ae.physio_ae (que importa pyarrow) ANTES de
# torch, para no romper pq.read_table (conflicto de DLL).
from ae import physio_ae as pa  # noqa: E402
import torch  # noqa: E402  (ya importado por pa en el orden correcto)

AE_V2_DIR = paths.DATA_ROOT / "ae_v2"
MANIFEST_COPY = paths.MANIFESTS_DIR / "ae_v2_manifest.json"
NORM_STATS_COPY = paths.MANIFESTS_DIR / "ae_v2_norm_stats.json"
TRAIN_LOG = AE_V2_DIR / "train_log.txt"

CLINICAL = ["Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS", "Primus/ETCO2"]
V7_COHORTS = ["synthetic_v7", "vaso_reinf_v7", "cf_v7"]

# Umbrales clínicos fijados en la fase C (no se ajustan post-hoc).
C2_THRESHOLDS = {
    "Solar8000/HR": 1.0,
    "Solar8000/ART_MBP": 1.0,
    "BIS/BIS": 0.5,
    "Primus/ETCO2": 0.5,
}
C1_TOLERANCE = 0.05  # tolerancia ya acordada en la iteración 3 del AE

MAX_VRAM_GIB = 6.0  # R6 del enunciado


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cap_vram(device: torch.device) -> None:
    """Aplica el presupuesto de VRAM acordado (máx. 6 GiB) si hay CUDA."""
    if device.type != "cuda":
        return
    props = torch.cuda.get_device_properties(device)
    total_gib = props.total_memory / (1024 ** 3)
    frac = min(1.0, MAX_VRAM_GIB / total_gib)
    try:
        torch.cuda.set_per_process_memory_fraction(frac, device)
        print(f"[ae_v2] VRAM cap: fracción {frac:.4f} sobre {total_gib:.1f} GiB "
              f"(<= {MAX_VRAM_GIB} GiB)", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[ae_v2] aviso: no se pudo aplicar el tope de VRAM: {exc}",
              flush=True)


def _peak_vram_mib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)


def escalones(per_variable: dict) -> dict:
    """En qué k cae cada variable clínica por debajo de su umbral."""
    thr = pa.DIAG5B_THRESHOLDS
    out = {}
    for v in pa.DIAG5B_VARS:
        curve = per_variable[v]
        k = next((i + 1 for i, x in enumerate(curve) if x < thr[v]), None)
        out[v] = k
    return out


def k_clinico_from(per_variable: dict) -> int | None:
    """k mínimo que cumple a la vez HR<1.0, ART_MBP<1.0, BIS<1.0, ETCO2<0.5."""
    thr = pa.DIAG5B_THRESHOLDS
    for k in range(1, pa.LATENT_DIM + 1):
        if all(per_variable[v][k - 1] < thr[v] for v in pa.DIAG5B_VARS):
            return k
    return None


def diag5b_by_cohort(model, norm_stats, excluded, device) -> dict:
    """Diag 5b sobre real y sobre cada cohorte v7 por separado (C6)."""
    out: dict = {}
    cohorts = ["real"] + V7_COHORTS
    for cohort in cohorts:
        print(f"[ae_v2] diag5b {cohort} ...", flush=True)
        vc = pa.load_cells([cohort], "val", excluded)
        gc = pa.evaluate(model, norm_stats, vc, device)
        group = "real" if cohort == "real" else "synthetic"
        d5b = gc["diag5b"]["cohorts"][group]
        out[cohort] = {
            "per_variable": d5b["per_variable"],
            "k_clinico": d5b["k_clinico"],
            "escalones": escalones(d5b["per_variable"]),
        }
    return out


def tails(model, norm_stats, val, device) -> dict:
    """C9: fracción de celdas v7 por encima del p99 del error real (4 clínicas)."""
    values = val["values"]
    masks = val["masks"]
    cell_source = np.repeat(val["case_source"], val["case_len"])
    zn = pa.normalize(values, norm_stats).astype(np.float32)
    zn[masks == 0] = 0.0
    z = pa._encode_all(model, zn, masks, device)
    recon = pa._decode_all(model, z, pa.LATENT_DIM, device)
    recon_phys = pa.denormalize(recon, norm_stats).astype(np.float32)
    err = np.where(masks == 1, np.abs(recon_phys - values), np.nan).astype(np.float32)
    out: dict = {}
    for track in CLINICAL:
        vi = pa.IMAGE_TRACKS.index(track)
        real_col = err[cell_source == "real", vi]
        real_col = real_col[np.isfinite(real_col)]
        p99 = float(np.percentile(real_col, 99))
        row = {"p99_real": p99}
        for cohort in V7_COHORTS:
            col = err[cell_source == cohort, vi]
            col = col[np.isfinite(col)]
            row[cohort] = {"n_cells": int(col.size),
                           "frac_above_p99": float(np.mean(col > p99))}
        out[track] = row
    return out


def compute_paso2_gates(gates: dict, diagnostics: dict,
                        diag5b: dict, tails_: dict,
                        ae_bal_manifest: dict) -> dict:
    """Computa los veredictos C1-C9 y los guarda (sin ajustarlos)."""
    ref_real = ae_bal_manifest["gates"]["gate2"]["real"]

    # C1 — la real no empeora (bloqueante).
    c1: dict = {"checks": {}, "pass": True}
    for t in CLINICAL:
        e = float(gates["gate2"]["real"][t])
        ref = float(ref_real[t])
        ok = e <= ref + C1_TOLERANCE
        c1["checks"][t] = {"ae_bal": ref, "ae_v2": e,
                           "diff": e - ref, "tolerance": C1_TOLERANCE, "ok": ok}
        c1["pass"] = c1["pass"] and ok

    # C2 — v7 por debajo de la resolución del monitor (bloqueante).
    c2: dict = {"checks": {}, "pass": True}
    for cohort in V7_COHORTS:
        row: dict = {}
        for t, thr in C2_THRESHOLDS.items():
            e = float(gates["gate2"][cohort][t])
            row[t] = {"error": e, "threshold": thr, "ok": e < thr}
            c2["pass"] = c2["pass"] and (e < thr)
        c2["checks"][cohort] = row

    # C3 — gate 1 (perfil de orden) sobre real y v7 (bloqueante).
    g1 = gates["gate1"]["profile"]
    c3 = {
        "real": {"monotone": g1["real"]["monotone"],
                 "k8_fraction": g1["real"]["k8_fraction"], "ok": g1["real"]["ok"]},
        "synthetic": {"monotone": g1["synthetic"]["monotone"],
                      "k8_fraction": g1["synthetic"]["k8_fraction"],
                      "ok": g1["synthetic"]["ok"]},
        "pass": bool(g1["real"]["ok"] and g1["synthetic"]["ok"]),
    }

    # C4 — gate 3 (|HR_sin_ART − HR_real| < 2 lpm) (bloqueante).
    g3 = gates["gate3"]
    c4 = {
        "err_hr_with_art": float(g3["err_hr_with_art"]),
        "err_hr_without_art": float(g3["err_hr_without_art"]),
        "hr_recon_diff": float(g3["hr_recon_diff"]),
        "n_cases": int(g3["n_cases"]),
        "n_cells": int(g3["n_cells"]),
        "threshold": 2.0,
        "ok": bool(g3["ok"]),
        "pass": bool(g3["ok"]),
    }

    # C5 — gates 4, 5, 7 y 8: los determina la suite de tests (no este script).
    c5 = {"pass": None, "note": "determinado por tests/test_paso2_ae_v2.py"}

    # C6 — diag 5b (informativo).
    c6 = diag5b

    # C7 — separabilidad en el latente (informativo).
    a1 = diagnostics["diag6"]["A1_support_overlap"]
    c7 = {
        "nn_overlap": a1,
        "cell_probe": gates["gate6"]["per_cell"],
    }

    # C8 — gate 6 (informativo).
    c8 = {
        "per_case": gates["gate6"]["per_case"],
        "per_cell": gates["gate6"]["per_cell"],
    }

    # C9 — colas (informativo).
    c9 = tails_

    blocking = {
        "C1": c1["pass"],
        "C2": c2["pass"],
        "C3": c3["pass"],
        "C4": c4["pass"],
        "C5": c5["pass"],
    }
    c5_computed_later = None  # se rellena desde los tests si se decide

    # k_def sobre ae_v2: max(k_clinico real, k_clinico de cada cohorte v7).
    k_list = [diag5b[c]["k_clinico"] for c in ["real"] + V7_COHORTS]
    k_list = [k for k in k_list if k is not None]
    k_def = int(max(k_list)) if k_list else None

    return {
        "C1_real_no_empeora": c1,
        "C2_v7_resolucion_monitor": c2,
        "C3_gate1_perfil": c3,
        "C4_gate3_art": c4,
        "C5_gates_4_5_7_8": c5,
        "C6_diag5b": c6,
        "C7_latente_separabilidad": c7,
        "C8_gate6": c8,
        "C9_colas": c9,
        "blocking": blocking,
        "k_def": k_def,
        "acceptance_criteria": {
            "C1": "gate2 real clínico <= ae_bal + 0.05 (HR/ART_MBP/BIS/ETCO2)",
            "C2": "gate2 por cohorte v7: HR<1.0, ART_MBP<1.0, BIS<0.5, ETCO2<0.5",
            "C3": "gate1 monotonía + k8_frac<=0.2 sobre real y v7",
            "C4": "gate3 |HR_sin_ART-HR_real| < 2 lpm",
            "C5": "gates 4/5/7/8 vía tests",
        },
    }


def build_manifest(result: dict, gates: dict, diagnostics: dict,
                   diag5b: dict, tails_: dict, paso2: dict,
                   device: torch.device, sha_physio: str,
                   sha_script: str, ae_bal_manifest: dict) -> dict:
    model = result["model"]
    enc_path = AE_V2_DIR / "encoder.pt"
    dec_path = AE_V2_DIR / "decoder.pt"
    hist = result["history"]
    torch_version = torch.__version__
    cuda_version = torch.version.cuda if device.type == "cuda" else None

    manifest = {
        "variant": "ae_v2",
        "base_variant": "ae_bal",
        "date": datetime.now().isoformat(),
        "seed": pa.SEED,
        "sha256_contract_ae": pa.sha256(pa.CONTRACT_PATH),
        "sha256_contract_tokens": pa.sha256(pa.TOKENS_CONTRACT_PATH),
        "sha256_physio_ae_py": sha_physio,
        "sha256_train_script": sha_script,
        "sha256_tokens_v1_manifest": pa.sha256(pa.TOKENS_MANIFEST),
        "sha256_windows_v2_manifest": pa.sha256(pa.WINDOWS_ROOT / "manifest.json"),
        "sha256_split_parquet": pa.sha256(pa.WINDOWS_ROOT / "split.parquet"),
        "sha256_report_tokens_v4": pa.sha256(pa.REPORT_TOKENS_V4),
        "sha256_encoder": sha256(enc_path),
        "sha256_decoder": sha256(dec_path),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "determinism": {
            "cudnn_deterministic": bool(getattr(torch.backends.cudnn,
                                                "deterministic", False)),
            "cudnn_benchmark": bool(getattr(torch.backends.cudnn,
                                            "benchmark", False)),
            "vram_max_observed_mib": _peak_vram_mib(device),
        },
        "scheduler": "warmup_linear+reduce_on_plateau",
        "scheduler_reductions": hist["scheduler_reductions"],
        "final_lr": hist["final_lr"],
        "best_epoch": hist["best_epoch"],
        "hyperparameters": {
            "lr": pa.LR,
            "weight_decay": pa.WEIGHT_DECAY,
            "batch_size": pa.BATCH_SIZE,
            "max_epochs": pa.MAX_EPOCHS,
            "warmup_epochs": pa.WARMUP_EPOCHS,
            "patience": pa.PATIENCE,
            "plateau_factor": pa.PLATEAU_FACTOR,
            "plateau_patience": pa.PLATEAU_PATIENCE,
            "plateau_threshold": pa.PLATEAU_THRESHOLD,
            "min_lr": pa.MIN_LR,
            "latent_dim": pa.LATENT_DIM,
            "n_vars": pa.N_VARS,
        },
        "val_sample": {
            "seed": pa.VAL_SAMPLE_SEED,
            "n_real": pa.VAL_SAMPLE_REAL,
            "n_synth": pa.VAL_SAMPLE_SYNTH,
            "criterion": "real + synthetic (misma muestra para las dos variantes)",
        },
        "image_tracks": pa.IMAGE_TRACKS,
        "excluded_no_phase_marks_caseids": sorted(pa.load_tokens_manifest().get(
            "excluded_no_phase_marks_caseids", [])),
        "sin_celdas_caseids": sorted(pa.load_tokens_manifest().get(
            "sin_celdas_caseids", [])),
        "norm_stats": result["norm_stats"],
        "training": hist,
        "n_train_cells_by_source": result["n_train_cells_by_source"],
        "n_val_cells_by_source": result["n_val_cells_by_source"],
        "gates": gates,
        "diagnostics": diagnostics,
        "diag5b_by_cohort": diag5b,
        "tails": tails_,
        "paso2_gates": paso2,
        "acceptance": {
            "regression_tolerance_lpm": 0.05,
            "regression_tolerance_mmhg": 0.05,
            "C1_tolerance": C1_TOLERANCE,
            "rationale": "tolerancia de la fase C fijada de antemano "
                         "(iteración 3 del AE)",
            "introduced_in": "paso2",
            "decided_post_hoc": False,
        },
        "provenance_gaps": ae_bal_manifest.get("provenance_gaps", []),
        "ae_bal_reference_gate2_real": {
            t: ae_bal_manifest["gates"]["gate2"]["real"][t] for t in pa.IMAGE_TRACKS
        },
        "elapsed_train_s": result.get("elapsed_train_s"),
    }
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrena ae_v2 (variante 50/50).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    _cap_vram(device)

    # A2: sha256 de physio_ae.py tal como está en disco (Windows, CRLF).
    sha_physio = pa.sha256(Path(pa.__file__).resolve())
    sha_script = sha256(Path(__file__).resolve())
    print(f"[ae_v2] sha256 physio_ae.py = {sha_physio}", flush=True)
    print(f"[ae_v2] sha256 train_script = {sha_script}", flush=True)

    ae_bal_manifest = json.loads(
        (paths.AE_DIR / "ae_bal" / "manifest_ae.json").read_text(encoding="utf-8"))

    t0 = _time.time()
    print(f"[ae_v2] entrenando variante ae_bal sobre real + v7 en {device} ...",
          flush=True)
    result = pa.train("ae_bal", device)

    # B1: el entrenamiento no puede haber parado por el tope de épocas.
    hist = result["history"]
    print(f"[ae_v2] épocas={hist['epochs_run']} best_epoch={hist['best_epoch']} "
          f"stop_reason={hist['stop_reason']} reducciones="
          f"{hist['scheduler_reductions']} final_lr={hist['final_lr']:.2e}",
          flush=True)
    if hist["stop_reason"] == "max_epochs":
        print("[ae_v2] ERROR: paró por tope de épocas (no convergido). "
              "No se evalúa.", flush=True)
        return 2

    model = result["model"].to(device)
    excluded = pa.excluded_caseids()

    print("[ae_v2] cargando val completo (real + v7) ...", flush=True)
    val = pa.load_cells(paths.ALL_COHORTS, "val", excluded)
    result["n_val_cells_by_source"] = val["n_by_source"]

    print("[ae_v2] evaluando gates 1-3, 6 y diag5 ...", flush=True)
    gates = pa.evaluate(model, result["norm_stats"], val, device)

    print("[ae_v2] diagnóstico 6 (separabilidad) ...", flush=True)
    diagnostics = pa.diagnose(model, result["norm_stats"], val, device)

    print("[ae_v2] C6 diag5b por cohorte ...", flush=True)
    d5b = diag5b_by_cohort(model, result["norm_stats"], excluded, device)

    print("[ae_v2] C9 colas ...", flush=True)
    tails_ = tails(model, result["norm_stats"], val, device)

    print("[ae_v2] computando veredictos C1-C9 ...", flush=True)
    paso2 = compute_paso2_gates(gates, diagnostics, d5b, tails_, ae_bal_manifest)

    AE_V2_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), AE_V2_DIR / "encoder.pt")
    torch.save(model.decoder.state_dict(), AE_V2_DIR / "decoder.pt")
    (AE_V2_DIR / "norm_stats.json").write_text(
        json.dumps(result["norm_stats"], indent=2), encoding="utf-8")

    manifest = build_manifest(result, gates, diagnostics, d5b, tails_, paso2,
                              device, sha_physio, sha_script, ae_bal_manifest)
    manifest["elapsed_total_s"] = round(_time.time() - t0, 2)
    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
    (AE_V2_DIR / "manifest_ae.json").write_text(manifest_json, encoding="utf-8")
    MANIFEST_COPY.write_text(manifest_json, encoding="utf-8")
    NORM_STATS_COPY.write_text(
        json.dumps(result["norm_stats"], indent=2), encoding="utf-8")

    print("\n===== PASO 2 — VEREDICTOS (fase C) =====", flush=True)
    for key in ("C1", "C2", "C3", "C4"):
        v = paso2["blocking"][key]
        print(f"  {key}: {'PASA' if v else 'FALLA'}", flush=True)
    print(f"  C5: {'via tests' if paso2['blocking']['C5'] is None else paso2['blocking']['C5']}",
          flush=True)
    print(f"  k_def (ae_v2) = {paso2['k_def']}", flush=True)
    print(f"  gate2 real HR={gates['gate2']['real']['Solar8000/HR']:.4f} "
          f"ART_MBP={gates['gate2']['real']['Solar8000/ART_MBP']:.4f} "
          f"BIS={gates['gate2']['real']['BIS/BIS']:.4f} "
          f"ETCO2={gates['gate2']['real']['Primus/ETCO2']:.4f}", flush=True)
    for cohort in V7_COHORTS:
        g2 = gates["gate2"][cohort]
        print(f"  gate2 {cohort:<14} HR={g2['Solar8000/HR']:.4f} "
              f"ART_MBP={g2['Solar8000/ART_MBP']:.4f} BIS={g2['BIS/BIS']:.4f} "
              f"ETCO2={g2['Primus/ETCO2']:.4f}", flush=True)
    print(f"  artefactos en {AE_V2_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
