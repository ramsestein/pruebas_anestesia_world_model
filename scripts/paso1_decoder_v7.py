"""PASO 1 — decoder congelado sobre v7 (FASES A y B).

Mide, con las funciones de src/ae/physio_ae.py (las mismas del criterio D2 del
paso 0) y el encoder/decoder de ae_bal CONGELADOS:
  A1  gate 2 sobre las cohortes v7 (val, mantenimiento, exclusiones del
      manifest de tokens_v1) — error absoluto medio por caso, mediana entre
      casos, unidades físicas, solo celdas máscara 1, 14 variables.
  A2  criterio de decisión (fijado en el enunciado, no se ajusta después).
  A3  colas (informativo): fracción de celdas por encima del p99 real.
  A4  solapamiento 20-NN en el latente + sonda de origen por celda.
  B1  reproducción del diag 5b real (k_clinico = 13, tolerancia +-0.0005).
  B2  diag 5b sobre v7 por cohorte y unión; escalones por variable.
  B3  k_def = max(k_clinico real, k_clinico de cada cohorte v7).

Salidas: data/diagnostics/paso1_decoder_v7/paso1_decoder_v7.json y
manifests/paso1_decoder_v7.json.

NO reentrena nada. NO modifica src/ae/physio_ae.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import paths  # noqa: E402

# IMPORTANTE (Windows): importar ae.physio_ae (que importa pyarrow) ANTES de
# torch, para no romper pq.read_table.
from ae import physio_ae as pa  # noqa: E402
import torch  # noqa: E402  (ya importado por pa en el orden correcto)

CLINICAL = ["Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS", "Primus/ETCO2"]
V7_COHORTS = ["synthetic_v7", "vaso_reinf_v7", "cf_v7"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def k_clinico_from(per_variable: dict) -> int | None:
    """k mínimo que cumple a la vez HR<1.0, ART_MBP<1.0, BIS<1.0, ETCO2<0.5."""
    thr = pa.DIAG5B_THRESHOLDS
    curves = {v: per_variable[v] for v in pa.DIAG5B_VARS}
    for k in range(1, pa.LATENT_DIM + 1):
        if all(curves[v][k - 1] < thr[v] for v in pa.DIAG5B_VARS):
            return k
    return None


def escalones(per_variable: dict) -> dict:
    """En qué k cae cada variable clínica por debajo de su umbral."""
    thr = pa.DIAG5B_THRESHOLDS
    out = {}
    for v in pa.DIAG5B_VARS:
        curve = per_variable[v]
        k = next((i + 1 for i, x in enumerate(curve) if x < thr[v]), None)
        out[v] = k
    return out


def main() -> int:
    device = torch.device("cpu")
    model = pa.load_model("ae_bal", "cpu").to(device)
    model.eval()
    norm_stats = pa.load_norm_stats("ae_bal")
    excluded = pa.excluded_caseids()
    manifest = json.loads((paths.AE_DIR / "ae_bal" / "manifest_ae.json").read_text(encoding="utf-8"))
    ref_gate2 = manifest["gates"]["gate2"]

    print("cargando val completo (real + v7) ...", flush=True)
    val = pa.load_cells(paths.ALL_COHORTS, "val", excluded)

    print("evaluando (gate2, gate6, diag5b) ...", flush=True)
    gates = pa.evaluate(model, norm_stats, val, device)

    # ---- A1: gate 2 sobre v7 ----
    gate2 = gates["gate2"]
    a1 = {c: gate2[c] for c in paths.ALL_COHORTS}
    a1_ref = {c: ref_gate2[c] for c in
              ["real", "synthetic_v5", "vaso_reinf_v5", "cf_v5"]}

    # ---- A2: criterio de decisión ----
    a2: dict = {"checks": {}, "fails": []}
    # construir referencias por cohorte v7 -> cohorte v5 equivalente
    v5_equiv = {"synthetic_v7": "synthetic_v5",
                "vaso_reinf_v7": "vaso_reinf_v5",
                "cf_v7": "cf_v5"}
    for cohort in V7_COHORTS:
        v5 = v5_equiv[cohort]
        for track in pa.IMAGE_TRACKS:
            e = gate2[cohort][track]
            ref = max(ref_gate2["real"][track], ref_gate2[v5][track])
            if track in CLINICAL:
                lim = 1.5 * ref
            else:
                lim = 2.0 * ref
            ok = (e <= lim)
            a2["checks"][f"{cohort}|{track}"] = {
                "error_v7": e, "ref_real": ref_gate2["real"][track],
                f"ref_{v5}": ref_gate2[v5][track], "ref_max": ref,
                "limit": lim, "ok": ok}
            if not ok:
                a2["fails"].append({"cohort": cohort, "track": track,
                                    "error_v7": e, "limit": lim, "ratio": e / ref})
    # umbrales absolutos del gate 2
    abs_checks = {}
    for cohort in V7_COHORTS:
        hr = gate2[cohort]["Solar8000/HR"]
        mbp = gate2[cohort]["Solar8000/ART_MBP"]
        abs_checks[cohort] = {"HR_ok": hr < 2.0, "ART_MBP_ok": mbp < 2.0,
                              "HR": hr, "ART_MBP": mbp}
        if not (hr < 2.0 and mbp < 2.0):
            a2["fails"].append({"cohort": cohort, "track": "abs_threshold",
                                "HR": hr, "ART_MBP": mbp})
    clinical_fail = any(f["track"] in CLINICAL or f["track"] == "abs_threshold"
                        for f in a2["fails"])
    if not a2["fails"]:
        verdict = "SIRVE"
    elif clinical_fail:
        verdict = "NO SIRVE"
    else:
        verdict = "SIRVE CON RESERVAS"
    a2["verdict"] = verdict
    a2["abs_thresholds"] = abs_checks

    # ---- A3: colas (informativo) ----
    print("A3 colas ...", flush=True)
    values = val["values"]
    masks = val["masks"]
    cell_source = np.repeat(val["case_source"], val["case_len"])
    zn = pa.normalize(values, norm_stats).astype(np.float32)
    zn[masks == 0] = 0.0
    z = pa._encode_all(model, zn, masks, device)
    recon = pa._decode_all(model, z, pa.LATENT_DIM, device)
    recon_phys = pa.denormalize(recon, norm_stats).astype(np.float32)
    err = np.where(masks == 1, np.abs(recon_phys - values), np.nan).astype(np.float32)
    a3: dict = {}
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
        a3[track] = row

    # ---- A4: diagnóstico 6 (solapamiento 20-NN + sonda por celda) ----
    print("A4 ...", flush=True)
    diag6 = pa.diagnose(model, norm_stats, val, device)["diag6"]
    a4 = {
        "nn_overlap": diag6["A1_support_overlap"],
        "cell_probe": gates["gate6"]["per_cell"],
    }

    # ---- B1: reproducción diag 5b real ----
    b1_real = gates["diag5b"]["cohorts"]["real"]
    ref_5b = manifest["gates"]["diag5b"]["cohorts"]["real"]
    b1 = {"k_clinico": b1_real["k_clinico"], "reproduced": b1_real["k_clinico"] == 13,
          "max_abs_diff": None}
    max_diff = 0.0
    for v in pa.DIAG5B_VARS:
        a = np.asarray(b1_real["per_variable"][v])
        b = np.asarray(ref_5b["per_variable"][v])
        max_diff = max(max_diff, float(np.nanmax(np.abs(a - b))))
    b1["max_abs_diff"] = max_diff
    b1["reproduced"] = (b1_real["k_clinico"] == 13) and (max_diff <= 0.0005)

    # ---- B2: diag 5b por cohorte v7 y unión ----
    b2: dict = {"union": {"per_variable": gates["diag5b"]["cohorts"]["synthetic"]["per_variable"],
                          "k_clinico": gates["diag5b"]["cohorts"]["synthetic"]["k_clinico"]},
                "cohorts": {}}
    b2["union"]["escalones"] = escalones(b2["union"]["per_variable"])
    for cohort in V7_COHORTS:
        print(f"B2 diag5b {cohort} ...", flush=True)
        vc = pa.load_cells([cohort], "val", excluded)
        gc = pa.evaluate(model, norm_stats, vc, device)
        d5b = gc["diag5b"]["cohorts"]["synthetic"]
        b2["cohorts"][cohort] = {"per_variable": d5b["per_variable"],
                                 "k_clinico": d5b["k_clinico"],
                                 "escalones": escalones(d5b["per_variable"])}

    # ---- B3: k_def ----
    k_list = [b1_real["k_clinico"]] + [b2["cohorts"][c]["k_clinico"] for c in V7_COHORTS]
    k_list = [k for k in k_list if k is not None]
    k_def = int(max(k_list)) if k_list else None
    b3 = {"k_clinico_real": b1_real["k_clinico"],
          "k_clinico_v7": {c: b2["cohorts"][c]["k_clinico"] for c in V7_COHORTS},
          "k_def": k_def,
          "previous_13": 13,
          "provisional": verdict == "NO SIRVE"}

    results = {
        "weights_sha256": {
            "encoder": sha256(paths.AE_DIR / "ae_bal" / "encoder.pt"),
            "decoder": sha256(paths.AE_DIR / "ae_bal" / "decoder.pt"),
        },
        "A1_gate2_v7": a1,
        "A1_references": a1_ref,
        "A2_verdict": a2,
        "A3_tails": a3,
        "A4": a4,
        "B1_diag5b_real_reproduction": b1,
        "B2_diag5b_v7": b2,
        "B3_k_def": b3,
    }

    out_dir = paths.DIAGNOSTICS_DIR / "paso1_decoder_v7"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paso1_decoder_v7.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (paths.MANIFESTS_DIR / "paso1_decoder_v7.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== VEREDICTO =====")
    print("decoder:", verdict)
    print("k_def =", k_def, "(anterior: 13)",
          "PROVISIONAL" if b3["provisional"] else "definitivo")
    print("A2 fails:", a2["fails"])
    print("escalones v7 union:", b2["union"]["escalones"])
    for c in V7_COHORTS:
        print(f"  {c}: k_clinico={b2['cohorts'][c]['k_clinico']} "
              f"escalones={b2['cohorts'][c]['escalones']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
