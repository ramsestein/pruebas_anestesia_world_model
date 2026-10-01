"""PASO 2 — FASE A: preparación de ae_v2 (composición y normalización).

Sin entrenar nada. Calcula y guarda:
  A3  composición del conjunto de entrenamiento: casos y celdas por cohorte
      en train y val, y la proporción efectiva que aporta cada cohorte
      sintética a la mitad sintética (muestreo uniforme por caso sobre la
      unión de las tres).
  A4  estadísticos de normalización sobre el split train de real + v7 con
      máscara 1 (misma regla que ae_bal), guardados en data/ae_v2/norm_stats.json
      y manifests/ae_v2_norm_stats.json, con tabla lado a lado con ae_bal.

Uso:
    python scripts/paso2_prep_ae_v2.py

NO modifica src/ae/physio_ae.py. Solo lee datos y escribe norm_stats.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import paths  # noqa: E402

# IMPORTANTE (Windows): importar ae.physio_ae (pyarrow) ANTES de torch.
from ae import physio_ae as pa  # noqa: E402
import torch  # noqa: E402  (ya importado por pa en el orden correcto)

AE_V2_DIR = paths.DATA_ROOT / "ae_v2"
NORM_STATS_COPY = paths.MANIFESTS_DIR / "ae_v2_norm_stats.json"


def cohort_composition(split: str, excluded) -> dict:
    """Casos y celdas por cohorte para un split, tras filtrar mantenimiento y
    exclusiones (misma regla que load_cells)."""
    data = pa.load_cells(paths.ALL_COHORTS, split, excluded)
    case_source = data["case_source"]  # por caso
    n_cases_by_source = {}
    for s in paths.ALL_COHORTS:
        n_cases_by_source[s] = int((case_source == s).sum())
    n_cells_by_source = dict(data["n_by_source"])
    # rellenar cohortes ausentes con 0
    for s in paths.ALL_COHORTS:
        n_cells_by_source.setdefault(s, 0)
    return {"n_cases": n_cases_by_source, "n_cells": n_cells_by_source}


def main() -> int:
    excluded = pa.excluded_caseids()
    print("excluded caseids:", len(excluded), flush=True)

    # ---- A3: composición ----
    print("[prep] composición train ...", flush=True)
    train = cohort_composition("train", excluded)
    print("[prep] composición val ...", flush=True)
    val = cohort_composition("val", excluded)

    n_cases_train = train["n_cases"]
    n_cases_val = val["n_cases"]
    n_cells_train = train["n_cells"]
    n_cells_val = val["n_cells"]

    # Proporción efectiva de cada cohorte sintética en la mitad sintética:
    # muestreo uniforme por CASO sobre la unión de las tres -> la probabilidad
    # de cada cohorte es (nº de casos de la cohorte) / (nº total de casos sintéticos).
    synth_total_cases = sum(n_cases_train[s] for s in pa.SYNTH_SOURCES)
    effective = {
        s: (n_cases_train[s] / synth_total_cases if synth_total_cases else 0.0)
        for s in pa.SYNTH_SOURCES
    }

    # ---- A4: normalización ----
    print("[prep] norm_stats real + v7 (train) ...", flush=True)
    parts = (pa.iter_partitions(["real"], "train")
             + pa.iter_partitions(pa.SYNTH_SOURCES, "train"))
    stats = pa.compute_norm_stats(parts, excluded)

    AE_V2_DIR.mkdir(parents=True, exist_ok=True)
    (AE_V2_DIR / "norm_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    NORM_STATS_COPY.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    ae_bal_stats = json.loads(
        (paths.AE_V1_DIR / "ae_bal" / "norm_stats.json").read_text(encoding="utf-8"))

    # ---- imprimir tablas para el informe ----
    print("\n===== A3 — COMPOSICIÓN (casos | celdas, tras filtro mantenimiento) =====",
          flush=True)
    print(f"{'cohort':<16}{'train_casos':>12}{'train_celdas':>14}"
          f"{'val_casos':>10}{'val_celdas':>12}{'efectiva':>10}", flush=True)
    for s in paths.ALL_COHORTS:
        eff = f"{effective[s]:.4f}" if s in effective else "-"
        print(f"{s:<16}{n_cases_train[s]:>12}{n_cells_train[s]:>14}"
              f"{n_cases_val[s]:>10}{n_cells_val[s]:>12}{eff:>10}", flush=True)
    print(f"  casos sintéticos train total: {synth_total_cases}", flush=True)

    print("\n===== A4 — NORM STATS (lado a lado ae_bal | ae_v2) =====", flush=True)
    print(f"{'track':<24}{'ae_bal_mean':>13}{'ae_v2_mean':>13}"
          f"{'ae_bal_std':>12}{'ae_v2_std':>12}", flush=True)
    for t in pa.IMAGE_TRACKS:
        bm = ae_bal_stats[t]["mean"]
        bs = ae_bal_stats[t]["std"]
        nm = stats[t]["mean"]
        ns = stats[t]["std"]
        print(f"{t:<24}{bm:>13.5f}{nm:>13.5f}{bs:>12.5f}{ns:>12.5f}", flush=True)

    print(f"\n  norm_stats guardado en {AE_V2_DIR / 'norm_stats.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
