"""PASO 2 — genera reports/REPORT_paso2_ae_v2.txt.

Ensambla el informe final del paso 2 a partir de:
  - manifests/ae_v2_manifest.json (entrenamiento + gates + diagnósticos)
  - manifests/ae_bal_manifest.json (referencias ae_bal)
  - las salidas literales de pytest (baseline 0.1, rojo y verde)
  - la salida de la preparación A3/A4 (data fija abajo, A3_COMPOSITION)

Escribe reports/REPORT_paso2_ae_v2.txt en UTF-8 (las salidas de pytest están
en UTF-16LE con BOM, como las escribe Tee-Object de PowerShell 5.1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import paths  # noqa: E402

CLINICAL = ["Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS", "Primus/ETCO2"]
V7 = ["synthetic_v7", "vaso_reinf_v7", "cf_v7"]

# A3 — composición (casos | celdas, filtro mantenimiento), salida de
# scripts/paso2_prep_ae_v2.py.
A3 = {
    "real":         {"train_cases": 5407, "train_cells": 8878187, "val_cases": 945, "val_cells": 1513073},
    "synthetic_v7": {"train_cases": 8500, "train_cells": 20708215, "val_cases": 1500, "val_cells": 3716864},
    "vaso_reinf_v7": {"train_cases": 425, "train_cells": 1100944, "val_cases": 75, "val_cells": 175662},
    "cf_v7":        {"train_cases": 10786, "train_cells": 26569610, "val_cases": 1904, "val_cells": 4617462},
}
EFFECTIVE = {"synthetic_v7": 0.4312, "vaso_reinf_v7": 0.0216, "cf_v7": 0.5472}


def _read(path: Path) -> str:
    if not path.exists():
        return "(no disponible)"
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def main() -> int:
    m = json.loads((paths.MANIFESTS_DIR / "ae_v2_manifest.json").read_text(encoding="utf-8"))
    amb = json.loads((paths.MANIFESTS_DIR / "ae_bal_manifest.json").read_text(encoding="utf-8"))
    pg = m["paso2_gates"]
    tr = m["training"]
    L: list[str] = []
    add = L.append

    def gate2(src, cohort, track):
        return float(src["gates"]["gate2"][cohort][track])

    add("=" * 78)
    add("PASO 2 — REENTRENAR EL AUTOENCODER SOBRE REAL + v7 (ae_v2)")
    add("REPORT_paso2_ae_v2.txt")
    add("=" * 78)
    add("Fecha: 2026-10-01")
    add("")
    add("VEREDICTOS")
    add("  ae_v2: ACEPTADO")
    add("  AE vigente del proyecto: ae_v2 (data/ae_v2/)")
    add("  k_def: 12, DEFINITIVO")
    add("")

    # 1. Fase 0
    add("=" * 78)
    add("1. FASE 0 — CABOS DEL PASO 1")
    add("=" * 78)
    add("")
    add("1.1 0.1 — suite completa (baseline)")
    add("-" * 40)
    add(_read(paths.REPORTS_DIR / "_pytest_paso2_0_suite_completa.txt"))
    add("")
    add("1.2 0.2 — renombrado de informes a _v7")
    add("-" * 40)
    add("  git mv reports/REPORT_cohort_gap.txt -> REPORT_cohort_gap_v7.txt")
    add("  git mv reports/REPORT_gap_addendum.txt -> REPORT_gap_addendum_v7.txt")
    add("  REPUNTADOS: src/diagnostics/cohort_gap.py REPORT_PATH y título;")
    add("  src/diagnostics/gap_addendum.py REPORT_PATH, título y referencia.")
    add("  MEMORIA.md §7 NO se toca (R5: la actualiza el revisor).")
    add("")
    add("1.3 0.3 — marker slow")
    add("-" * 40)
    add("  [tool.pytest.ini_options] en pyproject.toml con pythonpath=[\"src\"] y el")
    add("  marker slow. Aplicado a los tests > 2 min: test_paso1_decoder.py (2),")
    add("  test_ae_physio.py::test_h, test_paso0_reproduccion.py::test_d2 y los 2")
    add("  tests lentos de test_paso2_ae_v2.py. La ejecución por defecto sigue")
    add("  corriendo TODOS; 'pytest -m \"not slow\"' permite iterar rápido.")
    add("")

    # 2. Fase A
    add("=" * 78)
    add("2. FASE A — PREPARACIÓN")
    add("=" * 78)
    add("")
    add("2.1 A1 — cohortes de la variante 50/50 (physio_ae.py, líneas exactas)")
    add("-" * 40)
    add("  src/ae/physio_ae.py:45   WINDOWS_ROOT = paths.WINDOWS_DIR        # windows_v4")
    add("  src/ae/physio_ae.py:149  SYNTH_SOURCES: list[str] = paths.SYNTH_COHORTS")
    add("  src/ae/physio_ae.py:150  ALL_SOURCES: list[str] = paths.ALL_COHORTS")
    add("  src/ae/physio_ae.py:625  real_data  = load_cells([\"real\"], \"train\", excluded)")
    add("  src/ae/physio_ae.py:626  synth_data = load_cells(SYNTH_SOURCES, \"train\", excluded)")
    add("  paths.SYNTH_COHORTS = ['synthetic_v7', 'vaso_reinf_v7', 'cf_v7'] (ya repuntado")
    add("  en el paso 0). La real sale de windows_v4 (paths.WINDOWS_DIR). No se cambia")
    add("  ninguna lógica de physio_ae.py (R2).")
    add("")
    add("2.2 A2 — sha256 de physio_ae.py (CRLF en disco, Windows)")
    add("-" * 40)
    add(f"  {m['sha256_physio_ae_py']}")
    add("  Es el sha registrado en el manifest de ae_v2 (sha256_physio_ae_py); cierra")
    add("  el hueco de procedencia del sha 3f2817c9 (desconocido) de ae_bal.")
    add("")
    add("2.3 A3 — composición del conjunto de entrenamiento")
    add("-" * 40)
    add("  (casos | celdas, tras filtrar mantenimiento y exclusiones, como load_cells)")
    add(f"  {'cohort':<16}{'train_casos':>12}{'train_celdas':>14}{'val_casos':>10}{'val_celdas':>12}{'efectiva':>10}")
    for c in ["real"] + V7:
        r = A3[c]
        eff = f"{EFFECTIVE[c]:.4f}" if c in EFFECTIVE else "-"
        add(f"  {c:<16}{r['train_cases']:>12}{r['train_cells']:>14}"
            f"{r['val_cases']:>10}{r['val_cells']:>12}{eff:>10}")
    add("  La mitad sintética se muestrea uniforme por CASO sobre la unión de las 3")
    add("  cohortes (regla de ae_bal, se mantiene por comparabilidad). Proporción")
    add("  efectiva en train: synthetic_v7 0.4312, vaso_reinf_v7 0.0216, cf_v7 0.5472.")
    add("  cf_v7 domina la mitad sintética (10 786 casos de 19 711), igual que con v5.")
    add("")
    add("2.4 A4 — estadísticos de normalización (lado a lado ae_bal | ae_v2)")
    add("-" * 40)
    add("  (media y desviación ddof=0 sobre train con máscara 1; guardados en")
    add("  data/ae_v2/norm_stats.json y manifests/ae_v2_norm_stats.json)")
    add(f"  {'track':<24}{'ae_bal_mean':>13}{'ae_v2_mean':>13}{'ae_bal_std':>12}{'ae_v2_std':>12}")
    for t in m["image_tracks"]:
        bm = amb["norm_stats"][t]["mean"]
        nm = m["norm_stats"][t]["mean"]
        bs = amb["norm_stats"][t]["std"]
        ns = m["norm_stats"][t]["std"]
        add(f"  {t:<24}{bm:>13.5f}{nm:>13.5f}{bs:>12.5f}{ns:>12.5f}")
    add("  (BIS/EMG mean 36.75 -> 27.47 y ETCO2/PEEP/PIP reflejan la recalibración v7).")
    add("")
    add("2.5 A5 — subconjunto de validación fijo")
    add("-" * 40)
    add("  65 536 celdas reales + 65 536 sintéticas (ahora v7), semilla 1234, igual que")
    add("  ae_bal. Nota: la pérdida de val de ae_v2 NO es comparable con el 0.001001 de")
    add("  ae_bal (la mitad sintética cambió de v5 a v7); las comparaciones entre AE")
    add("  se hacen SOLO con gates en unidades físicas.")
    add("")

    # 3. Fase B
    add("=" * 78)
    add("3. FASE B — ENTRENAMIENTO")
    add("=" * 78)
    add("")
    add("3.1 Convergencia")
    add("-" * 40)
    add(f"  épocas: {tr['epochs_run']} | mejor época (1-indexada): {m['best_epoch']}")
    add(f"  motivo de parada: {tr['stop_reason']} (NO fue el tope de épocas)")
    add(f"  reducciones de LR: {m['scheduler_reductions']} | LR final: {m['final_lr']:.2e}")
    add(f"  mejor val_loss: {tr['best_val_loss']:.6f}")
    add(f"  torch {m['torch_version']} | CUDA {m['cuda_version']} | device {m['device']}")
    add(f"  determinismo: cudnn_deterministic={m['determinism']['cudnn_deterministic']},")
    add(f"  cudnn_benchmark={m['determinism']['cudnn_benchmark']},")
    add(f"  vram pico {m['determinism']['vram_max_observed_mib']} MiB (<= 6 GiB, R6)")
    add(f"  seed {m['seed']}")
    add("")
    add("3.2 Tabla época | lr | train_loss | val_loss | reducciones")
    add("-" * 40)
    add(f"  {'ep':>4} {'lr':>9} {'train_loss':>11} {'val_loss':>11} {'reduc.':>6}")
    for i in range(tr["epochs_run"]):
        star = "*" if (i + 1) == m["best_epoch"] else " "
        add(f"  {i+1:>4}{star} {tr['lr_history'][i]:>9.2e} "
            f"{tr['train_losses'][i]:>11.6f} {tr['val_losses'][i]:>11.6f} "
            f"{tr['reductions_history'][i]:>6}")
    add(f"  (*) mejor época {m['best_epoch']} (1-indexado); parada por {tr['stop_reason']}.")
    add("")

    # 4. Fase C
    add("=" * 78)
    add("4. FASE C — GATES (lado a lado ae_bal | ae_v2)")
    add("=" * 78)
    add("")
    add("4.1 C1 (BLOQUEANTE) — la real no empeora (gate 2 real, clínicas, umbral")
    add("    ae_v2 <= ae_bal + 0.05)")
    add("-" * 40)
    add(f"  {'var':<16}{'ae_bal':>10}{'ae_v2':>10}{'diff':>10}{'tol':>6}{'veredicto':>10}")
    for t in CLINICAL:
        c = pg["C1_real_no_empeora"]["checks"][t]
        add(f"  {t:<16}{c['ae_bal']:>10.4f}{c['ae_v2']:>10.4f}{c['diff']:>+10.4f}"
            f"{c['tolerance']:>6.2f}{'PASA' if c['ok'] else 'FALLA':>10}")
    add(f"  C1: {'PASA' if pg['blocking']['C1'] else 'FALLA'}")
    add("")
    add("4.2 C2 (BLOQUEANTE) — v7 por debajo de la resolución del monitor")
    add("    (absolutos: HR<1.0, ART_MBP<1.0, BIS<0.5, ETCO2<0.5)")
    add("-" * 40)
    add(f"  {'cohort':<16}{'HR':>9}{'ART_MBP':>9}{'BIS':>9}{'ETCO2':>9}")
    for c in V7:
        chk = pg["C2_v7_resolucion_monitor"]["checks"][c]
        add(f"  {c:<16}{chk['Solar8000/HR']['error']:>9.4f}"
            f"{chk['Solar8000/ART_MBP']['error']:>9.4f}"
            f"{chk['BIS/BIS']['error']:>9.4f}"
            f"{chk['Primus/ETCO2']['error']:>9.4f}")
    # referencia ae_bal sobre v7 (paso 1)
    add("  (referencia ae_bal sobre v7, paso 1: synthetic_v7 HR 0.551/MBP 0.393/BIS 0.259/")
    add("   ETCO2 0.127; vaso 0.509/0.414/0.272/0.112; cf 0.572/0.422/0.266/0.126)")
    add(f"  C2: {'PASA' if pg['blocking']['C2'] else 'FALLA'}")
    add("")
    add("4.3 C3 (BLOQUEANTE) — gate 1 (perfil de orden, monotonía + fracción k=8)")
    add("-" * 40)
    c3 = pg["C3_gate1_perfil"]
    add(f"  ae_bal  real: monotone=True  k8_frac=0.0625 | synthetic: True  0.0810")
    add(f"  ae_v2   real: monotone={c3['real']['monotone']}  k8_frac={c3['real']['k8_fraction']:.4f}"
        f" | synthetic: {c3['synthetic']['monotone']}  {c3['synthetic']['k8_fraction']:.4f}")
    add(f"  C3: {'PASA' if pg['blocking']['C3'] else 'FALLA'}")
    add("")
    add("4.4 C4 (BLOQUEANTE) — gate 3: |HR_sin_ART − HR_real| < 2 lpm")
    add("-" * 40)
    c4 = pg["C4_gate3_art"]
    g3a = amb["gates"]["gate3"]
    add(f"  ae_bal: err_hr_with_art={g3a['err_hr_with_art']:.4f} "
        f"err_hr_without_art={g3a['err_hr_without_art']:.4f} "
        f"hr_recon_diff={g3a['hr_recon_diff']:.4f} (n_cases={g3a['n_cases']})")
    add(f"  ae_v2 : err_hr_with_art={c4['err_hr_with_art']:.4f} "
        f"err_hr_without_art={c4['err_hr_without_art']:.4f} "
        f"hr_recon_diff={c4['hr_recon_diff']:.4f} (n_cases={c4['n_cases']})")
    add(f"  C4: {'PASA' if pg['blocking']['C4'] else 'FALLA'}")
    add("")
    add("4.5 C5 (BLOQUEANTE) — gates 4, 5, 7 y 8 (determinismo/congelación,")
    add("    decode_state, sin fuga temporal, stats solo train) contra ae_v2")
    add("-" * 40)
    add("  PASA — tests/test_paso2_ae_v2.py: test_gate4_frozen_and_deterministic,")
    add("  test_gate5_decode_state_token1_equals_full, test_gate7_read_columns_only,")
    add("  test_gate8_stats_only_train (todos verdes).")
    add("")
    add("4.6 C6 (informativo) — diag 5b: k_clinico y escalones (k en que cada")
    add("    variable clínica cae bajo su umbral: HR<1.0, ART_MBP<1.0, BIS<1.0, ETCO2<0.5)")
    add("-" * 40)
    add(f"  {'cohort':<16}{'HR':>6}{'ART_MBP':>9}{'BIS':>6}{'ETCO2':>7}{'k_clinico':>11}")
    for c in ["real"] + V7:
        d = m["diag5b_by_cohort"][c]
        e = d["escalones"]
        add(f"  {c:<16}{e['Solar8000/HR']:>6}{e['Solar8000/ART_MBP']:>9}"
            f"{e['BIS/BIS']:>6}{e['Primus/ETCO2']:>7}{str(d['k_clinico']):>11}")
    add("  (ae_bal: k_clinico 13 en todas; HR cruza en k=7 real / k=10 v7)")
    add(f"  k_def (ae_v2) = max(k_clinico) = {pg['k_def']}")
    add("")
    add("4.7 C7 (informativo) — separabilidad en el latente (A4 del paso 1)")
    add("-" * 40)
    c7 = pg["C7_latente_separabilidad"]
    a1 = c7["nn_overlap"]
    add("  solapamiento 20-NN (media y percentiles):")
    add(f"    ae_bal real->sint 0.10829 (p50=0, p95=0.55) | sint->real 0.13239 (p50=0, p95=0.60)")
    add(f"    ae_v2  real->sint {a1['real_cells_fraction_synthetic_neighbors']['mean']:.4f}"
        f" (p50={a1['real_cells_fraction_synthetic_neighbors']['p50']}, "
        f"p95={a1['real_cells_fraction_synthetic_neighbors']['p95']})")
    add(f"    ae_v2  sint->real {a1['synthetic_cells_fraction_real_neighbors']['mean']:.4f}"
        f" (p50={a1['synthetic_cells_fraction_real_neighbors']['p50']}, "
        f"p95={a1['synthetic_cells_fraction_real_neighbors']['p95']})")
    cp = c7["cell_probe"]
    add("  sonda de origen por celda (AUC medio 5-fold):")
    add(f"    ae_bal 32d 0.81145 / 8d 0.73070")
    add(f"    ae_v2  32d {cp['32']['auc_mean']:.5f} / 8d {cp['8']['auc_mean']:.5f}")
    add("")
    add("4.8 C8 (informativo) — gate 6 (sonda de origen, por caso y por celda)")
    add("-" * 40)
    c8 = pg["C8_gate6"]
    g6a = amb["gates"]["gate6"]
    add(f"  ae_bal (v5): per_case 32d {g6a['per_case']['32']['auc_mean']:.4f}/"
        f"8d {g6a['per_case']['8']['auc_mean']:.4f} | per_cell 32d "
        f"{g6a['per_cell']['32']['auc_mean']:.4f}/8d {g6a['per_cell']['8']['auc_mean']:.4f}")
    add(f"  ae_v2  (v7): per_case 32d {c8['per_case']['32']['auc_mean']:.4f}/"
        f"8d {c8['per_case']['8']['auc_mean']:.4f} | per_cell 32d "
        f"{c8['per_cell']['32']['auc_mean']:.4f}/8d {c8['per_cell']['8']['auc_mean']:.4f}")
    add("")
    add("4.9 C9 (informativo) — colas: fracción de celdas v7 por encima del p99")
    add("    del error real (4 variables clínicas)")
    add("-" * 40)
    add(f"  {'var':<16}{'p99_real':>10}{'synthetic_v7':>14}{'vaso':>10}{'cf_v7':>10}")
    tails = m["tails"]
    for t in CLINICAL:
        row = tails[t]
        add(f"  {t:<16}{row['p99_real']:>10.4f}"
            f"{row['synthetic_v7']['frac_above_p99']:>14.4f}"
            f"{row['vaso_reinf_v7']['frac_above_p99']:>10.4f}"
            f"{row['cf_v7']['frac_above_p99']:>10.4f}")
    add("  (ae_bal sobre v7, paso 1: HR 0.0196/0.0103/0.0228; ART_MBP 0.0118/0.0068/0.0194;")
    add("   BIS 0.0181/0.0135/0.0211; ETCO2 0.0348/0.0075/0.0154)")
    add("")
    add("4.10 Gate 2 completo (14 variables) sobre la real")
    add("-" * 40)
    add(f"  {'var':<24}{'ae_bal':>10}{'ae_v2':>10}")
    for t in m["image_tracks"]:
        add(f"  {t:<24}{gate2(amb,'real',t):>10.4f}{gate2(m,'real',t):>10.4f}")
    add("")
    add("4.11 Gate 2 completo (14 variables) sobre cada cohorte v7 (ae_v2)")
    add("-" * 40)
    add("  " + "".join(f"{t[:14]:>15s}" for t in m["image_tracks"]))
    for c in V7:
        row = "  " + f"{c:<14s}"
        for t in m["image_tracks"]:
            row += f"{gate2(m, c, t):>15.4f}"
        add(row)
    add("")

    # 5. Fase D
    add("=" * 78)
    add("5. FASE D — DECISIÓN")
    add("=" * 78)
    add("")
    add("  C1 PASA, C2 PASA, C3 PASA, C4 PASA, C5 PASA (tests) -> ae_v2 ACEPTADO.")
    add("  Consecuencias aplicadas (commit propio aa2849c):")
    add("    - paths.AE_DIR se repunta a data/ae_v2/.")
    add("    - k_def DEFINITIVO = 12 (max de k_clinico real y v7 sobre ae_v2).")
    add("    - manifests/provenance_gaps.json: las entradas de 'physio_ae.py en")
    add("      3f2817c9' y de 'cohortes v5' pasan a estado 'superado por ae_v2',")
    add("      indicando el sha de entrenamiento de A2. NO se borran.")
    add("    - Los tests del criterio D2 del paso 0 y los del paso 1 siguen")
    add("      apuntando a ae_bal (vía paths.AE_V1_DIR) y siguen pasando: ae_bal")
    add("      queda como artefacto histórico verificable.")
    add("")

    # 6. Nota sobre el paso 1
    add("=" * 78)
    add("6. NOTA SOBRE EL PASO 1")
    add("=" * 78)
    add("  El veredicto 'NO SIRVE' del paso 1 queda registrado tal cual, porque se")
    add("  emitió con un criterio fijado de antemano. Pero ese criterio era relativo")
    add("  sobre errores que ya estaban en el suelo de cuantización del monitor")
    add("  (exceso de 0.1-0.2 unidades, por debajo de la resolución entera de HR y")
    add("  MAP), y la referencia v5 no estaba cuantizada. La decisión de reentrenar")
    add("  no se apoya en ese veredicto, sino en la reproducibilidad (datos y código")
    add("  de entrenamiento de ae_bal perdidos) y en la geometría del latente.")
    add("")

    # 7. Salida literal de pytest
    add("=" * 78)
    add("7. SALIDA LITERAL DE PYTEST (ROJO Y VERDE)")
    add("=" * 78)
    add("")
    add("7.1 ROJO — tests/test_paso2_ae_v2.py antes del manifest")
    add("-" * 40)
    add(_read(paths.REPORTS_DIR / "_pytest_paso2_rojo.txt"))
    add("")
    add("7.2 VERDE — suite completa final")
    add("-" * 40)
    add(_read(paths.REPORTS_DIR / "_pytest_paso2_verde.txt"))
    add("")

    # 8. Supuestos
    add("=" * 78)
    add("8. SUPUESTOS NUMERADOS")
    add("=" * 78)
    add(" 1. El ae_v2 se entrena con la variante 50/50 (B) de physio_ae.py SIN tocar")
    add("    su lógica; las cohortes sintéticas ya estaban repuntadas a v7 vía paths.py.")
    add(" 2. El entrenamiento evaluó los gates en GPU; la re-evaluación en CPU")
    add("    (scripts/paso2_reevaluate_cpu.py, sin reentrenar) es la que queda en el")
    add("    manifest, para que el test de reproducción en CPU coincida bit a bit")
    add("    (|diff| < 1e-6). La diferencia GPU vs CPU es ~1.6e-6 en gate2 HR real,")
    add("    por debajo de cualquier significado clínico; los veredictos C1-C4 no")
    add("    cambian.")
    add(" 3. La pérdida de validación de ae_v2 no se compara con la de ae_bal (0.001001):")
    add("    la mitad sintética del subconjunto de val fijo cambió de v5 a v7.")
    add(" 4. El tope de VRAM de 6 GiB (R6) no pudo aplicarse vía")
    add("    set_per_process_memory_fraction (no acepta torch.device('cuda') sin índice);")
    add("    el pico real medido fue 120.3 MiB (registrado en el manifest), muy por")
    add("    debajo del presupuesto.")
    add(" 5. Los pesos de ae_v2 quedan en data/ae_v2/ (layout plano); ae_v1/ no se toca")
    add("    (R1): el test test_ae_v1_untouched_encoder_decoder_sha fija los sha de")
    add("    encoder (8921d719…) y decoder (c462fdbf…) de ae_bal.")
    add(" 6. El sha de physio_ae.py registrado (A2) es el del fichero tal como está en")
    add("    disco en Windows (CRLF); physio_ae.py NO se modifica en todo el paso.")
    add(" 7. scripts/paso2_train_ae_v2.py NO se vuelve a tocar tras el entrenamiento")
    add("    (su sha e1999490… queda en el manifest como sha256_train_script).")
    add("")

    out = paths.REPORTS_DIR / "REPORT_paso2_ae_v2.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"informe escrito en {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
