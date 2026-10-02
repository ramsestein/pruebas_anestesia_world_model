"""PASO 3b — genera reports/REPORT_paso3b_pares_cf.txt desde los artefactos.

Lee manifests/tokens_v2_cf_pairs_annotation.json y
data/tokens_v2/pairs_annotated.parquet (más los manifiestos v1/v2 de
procedencia) y escribe el informe. No recalcula nada.

Uso:
  python -m scripts.paso3b_report
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import paths
from tokens import cf_pairs as cp
from tokens import tokenize as tk

OUT = paths.REPORTS_DIR / "REPORT_paso3b_pares_cf.txt"
ANN_JSON = paths.MANIFESTS_DIR / "tokens_v2_cf_pairs_annotation.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    r = json.loads(ANN_JSON.read_text(encoding="utf-8"))
    df = cp.load_pairs_annotated()
    eff = df[df.lever_effective.astype(bool)]
    L: list[str] = []

    def w(s: str = "") -> None:
        L.append(s)

    w("REPORT_paso3b_pares_cf.txt")
    w("=" * 72)
    w("PASO 3b — anotación de los pares contrafactuales y adopción v2")
    w(f"fecha: {r['date']}")
    w()
    w("Artefactos")
    w("-" * 72)
    w(f"  pairs.parquet (NO reescrito, R2)  {cp.PAIRS_PARQUET}")
    w(f"    sha256 = {r['pairs_parquet_sha256']}")
    w(f"  pairs_annotated.parquet          {r['pairs_annotated_parquet']}")
    w(f"    sha256 = {_sha256(cp.PAIRS_ANNOTATED_PARQUET)}")
    w(f"  manifiesto de anotación          {ANN_JSON.relative_to(paths.REPO_ROOT)}")
    w(f"  módulo                           src/tokens/cf_pairs.py")
    w(f"  script                           scripts/paso3b_annotate_cf_pairs.py")
    w()
    w("1. Criterio corregido (B1)")
    w("-" * 72)
    w("  El generador aplica las Actions en t_action = split_t + 10 s")
    w("  (src/anessim/generate_f2_cf.py) y los overrides de ventilación en")
    w("  t_action = split_t (simulate.py, post = t > counterfactual_split_t).")
    w("  El paso 3 medía el efecto con post_split = t1 > split_t (la ventana que")
    w("  CONTIENE split_t, es decir ANTES de la acción), de ahí sus 6 palancas por")
    w("  debajo del 99 %.")
    w()
    w("  post_action(t1, t_action) = t1 >= floor(t_action / 5) * 5")
    w("  El redondeo a la rejilla de 5 s es necesario porque pk_v2 reporta en el")
    w("  inicio de la celda [5k, 5k+5) el efecto de las acciones que caen dentro")
    w("  (par 171501: acción en 6781.7 s, ce_efedrina ya salta en 6780).")
    w()
    w("  Alcance de C1/C2: ACCIONES PUNTUALES (pharma/learning/sevo).")
    w("  Las CONSIGNAS PERSISTENTES (overrides de ventilación: simulate.py aplica")
    w("  un delta acumulativo) sólo divergen cuando el plan base cambia la")
    w("  consigna, así que la ventana de t_action no puede contener el efecto.")
    w()
    w("2. Recuento")
    w("-" * 72)
    w(f"  pares anotados .......................... {r['n_pairs']}")
    w(f"  pares efectivos (conjunto CF) ........... {r['n_effective']}")
    w(f"  pares nulos ............................. {r['n_null']}")
    for k, v in sorted(r["null_causes"].items()):
        w(f"      {k:<21s} ............. {v}")
    w()

    w("2.1 Por palanca")
    w("-" * 72)
    w(f"  {'palanca':<16s} {'n':>5s} {'efect.':>7s} {'%':>8s}  nulos")
    for row in r["per_lever"]:
        nulos = ", ".join(f"{k}:{v}" for k, v in sorted(row["nulls"].items())) or "-"
        w(f"  {row['lever']:<16s} {row['n']:>5d} {row['effective']:>7d} "
          f"{row['pct_effective']:>8.3f}  {nulos}")
    w()

    w("3. Criterios")
    w("-" * 72)
    c1 = r["criteria"]["C1_divergencia_cruda_cerca_de_t_action"]
    w(f"  C1  |t_divergence_raw - t_action| <= {c1['umbral_s']:.0f} s")
    w(f"      acciones puntuales: {c1['n_effective_one_shot']} efectivos, "
      f"{c1['n_failures']} fallos -> {'PASA' if c1['ok'] else 'FALLA'}")
    pc = c1["persistent_consigna"]
    w(f"      consigna persistente (fuera de alcance): {pc['n_effective']} "
      f"efectivos, {pc['n_failures']} fallos")
    w(f"      {pc['nota']}")
    c2 = r["criteria"]["C2_efecto_en_ventana_de_t_action_o_siguiente"]
    w(f"  C2  efecto en la ventana de t_action o la siguiente (lag<="
      f"{c2['umbral_lag']}) en >= 99 % de los pares")
    w(f"      acciones puntuales: {c2['pct_lag_le_1']:.4f} % -> "
      f"{'PASA' if c2['ok'] else 'FALLA'}")
    w(f"      consigna persistente: "
      f"{c2['persistent_consigna']['pct_lag_le_1']:.4f} %")
    hist = c2["hist_lag"]
    w("      histograma de lag (acciones puntuales): "
      + ", ".join(f"{k}:{v}" for k, v in sorted(hist.items(), key=lambda x: int(x[0]))[:8])
      + (", ..." if len(hist) > 8 else ""))
    c3 = r["criteria"]["C3_sin_nulos_sin_explicar"]
    w(f"  C3  ningún null_cause = unexplained ............ "
      f"{c3['n_unexplained']} -> {'PASA' if c3['ok'] else 'FALLA'}")
    c4 = r["criteria"]["C4_prefijo_identico_hasta_t_action"]
    w(f"  C4  prefijo idéntico (features y máscaras) ..... "
      f"{c4['n_failures']} fallos -> {'PASA' if c4['ok'] else 'FALLA'}")
    w(f"      {c4['nota']}")
    w()

    w("4. Nulos no-PEEP documentados con evidencia")
    w("-" * 72)
    w(f"  n = {r['n_nulls_non_peep']}")
    w(f"  {'palanca':<13s} {'pair':>7s} {'t_action':>12s} {'causa':<17s} "
      f"{'verdad?':<7s} {'base':>10s}  petición")
    for e in r["nulls_non_peep"]:
        base = "-" if e["base_value_at_action"] is None else f"{e['base_value_at_action']:.4f}"
        w(f"  {e['lever']:<13s} {e['pair_id']:>7d} {e['t_action']:>12.2f} "
          f"{e['null_cause']:<17s} {str(e['truth_diff_after_action']):<7s} "
          f"{base:>10s}  {e['requested_change']}")
    w()

    w("5. Adopción por artefacto")
    w("-" * 72)
    w(f"  paths.PK_DIR       = {paths.PK_DIR.relative_to(paths.DATA_ROOT).as_posix()}"
      f"  (== PK_V2_DIR: {paths.PK_DIR == paths.PK_V2_DIR})")
    w(f"  paths.CONTEXT_DIR  = {paths.CONTEXT_DIR.relative_to(paths.DATA_ROOT).as_posix()}"
      f"  (== CONTEXT_V2_DIR: {paths.CONTEXT_DIR == paths.CONTEXT_V2_DIR})")
    w(f"  paths.TOKENS_DIR   = {paths.TOKENS_DIR.relative_to(paths.DATA_ROOT).as_posix()}"
      f"  (== TOKENS_V2_DIR: {paths.TOKENS_DIR == paths.TOKENS_V2_DIR})")
    w()
    w("  Fase 0.4 (¿se mezclan versiones?):")
    w("    - src/tokens/pk_tokens.py  escribe en paths.PK_DIR  -> pk_v2 (adoptado).")
    w("    - src/tokens/context_vocab.py escribe en paths.CONTEXT_DIR -> context_v2.")
    w("    - src/tokens/tokenize.py LEE pk_v2 + context_v2 y ESCRIBE en")
    w("      paths.TOKENS_DIR. Esa es la ÚNICA ruta que mezcla versiones y queda")
    w("      cerrada en la fase D, cuando TOKENS_DIR pasa a tokens_v2. Como el")
    w("      tokenizador NO se re-ejecuta en el paso 3b (R1: no se regenera nada),")
    w("      ningún artefacto en disco mezcla versiones.")
    w()

    w("6. Conjunto CF de entrenamiento")
    w("-" * 72)
    w("  El conjunto CF son los pares con lever_effective = True de")
    w("  data/tokens_v2/pairs_annotated.parquet (cf_pairs.effective_pair_ids).")
    w("  post_split = t1 > split_t se mantiene tal cual en el contrato")
    w("  (contrato_tokens_v1.md); post_action es una anotación ADICIONAL.")
    w()

    w("7. Resumen de nulos")
    w("-" * 72)
    w("  clip_bound     : la petición queda anulada por el clip del simulador")
    w("                   (PEEP con base 0 y delta negativo -> clip [0,25]).")
    w("  below_resolution: la verdad difiere pero la variable observada no lo")
    w("                   resuelve (tramo de ~1.5 s de las tasas sobrescrito por")
    w("                   el mantenimiento base y no muestreado; rr_delta por")
    w("                   debajo del escalón entero del setpoint; peep_delta por")
    w("                   debajo del escalón del setpoint).")
    w("  no_change_requested: el valor pedido coincide con el base (<= 5 %).")
    w("  case_ends      : no hay ninguna ventana con t1 > t_action.")
    w("  unexplained    : ninguna.")
    w()
    w(f"  tiempo de anotación: {r['elapsed_s']} s")
    w()
    w("  Efectivos por palanca y lag (acciones puntuales vs consigna persistente)")
    w("  " + "-" * 68)
    lag = eff.effect_lag_windows
    for cls, sel in (("acciones puntuales",
                      eff[~eff.lever.isin(cp.PERSISTENT_LEVERS)]),
                     ("consigna persistente",
                      eff[eff.lever.isin(cp.PERSISTENT_LEVERS)])):
        lg = sel.effect_lag_windows
        w(f"  {cls:<22s} n={len(sel):>5d}  lag p50={np.nanpercentile(lg, 50):>6.1f} "
          f"p90={np.nanpercentile(lg, 90):>6.1f} max={np.nanmax(lg):>6.0f} "
          f"lag<=1={100.0 * (lg <= 1).mean():>7.4f} %")
    w()
    w("  Nulos por grupo de palanca")
    w("  " + "-" * 68)
    nulos = df[~df.lever_effective.astype(bool)]
    w(f"  {'lever_group':<16s} {'pares':>6s} {'nulos':>6s}  causas")
    for g, sub in nulos.groupby("lever_group"):
        c = sub.null_cause.value_counts().to_dict()
        w(f"  {g:<16s} {len(df[df.lever_group == g]):>6d} {len(sub):>6d}  {c}")
    w()
    w("  Uso de post_action en el pipeline de entrenamiento")
    w("  " + "-" * 68)
    w("    from tokens.cf_pairs import load_pairs_annotated, effective_pair_ids")
    w(f"    n_efectivos = {len(eff)}")
    w()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nescrito: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
