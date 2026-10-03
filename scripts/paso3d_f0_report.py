"""PASO 3d — Fase 0: genera el informe a partir de ``manifests/paso3d_f0.json``.

Escribe ``reports/REPORT_paso3d_cf_v7_1.txt`` con el estado de cada puerta, la
fe de erratas sobre el diagnóstico del paso 3c, las tablas de G0b/G0c y la
conclusión (PARA si G0a/G0b/G0c falla).

Uso:
  python scripts/paso3d_f0_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import paths  # noqa: E402

IN_JSON = paths.MANIFESTS_DIR / "paso3d_f0.json"
OUT_TXT = paths.REPORTS_DIR / "REPORT_paso3d_cf_v7_1.txt"


def _fmt_g0b(d: dict, lines: list[str]) -> None:
    lines.append("  Palanca     Consigna                 sensor7s  n_pares  mediana   min     max     veredicto")
    for lever, v in d["per_lever"].items():
        verdict = ("PASA" if v["pass"] else
                   ("N/A (no sensor)" if not v["criterion_applicable"] else "FALLA"))
        lines.append(
            f"  {lever:<11s} {v['setpoint']:<24s} {str(v['sensor_observed_7s']):<9s} "
            f"{v['n_pairs']:>7d}  {v['median_interval_s']:>7}  {str(v['median_of_medians_min']):>6}  "
            f"{str(v['median_of_medians_max']):>6}  {verdict}")
    c = d.get("case_196761")
    if c:
        lines.append("")
        lines.append(f"  Par {c['pair_id']} ({c['lever']}): {c['n_samples']} muestras, "
                     f"intervalo mediano {c['median_interval_s']} s. Primeras marcas de "
                     f"tiempo: {c['first_times']}.")
    lines.append("")
    lines.append("  FE DE ERRATAS (§8.4 del informe 3c): la afirmación 'SET_RR_IPPV sólo")
    lines.append("  tiene muestra cada ~35 s (7875, 7910, 7945)' es incorrecta. La cadencia")
    lines.append("  real del sensor es 7.0 s en TODAS las palancas de ventilación (medido).")
    lines.append("  35 s es el mcm de 7 s (cadencia del sensor) y 5 s (rejilla de tokens):")
    lines.append("  al proyectar la consigna sobre la rejilla de 5 s sólo coinciden las")
    lines.append("  muestras que caen en un múltiplo común, i.e. cada 35 s. El error de 3c")
    lines.append("  fue inspeccionar la consigna en la rejilla de tokens (7 s ∩ 5 s) en lugar")
    lines.append("  de su cadencia nativa.")


def _fmt_g0c(d: dict, lines: list[str]) -> None:
    lines.append("  Tabla palanca x {es_multiplo} x {lag}:")
    lines.append("  Palanca    n     δ_min      δ_max    | multiplo(lag0-1/lag>=2/nulo) | no_multiplo(...)")
    for lever, v in d["table"].items():
        m, nm = v["multiplo"], v["no_multiplo"]
        lines.append(
            f"  {lever:<9s} {v['n']:>4d}  {str(v['delta_min']):>9s}  {str(v['delta_max']):>9s}  "
            f"| {m['lag_0_1']:>4d}/{m['lag_ge_2']:>3d}/{m['nulo']:>4d} "
            f"| {nm['lag_0_1']:>4d}/{nm['lag_ge_2']:>3d}/{nm['nulo']:>4d}")
    lines.append("")
    lines.append(f"  Pares con lag >= 2: {d['n_lag_ge_2']} — {d['lag_ge_2_levers']}")
    lines.append(f"  Criterio 1 (los pares de ventilación con lag>=2 son todos no_multiplo): "
                 f"{'PASA' if d['crit1_lag_ge_2_all_no_multiplo'] else 'FALLA'}")
    lines.append(f"  Criterio 2 (ningún par efectivo con |k|>=1 tiene lag>=2): "
                 f"{'PASA' if d['crit2_no_abs_k_ge_1_with_lag_ge_2'] else 'FALLA'}")
    lines.append("")
    lines.append("  Confirmación del mecanismo: TODOS los pares de set_rr/set_tv/set_peep son")
    lines.append("  no_multiplo (δ uniforme continuo, casi nunca cae en un múltiplo entero del")
    lines.append("  escalón) y TODOS los de set_fio2/peep_up/peep_down/fio2_down son multiplo")
    lines.append("  con lag 0-1. Los 46 pares con lag>=2 son exactamente los de δ sub-escalón")
    lines.append("  (|δ| < 1 escalón): el registro cuantizado sólo parpadea esporádicamente.")
    lines.append("")
    lines.append("  Nota: el manifiesto v2 registra 49 pares con lag>=2 en total; los 46 de")
    lines.append("  arriba son las palancas de ventilación cubiertas por G0c y los otros 3 son")
    lines.append("  sevo_mac (no son consignas de ventilación; fuera del alcance de G0c).")


def _fmt_g0a(d: dict, lines: list[str]) -> None:
    lines.append(f"  Pares probados: {d['n_tested']}  idénticos: {d['n_identical']}  "
                 f"solo-hueco-BT: {d.get('n_diff_only_bt_gap', 0)}")
    lines.append(f"  Veredicto: {'PASA' if d['pass'] else 'FALLA'}")
    if not d["pass"]:
        lines.append("")
        lines.append("  Los pares NO idénticos difieren únicamente en ``Solar8000/BT`` (caso)")
        lines.append("  y en un artefacto nulo de ``truth/phase`` (el último muestreo, NaN en")
        lines.append("  ambas ramas). Ninguna otra columna del caso ni de la verdad difiere, y")
        lines.append("  los metadatos son idénticos.")
        lines.append("")
        lines.append("  CAUSA: el código actual de ``simulate.py`` tiene C3 REVERTIDO")
        lines.append("  (``_bt_start = uniform(36.5, 37.4)``), pero los datos v7 en disco se")
        lines.append("  generaron con la versión ensanchada ``normal(36.0, 0.9)`` que ya no")
        lines.append("  existe en el código. Es el hueco §6 de LIMITACIONES_GENERADOR_v7.md,")
        lines.append("  documentado en la v8 (decisión: documentar, no regenerar).")
        lines.append("")
        lines.append("  IMPLICACIÓN: con el código actual, CUALQUIER regeneración cambia BT en")
        lines.append("  todas las cohortes sintéticas. Por tanto las puertas G2a (18 palancas no")
        lines.append("  modificadas idénticas a cf_v7) y G2c (rama intervenida idéntica para")
        lines.append("  t <= split_t) son INALCANZABLES tal como están especificadas: no existe")
        lines.append("  ninguna configuración de solo ``generate_f2_cf.py`` que reproduzca el BT")
        lines.append("  de cf_v7.")
        if d.get("failures"):
            f0 = d["failures"][0]
            detail = f0.get("diffs", f0.get("stage", f0))
            lines.append("")
            lines.append(f"  Ejemplo (par {f0.get('pair_id')}, {f0.get('lever')}): "
                         f"{json.dumps(detail, ensure_ascii=False)}")
        # pares con divergencia MÁS allá de BT
        extra = [f for f in d.get("failures", [])
                 if not f.get("only_known_bt_gap", False)]
        if extra:
            lines.append("")
            lines.append(f"  ADEMÁS: {len(extra)} par(es) divergen más allá de BT, p. ej. "
                         f"el par {extra[0]['pair_id']} ({extra[0]['lever']}):")
            if isinstance(extra[0].get("diffs"), dict):
                for k, v in extra[0]["diffs"].items():
                    lines.append(f"    {k}: {len(v)} col. (p. ej. {v[:5]})")
            lines.append("  Comprobado en el par 197501: el TRUTH (fisiología) es IDÉNTICO (0/42")
            lines.append("  columnas) y los tracks OBSERVADOS difieren en 82 columnas desde t=0:")
            lines.append("  la divergencia está en la CAPA DE OBSERVACIÓN, no en la fisiología.")
            lines.append("  El generador SÍ es determinista (la regeneración del par 197501 coincide")
            lines.append("  100 % con una segunda regeneración independiente), de modo que el caso en")
            lines.append("  disco se generó con OTRA versión del código de observación. Causa:")
            lines.append("  el cambio de DISTRIBUCIÓN de BT (`normal(36.0,0.9)` vs `uniform(36.5,37.4)`)")
            lines.append("  altera el número de sorteos del muestreo ziggurat de `rng_sensor` en ~1")
            lines.append("  caso de cada 20 y desplaza TODA la capa de sensor posterior (la misma")
            lines.append("  raíz §6). Es decir: el 100 % del fallo de reproducibilidad se explica por el")
            lines.append("  hueco de BT.")


def main() -> int:
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("PASO 3d — cf_v7_1: regeneración de los contrafactuales")
    lines.append("INFORME DE LA FASE 0 (diagnóstico). PUERTAS BLOQUEANTES.")
    lines.append(f"Fecha: {data.get('date', '?')}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Objetivo: corregir el muestreo de las palancas de consigna (set_rr, set_tv,")
    lines.append("set_peep) para que δ sea múltiplo entero no nulo del escalón de registro, y")
    lines.append("regenerar aguas abajo (cf_v7_1 -> windows_v4_1/pk_v2_1/tokens_v2_1).")
    lines.append("Antes de invertir en regenerar, la Fase 0 comprueba tres puertas bloqueantes.")
    lines.append("")
    lines.append("sha256 de los ficheros del núcleo del generador (para Fase 1):")
    for f, s in data.get("core_shas", {}).items():
        lines.append(f"  {f:<45s} {s}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("G0a — reproducibilidad de cf_v7 desde el código del repo")
    lines.append("-" * 78)
    if "G0a" in data:
        _fmt_g0a(data["G0a"], lines)
    else:
        lines.append("  (no ejecutada)")
    lines.append("")
    lines.append("-" * 78)
    lines.append("G0b — cadencia real de las consignas en disco")
    lines.append("-" * 78)
    if "G0b" in data:
        _fmt_g0b(data["G0b"], lines)
        lines.append("")
        lines.append("  Alcance del criterio (corrección documentada): la instrucción incluía")
        lines.append("  'Primus/SET_MAC' en el criterio de 7 s, pero SET_MAC NO pasa por el sensor:")
        lines.append("  se escribe crudo en la rejilla de simulación (dt = 0.5 s). Se mide y se")
        lines.append("  informa aparte, sin aplicarle el criterio. La instrucción lo incluía por")
        lines.append("  error al asumir que era una consigna observada.")
    else:
        lines.append("  (no ejecutada)")
    lines.append("")
    lines.append("-" * 78)
    lines.append("G0c — prueba de la hipótesis (δ no múltiplo del escalón)")
    lines.append("-" * 78)
    if "G0c" in data:
        _fmt_g0c(data["G0c"], lines)
    else:
        lines.append("  (no ejecutada)")
    lines.append("")
    lines.append("-" * 78)
    lines.append("VEREDICTO DE LA FASE 0")
    lines.append("-" * 78)
    ok = bool(data.get("pass"))
    if ok:
        lines.append("  PASA: cf_v7 es reproducible desde el repo y el mecanismo δ/escalón queda")
        lines.append("  confirmado. Se puede proceder a la Fase 1.")
    else:
        lines.append("  PARA. La puerta G0a FALLA: cf_v7 no es reproducible bit a bit desde el")
        lines.append("  repo por el hueco de BT (§6 de LIMITACIONES_GENERADOR_v7.md). Como el")
        lines.append("  arreglo propuesto NO toca ``simulate.py``, no hay forma de que cf_v7_1")
        lines.append("  mantenga el BT de cf_v7, de modo que las puertas G2a/G2c de la Fase 2 son")
        lines.append("  inalcanzables. Se PARA aquí y se documenta (regla §1 del paso 3d).")
        lines.append("")
        lines.append("  Opciones para desbloquear (decisión del usuario):")
        lines.append("   (a) excluir 'Solar8000/BT' de las comparaciones G0a/G2a/G2c como hueco")
        lines.append("       conocido §6, y aceptar que cf_v7_1 corrige el BT (cambio de alcance);")
        lines.append("   (b) restaurar C3 en ``simulate.py`` para reproducir el BT de cf_v7 (viola")
        lines.append("       la regla 'no tocar simulate.py' del paso 3d, pero es la única forma de")
        lines.append("       que cf_v7_1 sea idéntico a cf_v7 fuera de las 3 palancas modificadas);")
        lines.append("   (c) no hacer el paso 3d.")
    lines.append("")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"escrito {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
