"""PASO 3d — Fase 0: genera el informe a partir de ``manifests/paso3d_f0.json``.

Escribe ``reports/REPORT_paso3d_cf_v7_1.txt`` con el estado de cada puerta, la
fe de erratas sobre el diagnóstico del paso 3c, las tablas de G0b/G0c y la
conclusión (PARA si G0a/G0b/G0c falla).

Uso:
  python scripts/paso3d_f0_report.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import paths  # noqa: E402

IN_JSON = paths.MANIFESTS_DIR / "paso3d_f0.json"
IN_JSON_PRIME = paths.MANIFESTS_DIR / "paso3d_f0prime.json"
OUT_TXT = paths.REPORTS_DIR / "REPORT_paso3d_cf_v7_1.txt"
BASE_COMMIT = "f29962c"  # "Paso 3d / Fase 0: diagnostico G0a/G0b/G0c"


def _git_diff(path: str) -> str:
    p = subprocess.run(["git", "diff", BASE_COMMIT, "--", path],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return (p.stdout or "").rstrip()


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
        extra = [f for f in d.get("failures", [])
                 if not f.get("only_known_bt_gap", False)]
        lines.append("")
        lines.append(f"  Pares NO idénticos: {d['n_tested'] - d['n_identical']}. De ellos, "
                     f"{d.get('n_diff_only_bt_gap', 0)} difieren SOLO en Solar8000/BT.")
        if extra:
            lines.append(f"  Los otros {len(extra)} divergen más allá de BT, p. ej. el par "
                         f"{extra[0]['pair_id']} ({extra[0]['lever']}):")
            if isinstance(extra[0].get("diffs"), dict):
                for k, v in extra[0]["diffs"].items():
                    lines.append(f"    {k}: {len(v)} col. (p. ej. {v[:5]})")
    if d.get("failures"):
        f0 = d["failures"][0]
        detail = f0.get("diffs", f0.get("stage", f0))
        lines.append("")
        lines.append(f"  Ejemplo (par {f0.get('pair_id')}, {f0.get('lever')}): "
                     f"{json.dumps(detail, ensure_ascii=False)}")


def _fmt_g0d(d: dict, lines: list[str]) -> None:
    lines.append(f"  rc base={d.get('rc_base')}  rc vaso={d.get('rc_vaso')}")
    for name, v in d.get("per_cohort", {}).items():
        lines.append(f"  {name:<16s} n={v['n_cases']:>3d}  idénticos={v['n_identical']:>3d}  "
                     f"no_regenerados={v.get('n_missing', 0):>3d}  "
                     f"{'PASA' if v['pass'] else 'FALLA'}")
    if d.get("diffs"):
        lines.append("")
        lines.append(f"  Primeras diferencias ({len(d['diffs'])} en total):")
        for rec in d["diffs"][:8]:
            if "diff_columns" in rec:
                ps = ""
                if "presence_same" in rec:
                    ps = f" presence_same={rec['presence_same']}"
                lines.append(f"    {rec['cohort']} caso {rec['caseid']} {rec['artifact']}: "
                             f"{rec['n_diff']} col. (p. ej. {rec['diff_columns'][:5]}){ps}")
            else:
                lines.append(f"    {rec['cohort']} caso {rec['caseid']} {rec['artifact']}: "
                             f"{rec.get('reason')}")
    lines.append("")
    lines.append("  NOTA: en synthetic_v7 los 20 casos difieren en el CONJUNTO DE COLUMNAS")
    lines.append("  (presence_same=False) con el TRUTH idéntico. Causa: la presencia de tracks")
    lines.append("  depende de n_total = len(caseids) (TrackPresenceSampler.sample) y de un estado")
    lines.append("  secuencial, así que una regeneración PARCIAL (20 de 10000) no puede")
    lines.append("  reproducirla. NO es BT. vaso_reinf_v7 (10/10) SÍ reproduce.")


def _fase0(lines: list[str], data: dict) -> None:
    lines.append("=" * 78)
    lines.append("PARTE 1 — FASE 0 (commit " + BASE_COMMIT + ")")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Objetivo: corregir el muestreo de las palancas de consigna (set_rr, set_tv,")
    lines.append("set_peep) para que δ sea múltiplo entero no nulo del escalón de registro, y")
    lines.append("regenerar aguas abajo (cf_v7_1 -> windows_v4_1/pk_v2_1/tokens_v2_1).")
    lines.append("La Fase 0 comprueba tres puertas bloqueantes antes de invertir en regenerar.")
    lines.append("")
    lines.append("-" * 78)
    lines.append("G0a — reproducibilidad de cf_v7 desde el código del repo")
    lines.append("-" * 78)
    if "G0a" in data:
        _fmt_g0a(data["G0a"], lines)
        lines.append("")
        lines.append("  NOTA (Fase 0′): G0a se ejecutó sobre 21 pares (1 por palanca), no 210, y el")
        lines.append("  'artefacto nulo de truth/phase' era un defecto del COMPARADOR (corregido en")
        lines.append("  la Fase 0′). Véase la Parte 2.")
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
    lines.append("  VEREDICTO FASE 0: PARA (G0a falló por el hueco de BT §6).")
    lines.append("")


def _fase0prime(lines: list[str], data: dict) -> None:
    lines.append("=" * 78)
    lines.append("PARTE 2 — FASE 0′ (Enmienda 1: el repo debe reproducir los datos v7)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("1. CORRECCIONES A LA FASE 0")
    lines.append("   1.1 Tamaño de G0a: la Fase 0 ejecutó G0a sobre 21 pares (1 por palanca), NO")
    lines.append("       210. Este informe declara siempre el tamaño real (el manifiesto f0.json")
    lines.append("       lo registra en n_tested). La entrada de provenance_gaps del commit "
                 + BASE_COMMIT)
    lines.append("       afirmaba '210 pares … 0/210 idénticos': era FALSA y queda corregida.")
    lines.append("   1.2 'truth/phase' era un ARTEFACTO DEL COMPARADOR, no una diferencia:")
    lines.append("       _diff_columns comparaba columnas object con np.array_equal, donde")
    lines.append("       nan != nan. Corregido a pd.Series.equals / equal_nan (nulo == nulo en la")
    lines.append("       misma posición). Test: tests/test_paso3d_comparador.py.")
    lines.append("   1.3 Se RECHAZA la opción (a): corregir BT sólo en cf crearía una diferencia de")
    lines.append("       cohorte en una de las 14 variables de la imagen del AE (ae_v2 se entrenó")
    lines.append("       con el BT normal(36.0,0.9) de toda la v7; y en ~1/20 casos cambiaría toda")
    lines.append("       la capa de observación). La corrección de BT pertenece a una regeneración")
    lines.append("       GLOBAL + reentrenamiento del AE. Decisión v8 §6 precisada: 'en la próxima")
    lines.append("       regeneración GLOBAL'.")
    lines.append("   1.4 Se adopta la opción (b): el repo debe contener el código que generó los")
    lines.append("       datos vigentes.")
    lines.append("")
    lines.append("2. CAMBIO DE CÓDIGO (única excepción a 'no tocar simulate.py')")
    lines.append("   Se añade el campo de configuración ``bt_start_model``:")
    lines.append("     'v7_normal'          = clip(normal(36.0, 0.9), 33.5, 37.3) — la línea que")
    lines.append("                            generó v7 en disco (el CLIP se recuperó empíricamente);")
    lines.append("     'uniform_c3_revert'  = uniform(36.5, 37.4) — corrección C3 pendiente de la")
    lines.append("                            regeneración GLOBAL.")
    lines.append("   Por defecto 'v7_normal'; synthetic_v7.yaml lo fija a 'v7_normal'. El sorteo se")
    lines.append("   centraliza en ``simulate.sample_bt_start``. Sin otros cambios de comportamiento.")
    lines.append("")
    lines.append("   RECUPERACIÓN DEL CLIP (por qué hacía falta): con sólo ``normal(36.0, 0.9)``,")
    lines.append("   G0a′ daba 196/210: los 14 pares restantes diferían SÓLO en Solar8000/BT, con un")
    lines.append("   desplazamiento CASI CONSTANTE (0.1 a 1.1 °C) que no arrastraba ninguna otra")
    lines.append("   columna. Eso descarta un desplazamiento del flujo de ``rng_sensor`` (que")
    lines.append("   afectaría a toda la capa de observación) y apunta a una transformación distinta")
    lines.append("   del MISMO sorteo. Instrumentando ``sample_bt_start`` se obtuvo el valor sorteado")
    lines.append("   S y el implicado por los datos (S + offset): los pares con S > 37.3 daban")
    lines.append("   S_implicado ≈ 37.30 y el par con S < 33.5 daba S_implicado ≈ 33.50, es decir, un")
    lines.append("   CLIP en [33.5, 37.3] que no constaba en el código. Con el clip, los 210 pares son")
    lines.append("   idénticos. Un clip no consume sorteos, así que el flujo queda alineado y sólo")
    lines.append("   cambia BT: exactamente lo observado.")
    lines.append("")
    for path in ("src/anessim/simulate.py", "src/anessim/config.py",
                 "src/anessim/configs/synthetic_v7.yaml"):
        lines.append("   DIFF COMPLETO de " + path + " (vs " + BASE_COMMIT + "):")
        diff = _git_diff(path)
        if diff:
            for ln in diff.splitlines():
                lines.append("   " + ln)
        else:
            lines.append("   (sin diferencias)")
        lines.append("")
    lines.append("   sha256 del núcleo (Fase 0′; referencia para test_nucleo_generador_intacto):")
    for f, s in data.get("core_shas", {}).items():
        lines.append(f"     {f:<45s} {s}")
    lines.append("")
    lines.append("3. G0a′ — reproducibilidad de cf_v7 (10 pares/palanca + 197501)")
    lines.append("-" * 78)
    lines.append("   Criterio: 210/210 idénticos (10 pares x 21 palancas) + el par 197501, sin")
    lines.append("   excluir ninguna columna.")
    if "G0a" in data:
        _fmt_g0a(data["G0a"], lines)
        g = data["G0a"]
        lines.append("")
        lines.append(f"   G0a′: {'PASA' if g['pass'] else 'FALLA'}")
        if g["pass"]:
            lines.append("   El par 197501 sale idéntico: CONFIRMA la hipótesis del ziggurat")
            lines.append("   (normal() consume un número variable de sorteos; uniform() siempre 1).")
    else:
        lines.append("   (no ejecutada)")
    lines.append("")
    lines.append("4. G0d — reproducibilidad de synthetic_v7 y vaso_reinf_v7 (INFORMATIVA)")
    lines.append("-" * 78)
    if "G0d" in data:
        _fmt_g0d(data["G0d"], lines)
    else:
        lines.append("   (no ejecutada)")
    lines.append("")
    lines.append("-" * 78)
    lines.append("VEREDICTO DE LA FASE 0′")
    lines.append("-" * 78)
    if bool(data.get("pass")):
        lines.append("  PASA: con bt_start_model='v7_normal' el código del repo reproduce cf_v7.")
        lines.append("  Se procede a la Fase 1.")
    else:
        lines.append("  PARA: G0a′ (o alguna puerta de la Fase 0) FALLA. Véase arriba.")
    lines.append("")


def main() -> int:
    data0 = json.loads(IN_JSON.read_text(encoding="utf-8"))
    data1 = (json.loads(IN_JSON_PRIME.read_text(encoding="utf-8"))
             if IN_JSON_PRIME.exists() else None)
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("PASO 3d — cf_v7_1: regeneración de los contrafactuales")
    lines.append("INFORME DE LAS FASES 0 y 0′")
    lines.append(f"Fecha Fase 0: {data0.get('date', '?')}"
                 + (f"  |  Fase 0′: {data1.get('date')}" if data1 else ""))
    lines.append("=" * 78)
    lines.append("")
    _fase0(lines, data0)
    if data1 is not None:
        _fase0prime(lines, data1)
    else:
        lines.append("(FASE 0′: manifiesto paso3d_f0prime.json no encontrado)")
        lines.append("")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"escrito {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
