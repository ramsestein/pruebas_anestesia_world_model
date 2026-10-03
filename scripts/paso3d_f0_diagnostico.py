"""PASO 3d — Fase 0: diagnóstico sobre disco (puertas G0a/G0b/G0c).

SIN cambios de código ni de datos. Sólo lee y regenera en un directorio temporal
FUERA de ``data/``.

Puertas (todas bloqueantes):

  G0a  reproducibilidad de cf_v7 desde el código del repo: regenera los 10
       primeros pares de cada una de las 21 palancas (210 pares) con los mismos
       seed/caseid/collection/config y compara caso (ambas ramas), truth y
       metadata. Criterio: 210/210 idénticos (mismas columnas, dtypes y valores,
       NaN == NaN).

  G0b  cadencia real de las consignas en disco: mediana del intervalo entre
       muestras no-NaN de cada ``Primus/SET_*`` en mantenimiento. Criterio:
       mediana = 7.0 s ± 0.5 s en todas las palancas.

  G0c  prueba de la hipótesis (δ no múltiplo del escalón de registro) antes de
       invertir en regenerar. Criterios bloqueantes:
         1. los pares de ventilación con lag >= 2 son todos ``no_multiplo``;
         2. entre los pares efectivos con |k| >= 1, ninguno tiene lag >= 2.

Escribe ``manifests/paso3d_f0.json``.

Uso:
  python scripts/paso3d_f0_diagnostico.py            # G0a+G0b+G0c
  python scripts/paso3d_f0_diagnostico.py --only g0b # sólo una puerta
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time as _time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import paths  # noqa: E402
from anessim.config import SimulatorConfig  # noqa: E402
from anessim.scripts import generate_f2_cf as gcf  # noqa: E402

OUT_JSON = paths.MANIFESTS_DIR / "paso3d_f0.json"
CF_DIR = paths.COHORTS["cf_v7"]
CONFIG_PATH = ROOT / "src" / "anessim" / "configs" / "synthetic_v7.yaml"

CORE_FILES = [
    "src/anessim/simulate.py",
    "src/anessim/sensors.py",
    "src/anessim/respiratory.py",
    "src/anessim/render.py",
    "src/anessim/scripts/generate_f2_cf.py",
    "src/anessim/configs/synthetic_v7.yaml",
]

PHARMA = frozenset(gcf.PHARMA_LEVERS)
VENT = frozenset(gcf.VENT_LEVERS)

# Escalón de registro (QUANT_GRID) de la consigna observada de cada override.
# δ se expresa EN UNIDADES DE ESCALÓN antes de calcular k.
LEVER_DELTA_UNIT: dict[str, tuple[str, float]] = {
    "set_rr": ("rr_delta", 1.0),      # rpm
    "set_tv": ("tv_delta", 10.0),     # mL -> unidades de 0.01 L
    "set_peep": ("peep_delta", 1.0),  # cmH2O
    "set_fio2": ("fio2_set", 1.0),    # % (valor absoluto, ver _delta_reg)
    "peep_up": ("peep_delta", 1.0),
    "peep_down": ("peep_delta", 1.0),
    "fio2_down": ("fio2_delta", 1.0),  # fracción -> %
}

VENT_OVERRIDE_LEVERS = sorted(LEVER_DELTA_UNIT)

# Levers cuya consigna registrada SÍ pasa por el modelo de observación con
# cadencia de 7 s (``sensor.observe(..., 7.0, ...)`` en ``_build_tracks``). El
# criterio de G0b (mediana = 7.0 s ± 0.5 s) se aplica SOLO a estas.
SENSOR_OBSERVED_LEVERS = [
    "set_rr", "set_tv", "set_peep", "set_fio2",
    "peep_up", "peep_down", "fio2_down",
]
# ``Primus/SET_MAC`` NO se observa con sensor: se escribe crudo en la rejilla de
# simulación (dt = 0.5 s). Se mide y se informa aparte, no se le aplica el
# criterio de 7 s.
RAW_SETPOINT_LEVERS = ["sevo_mac", "sevo_up"]

# Consigna observada por palanca (SET_* que registra el override).
LEVER_SETPOINT: dict[str, str] = {
    "set_rr": "Primus/SET_RR_IPPV",
    "set_tv": "Primus/SET_TV_L",
    "set_peep": "Primus/SET_INTER_PEEP",
    "set_fio2": "Primus/SET_FIO2",
    "peep_up": "Primus/SET_INTER_PEEP",
    "peep_down": "Primus/SET_INTER_PEEP",
    "fio2_down": "Primus/SET_FIO2",
    "sevo_mac": "Primus/SET_MAC",
    "sevo_up": "Primus/SET_MAC",
}


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collection_of(lever: str) -> str:
    if lever in PHARMA:
        return "pharma"
    if lever in VENT:
        return "vent"
    return "learning"


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    """Igualdad estricta: columnas, dtypes y valores (NaN == NaN)."""
    if list(a.columns) != list(b.columns):
        return False, f"columnas difieren ({len(a.columns)} vs {len(b.columns)})"
    for c in a.columns:
        da, db = a[c].dtype, b[c].dtype
        if str(da) != str(db):
            return False, f"dtype de {c}: {da} vs {db}"
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=True, check_exact=True)
    except AssertionError as exc:  # noqa: PERF203
        return False, str(exc).splitlines()[0][:200]
    return True, ""


def _diff_columns(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """Lista de columnas cuyos valores difieren (NaN == NaN)."""
    if list(a.columns) != list(b.columns):
        return ["<columnas>"]

    def _eq(va: np.ndarray, vb: np.ndarray) -> bool:
        if va.dtype.kind in "fc" and vb.dtype.kind in "fc":
            return bool(np.array_equal(va, vb, equal_nan=True))
        try:
            return bool(np.array_equal(va, vb))
        except TypeError:
            return bool(np.array_equal(va.astype(str), vb.astype(str)))

    return [c for c in a.columns if not _eq(a[c].to_numpy(), b[c].to_numpy())]


def _read_parquet(p: Path) -> pd.DataFrame:
    return pq.read_table(p).to_pandas()


# ---------------------------------------------------------------------------
# G0a — reproducibilidad
# ---------------------------------------------------------------------------

def _g0a_task(task: dict) -> dict:
    return gcf._worker(task)


def g0a(pairs: pd.DataFrame, config_dict: dict, workers: int,
        n_per_lever: int = 10, tmp_dir: str | None = None,
        compare_only: bool = False) -> dict:
    t0 = _time.time()
    tasks = []
    for lever in sorted(pairs.lever.unique()):
        sub = pairs[pairs.lever == lever].sort_values("pair_id").head(n_per_lever)
        for r in sub.itertuples():
            pid = int(r.pair_id)
            meta = json.loads((CF_DIR / "metadata" / f"cf_pair_{pid}.json").read_text())
            tasks.append((pid, lever, int(meta["seed"])))

    if tmp_dir is not None:
        tmp_root = Path(tmp_dir)
        resume = True
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="paso3d_g0a_", dir=str(ROOT.parent)))
        resume = False
    for sub in ("cases", "truth", "metadata", "clinical", "clinical_notes"):
        (tmp_root / sub).mkdir(parents=True, exist_ok=True)

    results: dict[int, str] = {}
    if compare_only:
        results = {pid: "ok" for pid, _, _ in tasks}
    else:
        work = [{
            "out_dir": str(tmp_root),
            "caseid": pid,
            "lever": lever,
            "seed": seed,
            "collection": collection_of(lever),
            "config_dict": config_dict,
            "force": not resume,
        } for pid, lever, seed in tasks]

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_g0a_task, w): w["caseid"] for w in work}
            done = 0
            for fut in as_completed(futs):
                pid = futs[fut]
                try:
                    r = fut.result()
                    results[pid] = "error: " + r["error"] if "error" in r else "ok"
                except Exception as exc:  # noqa: BLE001
                    results[pid] = f"exc: {exc!r}"
                done += 1
                if done % 10 == 0 or done == len(work):
                    print(f"    g0a {done}/{len(work)} ({_time.time()-t0:.0f}s)",
                          flush=True)

    # comparación
    n_ident = 0
    n_diff_only_bt = 0
    failures: list[dict] = []
    per_lever = defaultdict(lambda: {"n": 0, "ident": 0, "diff_only_bt": 0})
    for pid, lever, seed in tasks:
        cid_a, cid_b = pid, pid + 1
        per_lever[lever]["n"] += 1
        if results.get(pid) != "ok":
            failures.append({"pair_id": pid, "lever": lever, "stage": results.get(pid)})
            continue
        artifact_diffs: dict[str, list[str]] = {}
        for kind, rel in (
            ("case_a", f"cases/{cid_a}.parquet"),
            ("case_b", f"cases/{cid_b}.parquet"),
            ("truth_a", f"truth/{cid_a}_truth.parquet"),
            ("truth_b", f"truth/{cid_b}_truth.parquet"),
        ):
            a = _read_parquet(CF_DIR / rel)
            b = _read_parquet(tmp_root / rel)
            d = _diff_columns(a, b)
            if d:
                artifact_diffs[kind] = d
        try:
            ja = json.loads((CF_DIR / "metadata" / f"cf_pair_{pid}.json").read_text())
            jb = json.loads((tmp_root / "metadata" / f"cf_pair_{pid}.json").read_text())
            if ja != jb:
                artifact_diffs["metadata"] = ["<json>"]
        except FileNotFoundError as exc:
            artifact_diffs["metadata"] = [str(exc)]
        if not artifact_diffs:
            n_ident += 1
            per_lever[lever]["ident"] += 1
            continue
        # ¿la única diferencia sustantiva es Solar8000/BT (y phase nulo)?
        only_known = all(
            set(cols) <= ({"Solar8000/BT"} if k.startswith("case") else {"phase"})
            for k, cols in artifact_diffs.items())
        if only_known:
            n_diff_only_bt += 1
            per_lever[lever]["diff_only_bt"] += 1
        failures.append({"pair_id": pid, "lever": lever,
                         "diffs": {k: v for k, v in artifact_diffs.items()},
                         "only_known_bt_gap": bool(only_known)})

    ok_all = n_ident == len(tasks)
    out = {
        "n_tested": len(tasks),
        "n_identical": n_ident,
        "n_diff_only_bt_gap": n_diff_only_bt,
        "pass": bool(ok_all),
        "per_lever": {k: dict(v) for k, v in sorted(per_lever.items())},
        "failures": failures[:60],
        "tmp_dir": str(tmp_root),
        "elapsed_s": round(_time.time() - t0, 1),
    }
    print(f"  G0a: {n_ident}/{len(tasks)} idénticos "
          f"(solo-BT: {n_diff_only_bt}) ({'PASA' if ok_all else 'FALLA'}) "
          f"[{out['elapsed_s']}s]")
    if failures:
        print(f"  G0a fallos (primeros 3): {failures[:3]}")
    return out


# ---------------------------------------------------------------------------
# G0b — cadencia real de las consignas
# ---------------------------------------------------------------------------

def _maintenance_bounds(truth: pd.DataFrame) -> tuple[float, float]:
    if "phase" not in truth.columns:
        return float(truth["time"].min()), float(truth["time"].max())
    m = truth[truth["phase"] == "maintenance"]
    if not len(m):
        return float(truth["time"].min()), float(truth["time"].max())
    return float(m["time"].min()), float(m["time"].max())


def _median_interval(times: np.ndarray) -> float:
    if len(times) < 3:
        return float("nan")
    return float(np.median(np.diff(times)))


def g0b(pairs: pd.DataFrame, target_pair: int = 196761) -> dict:
    t0 = _time.time()
    per_lever: dict[str, dict] = {}
    for lever in SENSOR_OBSERVED_LEVERS + RAW_SETPOINT_LEVERS:
        col = LEVER_SETPOINT[lever]
        sub = pairs[pairs.lever == lever]
        medians: list[float] = []
        n_samples: list[int] = []
        for r in sub.itertuples():
            for cid in (int(r.caseid_intervencion), int(r.caseid_base)):
                p = CF_DIR / "cases" / f"{cid}.parquet"
                if not p.exists():
                    continue
                df = pq.read_table(p, columns=["time", col]).to_pandas()
                tr = pq.read_table(
                    CF_DIR / "truth" / f"{cid}_truth.parquet",
                    columns=["time", "phase"]).to_pandas()
                lo, hi = _maintenance_bounds(tr)
                t = df["time"].to_numpy(np.float64)
                v = df[col].to_numpy(np.float64)
                sel = np.isfinite(v) & (t >= lo) & (t <= hi)
                tt = t[sel]
                if len(tt) >= 3:
                    medians.append(_median_interval(tt))
                    n_samples.append(len(tt))
        med = float(np.median(medians)) if medians else float("nan")
        sensor = lever in SENSOR_OBSERVED_LEVERS
        passes = bool(
            sensor and medians and abs(med - 7.0) <= 0.5
            and abs(min(medians) - 7.0) <= 0.5
            and abs(max(medians) - 7.0) <= 0.5)
        per_lever[lever] = {
            "setpoint": col,
            "sensor_observed_7s": sensor,
            "n_pairs": int(len(sub)),
            "median_interval_s": round(med, 4),
            "median_of_medians_min": round(float(np.min(medians)), 4) if medians else None,
            "median_of_medians_max": round(float(np.max(medians)), 4) if medians else None,
            "criterion_applicable": sensor,
            "pass": passes,
        }
    ok_all = all(v["pass"] for v in per_lever.values() if v["criterion_applicable"])


    # caso 196761 citado por 3c
    case_note = None
    if target_pair in set(pairs.pair_id):
        lever = str(pairs.loc[pairs.pair_id == target_pair, "lever"].iloc[0])
        col = LEVER_SETPOINT.get(lever, "Primus/SET_RR_IPPV")
        cid = int(pairs.loc[pairs.pair_id == target_pair, "caseid_intervencion"].iloc[0])
        p = CF_DIR / "cases" / f"{cid}.parquet"
        if p.exists():
            df = pq.read_table(p, columns=["time", col]).to_pandas()
            t = df["time"].to_numpy(np.float64)
            v = df[col].to_numpy(np.float64)
            tt = t[np.isfinite(v)]
            case_note = {
                "pair_id": target_pair, "lever": lever, "setpoint": col, "caseid": cid,
                "n_samples": int(len(tt)),
                "median_interval_s": round(_median_interval(tt), 3),
                "first_times": [round(float(x), 2) for x in tt[:12]],
            }

    out = {
        "per_lever": per_lever,
        "criterion": "mediana = 7.0 s ± 0.5 s por palanca",
        "criterion_scope": (
            "El criterio se aplica a las consignas que pasan por el modelo de "
            "observación con cadencia de 7 s (SET_FIO2, SET_RR_IPPV, SET_TV_L, "
            "SET_INTER_PEEP). 'Primus/SET_MAC' NO se observa con sensor: se "
            "escribe crudo en la rejilla de simulación (dt = 0.5 s), por lo que "
            "se mide y se informa aparte SIN aplicarle el criterio de 7 s. La "
            "instrucción lo incluía por error al asumir que era una consigna "
            "observada; se documenta esta corrección de alcance."
        ),
        "pass": bool(ok_all),
        "case_196761": case_note,
        "explanation_35s": (
            "35 s = mcm(7 s cadencia del sensor, 5 s rejilla de tokens). Las "
            "muestras de SET_* existen cada 7 s en el eje crudo de 0.5 s; al "
            "llevarlas a la rejilla de 5 s (p. ej. al inspeccionar la serie en "
            "la capa de tokens o al quedarse con las muestras que caen en puntos "
            "múltiplos de 5 s) sólo sobreviven las que caen en un múltiplo común, "
            "esto es, cada 35 s. La cadencia real del sensor es 7 s."
        ),
        "elapsed_s": round(_time.time() - t0, 1),
    }
    print(f"  G0b: {'PASA' if ok_all else 'FALLA'} "
          f"({len(per_lever)} palancas) [{out['elapsed_s']}s]")
    return out


# ---------------------------------------------------------------------------
# G0c — prueba de la hipótesis
# ---------------------------------------------------------------------------

def _delta_reg(meta: dict) -> tuple[float, str] | None:
    """δ de la intervención EN UNIDADES DE ESCALÓN de registro."""
    ov = meta.get("vent_override_a") or {}
    lever = str(meta["lever"])
    if lever not in LEVER_DELTA_UNIT:
        return None
    key, step = LEVER_DELTA_UNIT[lever]
    if lever == "set_fio2":
        # override absoluto: δ = (fio2_set - baseline) * 100 / 1 (%)
        phys = meta.get("physiology_override") or {}
        base = float(phys.get("fio2_baseline", 0.50))
        return (float(ov["fio2_set"]) - base) * 100.0 / step, key
    if lever == "fio2_down":
        return float(ov["fio2_delta"]) * 100.0 / step, key
    return float(ov[key]) / step, key


def g0c(pairs: pd.DataFrame, ann2: pd.DataFrame) -> dict:
    t0 = _time.time()
    ann = ann2.set_index("pair_id")
    table: dict[str, dict] = {}
    rows: list[dict] = []
    for lever in VENT_OVERRIDE_LEVERS:
        sub = pairs[pairs.lever == lever]
        for r in sub.itertuples():
            pid = int(r.pair_id)
            if pid not in ann.index:
                continue
            meta = json.loads((CF_DIR / "metadata" / f"cf_pair_{pid}.json").read_text())
            d = _delta_reg(meta)
            if d is None:
                continue
            k, key = d
            es_mult = bool(abs(k - round(k)) < 1e-6 and round(k) != 0)
            lag = ann.at[pid, "effect_lag_boundary"]
            eff = bool(ann.at[pid, "lever_effective_v2"])
            lag = float(lag) if np.isfinite(lag) else float("nan")
            if not eff or not np.isfinite(lag):
                lag_cat = "nulo"
            elif lag <= 1:
                lag_cat = "lag_0_1"
            else:
                lag_cat = "lag_ge_2"
            rows.append({"pair_id": pid, "lever": lever, "delta": float(k),
                         "key": key, "es_multiplo": es_mult, "lag": lag,
                         "lag_cat": lag_cat, "abs_k_ge_1": abs(k) >= 1.0})
    df = pd.DataFrame(rows)
    for lever in VENT_OVERRIDE_LEVERS:
        d = df[df.lever == lever]
        table[lever] = {
            "n": int(len(d)),
            "multiplo": {c: int(((d.es_multiplo) & (d.lag_cat == c)).sum())
                         for c in ("lag_0_1", "lag_ge_2", "nulo")},
            "no_multiplo": {c: int(((~d.es_multiplo) & (d.lag_cat == c)).sum())
                            for c in ("lag_0_1", "lag_ge_2", "nulo")},
            "delta_min": round(float(d.delta.min()), 4) if len(d) else None,
            "delta_max": round(float(d.delta.max()), 4) if len(d) else None,
        }
    sub_bad = df[(df.lag_cat == "lag_ge_2")]
    vent_lag2 = sub_bad[sub_bad.lever.isin(["set_rr", "set_tv", "set_peep",
                                            "set_fio2", "peep_up", "peep_down",
                                            "fio2_down"])]
    crit1 = bool(len(vent_lag2) and (~vent_lag2.es_multiplo).all())
    viol2 = df[(df.abs_k_ge_1) & (df.lag_cat == "lag_ge_2")]
    crit2 = bool(len(viol2) == 0)
    out = {
        "table": table,
        "n_lag_ge_2": int(len(sub_bad)),
        "lag_ge_2_levers": dict(Counter(sub_bad.lever)),
        "lag_ge_2_pairs": sub_bad[["pair_id", "lever", "delta", "es_multiplo", "lag"]]
                          .round(4).to_dict("records"),
        "crit1_lag_ge_2_all_no_multiplo": crit1,
        "crit2_no_abs_k_ge_1_with_lag_ge_2": crit2,
        "crit2_violations": viol2[["pair_id", "lever", "delta", "lag"]]
                             .round(4).to_dict("records"),
        "pass": bool(crit1 and crit2),
        "elapsed_s": round(_time.time() - t0, 1),
    }
    print(f"  G0c: crit1={crit1} crit2={crit2} "
          f"({'PASA' if out['pass'] else 'FALLA'}) lag>=2: {len(sub_bad)} "
          f"[{out['elapsed_s']}s]")
    if not crit1:
        print(f"  G0c contraejemplos crit1: {vent_lag2[~vent_lag2.es_multiplo].to_dict('records')[:5]}")
    if not crit2:
        print(f"  G0c violaciones crit2: {out['crit2_violations'][:5]}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["g0a", "g0b", "g0c"], default=None)
    ap.add_argument("--finalize", action="store_true",
                    help="recalcula el veredicto global 'pass' sin re-ejecutar puertas")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--g0a-n", type=int, default=10,
                    help="pares por palanca a regenerar en G0a (10 por defecto)")
    ap.add_argument("--g0a-tmp", type=str, default=None,
                    help="directorio temporal a reutilizar (resume) en G0a")
    ap.add_argument("--g0a-compare-only", action="store_true",
                    help="no simula: sólo compara lo ya presente en --g0a-tmp")
    ap.add_argument("--target-pair", type=int, default=196761)
    args = ap.parse_args()

    t0 = _time.time()
    pairs = pq.read_table(paths.TOKENS_V2_DIR / "pairs.parquet").to_pandas()
    ann2 = pq.read_table(paths.TOKENS_V2_DIR / "pairs_annotated_v2.parquet").to_pandas()

    cores = {f: sha256(ROOT / f) for f in CORE_FILES}

    # FUSIONA con el JSON existente para no perder puertas ya calculadas (tanto
    # con --only como con --finalize).
    result: dict = {}
    if OUT_JSON.exists():
        try:
            result = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            result = {}
    result.update({
        "date": pd.Timestamp.now().isoformat(),
        "core_shas": cores,
        "n_pairs": int(len(pairs)),
        "n_levers": int(pairs.lever.nunique()),
    })

    if not args.finalize:
        if args.only in (None, "g0a"):
            cfg = SimulatorConfig.from_yaml(CONFIG_PATH)
            cfg.output_dir = CF_DIR
            result["G0a"] = g0a(pairs, cfg.to_dict(), args.workers, args.g0a_n,
                                args.g0a_tmp, args.g0a_compare_only)
        if args.only in (None, "g0b"):
            result["G0b"] = g0b(pairs, args.target_pair)
        if args.only in (None, "g0c"):
            result["G0c"] = g0c(pairs, ann2)

    gates = [g for g in ("G0a", "G0b", "G0c") if g in result]
    if len(gates) == 3:
        result["pass"] = bool(all(result[g]["pass"] for g in gates))
        result["elapsed_s"] = round(_time.time() - t0, 1)
        print(f"  F0 global: {'PASA' if result['pass'] else 'PARA'}")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"  escrito {OUT_JSON}")
    return 0 if result.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
