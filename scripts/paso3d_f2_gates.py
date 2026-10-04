"""Paso 3d / Fase 2 — compuertas G2a-G2d sobre la cohorte regenerada cf_v7_1.

Terminología. En este generador (``generate_f2_cf._make_intervention``) la rama
**A** = ``caseid_a`` es la INTERVENIDA y la rama **B** = ``caseid_b`` es el
CONTROL: en las palancas de consigna el override vive en ``vent_override_a`` y
``vent_override_b`` es ``None``; en las farmacológicas la acción vive en
``intervention_a``. Por eso:

  G2a  los 5220 pares de palancas NO modificadas son idénticos bit a bit a
       cf_v7: manifiesto ``cf_pair_*.json`` + ``cases``/``truth``/``clinical``/
       ``metadata`` de AMBAS ramas.
  G2b  en los 1125 pares modificados (set_rr / set_tv / set_peep), la rama de
       CONTROL (B) es idéntica bit a bit a cf_v7 en los cuatro artefactos.
  G2c  en esos mismos pares, la rama INTERVENIDA (A) coincide con su versión
       cf_v7 en TODAS las columnas de ``cases`` y ``truth`` para ``t < split_t``
       (ninguna columna excluida); además el encabezado (caseid_a, caseid_b,
       seed, split_t, lever, collection) y el ``subjectid`` clínico son
       idénticos.

       Frontera (``t < split_t``, estricto). El override contrafactual está
       ACTIVO desde ``split_t``: el primer instante de simulación ``>= split_t``
       ya muestra el valor aplicado nuevo (verificado sobre el par 198303, el
       único que falló al usar ``t <= split_t``). Además, ``time`` en el
       sidecar ``truth`` es float32, así que comparar contra el escalar float64
       ``split_t`` lo REDONDEA a float32: para ``split_t = 9500.49995589281``
       el redondeo da exactamente 9500.5 y el filtro ``<=`` colaba la primera
       fila post-intervención. El prefijo se filtra por tanto con ``time``
       promovido a float64 y comparación estricta.
  G2d  el δ registrado en el manifiesto está en la REJILLA de registro:
       ``|δ - round(δ)| < 1e-9``, ``δ != 0`` y dentro del rango declarado.

Se informa además, sin ser criterio, en cuántos pares la rama intervenida A
DIFIERE de la de cf_v7 tras ``split_t`` (es decir, en cuántos pares el cambio de
muestreo de v7.1 tuvo efecto).

Uso:
  python scripts/paso3d_f2_gates.py --v7-1 data/cf_v7_1 --workers 12
  python scripts/paso3d_f2_gates.py --limit 20      # prueba rápida
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: F401  (orden de import: numpy -> pandas -> pyarrow)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import paths  # noqa: E402
from paso3d_f0_diagnostico import (  # noqa: E402
    _diff_columns,
    _read_parquet,
)

V7 = Path(paths.COHORTS["cf_v7"])
V71 = ROOT / "data" / "cf_v7_1"

# Palancas cuyo muestreo cambió en v7.1 (Fase 1). Las 18 restantes deben ser
# idénticas bit a bit.
MODIFIED_LEVERS = ("set_rr", "set_tv", "set_peep")

# δ registrado -> (clave del override, escala a unidades de escalón, |δ| máximo
# en unidades de escalón).
DELTA_SPEC: dict[str, tuple[str, float, float]] = {
    "set_rr": ("rr_delta", 1.0, 4.0),      # rpm, escalón 1
    "set_tv": ("tv_delta", 10.0, 15.0),    # mL, escalón 10 mL (0.01 L)
    "set_peep": ("peep_delta", 1.0, 5.0),  # cmH2O, escalón 1
}

EXPECTED_COUNTS = {"pharma": 870, "vent": 600, "learning": 4875}


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------

def _json_read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _load_manifests(cf_dir: Path, limit: int = 0) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted((cf_dir / "metadata").glob("cf_pair_*.json")):
        m = _json_read(p)
        out[int(m["caseid_a"])] = m
        if limit and len(out) >= limit:
            break
    return out


def _parquet_path(cf_dir: Path, kind: str, cid: int) -> Path:
    if kind == "case":
        return cf_dir / "cases" / f"{cid:04d}.parquet"
    if kind == "truth":
        return cf_dir / "truth" / f"{cid:04d}_truth.parquet"
    if kind == "clinical":
        return cf_dir / "clinical" / f"{cid:04d}_clinical.parquet"
    raise ValueError(kind)


def _meta_path(cf_dir: Path, cid: int) -> Path:
    return cf_dir / "metadata" / f"{cid:04d}_meta.json"


def _cmp_parquet(dir_a: Path, dir_b: Path, kind: str, cid: int,
                 prefix_t: float | None = None) -> list[str]:
    """Columnas que difieren entre las dos cohortes. ``prefix_t`` restringe a
    las filas con ``time < prefix_t`` (estrictamente anteriores al split; ver
    la nota de frontera del encabezado). ``time`` se promueve a float64 porque
    en el sidecar ``truth`` es float32."""
    pa, pb = _parquet_path(dir_a, kind, cid), _parquet_path(dir_b, kind, cid)
    if not pa.exists() or not pb.exists():
        return ["<falta_archivo>"]
    da, db = _read_parquet(pa), _read_parquet(pb)
    if prefix_t is not None:
        ta = da["time"].to_numpy(dtype=np.float64)
        tb = db["time"].to_numpy(dtype=np.float64)
        da = da.loc[ta < prefix_t].reset_index(drop=True)
        db = db.loc[tb < prefix_t].reset_index(drop=True)
    return _diff_columns(da, db)


def _cmp_meta(dir_a: Path, dir_b: Path, cid: int) -> list[str]:
    pa, pb = _meta_path(dir_a, cid), _meta_path(dir_b, cid)
    ea, eb = pa.exists(), pb.exists()
    if ea != eb:
        return ["<falta_archivo>"]
    if not ea:
        return []
    return [] if _json_read(pa) == _json_read(pb) else ["<json>"]


def _changed_after(dir_a: Path, dir_b: Path, cid: int, split_t: float) -> bool:
    """¿Difiere la rama A de cf_v7_1 respecto a la de cf_v7 DESPUÉS del
    split? (es decir, ¿el cambio de muestreo de v7.1 tuvo efecto?)"""
    pa, pb = _parquet_path(dir_a, "case", cid), _parquet_path(dir_b, "case", cid)
    if not pa.exists() or not pb.exists():
        return False
    da, db = _read_parquet(pa), _read_parquet(pb)
    ta = da["time"].to_numpy(dtype=np.float64)
    tb = db["time"].to_numpy(dtype=np.float64)
    da = da.loc[ta > split_t].reset_index(drop=True)
    db = db.loc[tb > split_t].reset_index(drop=True)
    return len(_diff_columns(da, db)) > 0


def _subjectid(cf_dir: Path, cid: int) -> int | None:
    p = _parquet_path(cf_dir, "clinical", cid)
    if not p.exists():
        return None
    tbl = _read_parquet(p)
    return int(tbl["subjectid"].iloc[0]) if "subjectid" in tbl else None


# ---------------------------------------------------------------------------
# tareas por par
# ---------------------------------------------------------------------------

def _task_g2a(task: dict) -> dict:
    cid_a = int(task["caseid_a"])
    diffs: dict[str, list[str]] = {}
    for cid in (cid_a, cid_a + 1):
        for kind in ("case", "truth", "clinical"):
            d = _cmp_parquet(V7, V71, kind, cid)
            if d:
                diffs[f"{cid}:{kind}"] = d
        d = _cmp_meta(V7, V71, cid)
        if d:
            diffs[f"{cid}:meta"] = d
    return {"caseid_a": cid_a, "diffs": diffs}


def _task_g2bc(task: dict) -> dict:
    cid_a = int(task["caseid_a"])
    split_t = float(task["split_t"])
    # G2b — rama de control B, identidad completa.
    b: dict[str, list[str]] = {}
    for kind in ("case", "truth", "clinical"):
        d = _cmp_parquet(V7, V71, kind, cid_a + 1)
        if d:
            b[f"{cid_a + 1}:{kind}"] = d
    d = _cmp_meta(V7, V71, cid_a + 1)
    if d:
        b[f"{cid_a + 1}:meta"] = d
    # G2c — rama intervenida A, prefijo t < split_t.
    a: dict[str, list[str]] = {}
    n_pref = 0
    for kind in ("case", "truth"):
        d = _cmp_parquet(V7, V71, kind, cid_a, prefix_t=split_t)
        if d:
            a[f"{cid_a}:{kind}"] = d
        if kind == "truth":
            p = _parquet_path(V71, "truth", cid_a)
            if p.exists():
                t = _read_parquet(p)["time"].to_numpy(dtype=np.float64)
                n_pref = int((t < split_t).sum())
    d = _cmp_meta(V7, V71, cid_a)
    if d:
        a[f"{cid_a}:meta"] = d
    sa, sb = _subjectid(V7, cid_a), _subjectid(V71, cid_a)
    if sa != sb:
        a[f"{cid_a}:subjectid"] = [f"{sa} != {sb}"]
    return {
        "caseid_a": cid_a,
        "b_diffs": b,
        "a_diffs": a,
        "n_prefix_rows": n_pref,
        "a_changed": _changed_after(V7, V71, cid_a, split_t),
    }


# ---------------------------------------------------------------------------
# G2d
# ---------------------------------------------------------------------------

def _delta_on_grid(meta: dict) -> tuple[bool, float | None, str]:
    """(ok, k, motivo) con k = δ en unidades de escalón (debe ser entero != 0)."""
    lever = str(meta["lever"])
    if lever not in DELTA_SPEC:
        return False, None, "palanca_desconocida"
    key, step, kmax = DELTA_SPEC[lever]
    ov = meta.get("vent_override_a") or {}
    if key not in ov:
        return False, None, "sin_delta"
    delta = float(ov[key])
    k = delta / step
    if abs(k - round(k)) >= 1e-9:
        return False, k, "sub_escalon"
    if round(k) == 0:
        return False, k, "delta_nulo"
    if abs(round(k)) > kmax:
        return False, k, "fuera_de_rango"
    return True, k, "ok"


def g2d(pairs: dict[int, dict]) -> dict:
    bad: list[dict] = []
    k_counter: dict[str, dict[str, int]] = {}
    for cid, meta in pairs.items():
        ok, k, why = _delta_on_grid(meta)
        if not ok:
            bad.append({"caseid_a": cid, "lever": meta["lever"],
                        "k": None if k is None else round(k, 6), "motivo": why})
        else:
            lever = str(meta["lever"])
            k_counter.setdefault(lever, {})
            key = str(int(round(k)))
            k_counter[lever][key] = k_counter[lever].get(key, 0) + 1
    return {
        "n_tested": len(pairs),
        "n_on_grid": len(pairs) - len(bad),
        "n_off_grid": len(bad),
        "off_grid_examples": bad[:10],
        "delta_step_counts": {lev: dict(sorted(c.items(), key=lambda kv: int(kv[0])))
                             for lev, c in sorted(k_counter.items())},
        "pass": not bad,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _run_tasks(pool_tasks: list[dict], fn, workers: int, label: str) -> list[dict]:
    res: list[dict] = []
    t0 = _time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, t): t["caseid_a"] for t in pool_tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                res.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                res.append({"caseid_a": futs[fut], "error": repr(exc)})
            if i % 500 == 0 or i == len(pool_tasks):
                el = _time.time() - t0
                print(f"  {label}: {i}/{len(pool_tasks)}  ({el:.0f}s)", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7", type=Path, default=V7)
    ap.add_argument("--v7-1", type=Path, default=V71)
    ap.add_argument("--out", type=Path, default=paths.MANIFESTS_DIR / "paso3d_f2.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0,
                    help="solo N pares por grupo (prueba rápida)")
    args = ap.parse_args()

    t0 = _time.time()
    v7, v71 = Path(args.v7), Path(args.v7_1)
    for p in (v7, v71):
        if not p.exists():
            print(f"ERROR: no existe {p}", file=sys.stderr)
            return 2
    if not (v71 / "metadata").exists():
        print(f"ERROR: {v71} no tiene 'metadata/' (¿generación incompleta?)",
              file=sys.stderr)
        return 2

    print(f"cf_v7   = {v7}")
    print(f"cf_v7_1 = {v71}")
    mans_a = _load_manifests(v7, args.limit)
    mans_b = _load_manifests(v71, args.limit)
    print(f"pares cf_v7={len(mans_a)}  cf_v7_1={len(mans_b)}")

    # Emparejamiento por (caseid_a): ninguna alta ni baja.
    set_a, set_b = set(mans_a), set(mans_b)
    only_a, only_b = sorted(set_a - set_b), sorted(set_b - set_a)
    counts_b = {}
    for m in mans_b.values():
        counts_b[m["collection"]] = counts_b.get(m["collection"], 0) + 1

    # Manifiestos idénticos palanca a palanca.
    mod_ids = [c for c, m in sorted(mans_b.items()) if m["lever"] in MODIFIED_LEVERS]
    oth_ids = [c for c, m in sorted(mans_b.items()) if m["lever"] not in MODIFIED_LEVERS]
    manifest_diff: dict[str, list[str]] = {}
    for cid in oth_ids:
        if cid in mans_a and mans_a[cid] != mans_b[cid]:
            keys = [k for k in set(mans_a[cid]) | set(mans_b[cid])
                    if mans_a[cid].get(k) != mans_b[cid].get(k)]
            manifest_diff[str(cid)] = sorted(keys)
    print(f"pares modificados={len(mod_ids)}  otras palancas={len(oth_ids)}")
    print(f"manifiestos divergentes en las otras palancas: {len(manifest_diff)}")

    # ── G2a ────────────────────────────────────────────────────────────────
    print("G2a — identidad bit a bit de las 18 palancas no modificadas")
    res_a = _run_tasks([{"caseid_a": c} for c in oth_ids], _task_g2a,
                       args.workers, "G2a")
    a_bad = [r for r in res_a if r.get("diffs") or r.get("error")]
    g2a = {
        "n_tested": len(oth_ids),
        "n_identical": len(oth_ids) - len(a_bad),
        "n_diff": len(a_bad),
        "manifest_diffs": len(manifest_diff),
        "examples": [{"caseid_a": r["caseid_a"],
                      "detalle": r.get("diffs") or r.get("error")} for r in a_bad[:10]],
        "pass": not a_bad and not manifest_diff,
    }

    # ── G2b / G2c / G2d ────────────────────────────────────────────────────
    print("G2b/G2c — paridad de la cohorte intervenida con cf_v7")
    tasks = [{"caseid_a": c, "split_t": float(mans_b[c]["split_t"])} for c in mod_ids]
    res_bc = _run_tasks(tasks, _task_g2bc, args.workers, "G2b/c")
    b_bad = [r for r in res_bc if r.get("b_diffs") or r.get("error")]
    c_bad = [r for r in res_bc if r.get("a_diffs") or r.get("error")]
    n_changed = sum(1 for r in res_bc if r.get("a_changed"))
    pref_rows = [r.get("n_prefix_rows", 0) for r in res_bc]
    g2b = {
        "n_tested": len(mod_ids),
        "n_identical": len(mod_ids) - len(b_bad),
        "n_diff": len(b_bad),
        "examples": [{"caseid_a": r["caseid_a"],
                      "detalle": r.get("b_diffs") or r.get("error")} for r in b_bad[:10]],
        "pass": not b_bad,
    }
    g2c = {
        "n_tested": len(mod_ids),
        "n_prefix_identical": len(mod_ids) - len(c_bad),
        "n_prefix_diff": len(c_bad),
        "n_a_changed_vs_v7_after_split": n_changed,
        "prefix_rows_min": min(pref_rows) if pref_rows else 0,
        "prefix_rows_median": int(np.median(pref_rows)) if pref_rows else 0,
        "examples": [{"caseid_a": r["caseid_a"],
                      "detalle": r.get("a_diffs") or r.get("error")} for r in c_bad[:10]],
        "pass": not c_bad and min(pref_rows or [0]) > 0,
    }
    g2d_res = g2d({c: mans_b[c] for c in mod_ids})

    counts_ok = all(counts_b.get(k) == v for k, v in EXPECTED_COUNTS.items())

    result = {
        "date": datetime.now(timezone.utc).isoformat(),
        "cf_v7": str(v7),
        "cf_v7_1": str(v71),
        "n_pairs_v7": len(mans_a),
        "n_pairs_v7_1": len(mans_b),
        "counts_by_collection": counts_b,
        "counts_ok": counts_ok,
        "only_in_v7": only_a[:20],
        "only_in_v7_1": only_b[:20],
        "n_modified_pairs": len(mod_ids),
        "n_other_pairs": len(oth_ids),
        "g2a": g2a,
        "g2b": g2b,
        "g2c": g2c,
        "g2d": g2d_res,
        "pass": bool(not only_a and not only_b and counts_ok
                     and g2a["pass"] and g2b["pass"] and g2c["pass"] and g2d_res["pass"]),
        "elapsed_s": round(_time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    for name in ("g2a", "g2b", "g2c", "g2d"):
        print(f"{name.upper()}: {'PASA' if result[name]['pass'] else 'FALLA'}  "
              f"({json.dumps({k: v for k, v in result[name].items() if not isinstance(v, (list, dict))}, ensure_ascii=False)})")
    print(f"FASE 2 GLOBAL: {'PASA' if result['pass'] else 'FALLA'}  "
          f"({result['elapsed_s']}s)  -> {args.out}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
