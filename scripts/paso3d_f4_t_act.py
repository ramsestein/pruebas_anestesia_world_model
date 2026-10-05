"""Paso 3d / Fase 4 (parte 1) — anclaje ``t_act`` y compuertas P3 y N1.

``t_act`` es el instante del ACTO, no del plan:

  * palancas de consigna (``set_rr``, ``set_tv``, ``set_peep`` y el resto de
    overrides de ventilador): el PRIMER instante en que la verdad APLICADA
    (``truth[*_applied]``) difiere entre las dos ramas del par — o sea, el
    momento en que el override surte efecto de verdad. ``t_act_source`` =
    ``truth_applied``. El ``time`` del truth se promueve a float64 (es float32
    en el sidecar; comparar un escalar float64 contra un array float32 redondea
    el escalar: fue el bug de la frontera de G2c).
  * actos puntuales (bolos y cambios de bomba: ``intervention_a``): ``t_act`` =
    ``t_s`` de la primera acción del plan. ``t_act_source`` = ``plan``.

Compuertas de esta parte:

  N1  ningún par de ``set_rr``/``set_tv``/``set_peep`` con diferencia
      REGISTRADA nula (``below_resolution``): δ en la rejilla y palanca con
      efecto ⇒ la consigna registrada se separa al menos un escalón. 100 %.
  P3  en cada muestra no-NaN de la consigna de la palanca POSTERIOR a ``t_act``,
      en ambas ramas, ``(SET_A − SET_B)`` en unidades de registro coincide con
      ``q(applied_A − applied_B)`` del truth en ese instante, con tolerancia de
      medio escalón; se exige en la primera muestra y en todas las siguientes.
      100 %. Se informa aparte, por palanca, el nº de pares con diferencia
      APLICADA distinta de δ (recorte de PEEP y similares).

Nota: P3 se comprueba desde ``t_act`` hasta el FIN DEL REGISTRO, no sólo hasta el
fin del mantenimiento; es un superconjunto (más estricto), así que si pasa,
pasa también la versión restringida.

Uso:
  python scripts/paso3d_f4_t_act.py --workers 8 --out manifests/paso3d_f4_t_act.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

CF_V7_1 = ROOT / "data" / "cf_v7_1"
MODIFIED_LEVERS = ("set_rr", "set_tv", "set_peep")

# ``Action.summary()`` -> "t=6777.8s BOLUS propofol 64.8mg"
_PLAN_T_RE = re.compile(r"^t=([0-9.]+)s")

# Palanca -> (columna aplicada del truth, columna de consigna registrada,
#              escalón de registro EN UNIDADES DE LA COLUMNA APLICADA,
#              factor consigna registrada -> unidades de la columna aplicada).
# OJO con las unidades: ``Primus/SET_TV_L`` va en LITROS y ``tv_applied`` en mL
# (factor 1000, comprobado: registrado −0.09 L frente a aplicado −90 mL), y
# ``Primus/SET_FIO2`` va en % mientras ``fio2_applied`` va en fracción (0.01).
# Para el resto (rpm y cmH2O) coinciden. El δ del manifiesto ya viene en unidades
# de la columna aplicada (mL para set_tv, fracción para set_fio2).
LEVER_SPEC: dict[str, tuple[str, str, float, float]] = {
    "set_rr": ("rr_applied", "Primus/SET_RR_IPPV", 1.0, 1.0),
    "set_tv": ("tv_applied", "Primus/SET_TV_L", 10.0, 1000.0),
    "set_peep": ("peep_applied", "Primus/SET_INTER_PEEP", 1.0, 1.0),
    "set_fio2": ("fio2_applied", "Primus/SET_FIO2", 0.01, 0.01),
    "peep_up": ("peep_applied", "Primus/SET_INTER_PEEP", 1.0, 1.0),
    "peep_down": ("peep_applied", "Primus/SET_INTER_PEEP", 1.0, 1.0),
    "fio2_down": ("fio2_applied", "Primus/SET_FIO2", 0.01, 0.01),
}


def _read(path: Path, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    schema = pq.read_schema(path)
    cols = [c for c in (columns or schema.names) if c in schema.names]
    return pq.read_table(path, columns=cols).to_pandas()


def _task(pair: dict) -> dict:
    cid_a = int(pair["caseid_a"])
    lever = str(pair["lever"])
    split_t = float(pair["split_t"])
    out: dict = {"caseid_a": cid_a, "lever": lever, "pair_id": pair.get("pair_id")}
    truth_a = _read(CF_V7_1 / "truth" / f"{cid_a:04d}_truth.parquet")
    t_a = truth_a["time"].to_numpy(dtype=np.float64)

    spec = LEVER_SPEC.get(lever)
    if spec is not None and pair.get("vent_override_a"):
        applied_col = spec[0]
        if applied_col not in truth_a:
            out["error"] = f"falta {applied_col} en truth"
            return out
        cid_b = int(pair["caseid_b"])
        truth_b = _read(CF_V7_1 / "truth" / f"{cid_b:04d}_truth.parquet")
        t_b = truth_b["time"].to_numpy(dtype=np.float64)
        v_a = truth_a[applied_col].to_numpy(dtype=np.float64)
        v_b = truth_b[applied_col].to_numpy(dtype=np.float64)
        if len(t_a) != len(t_b):
            out["error"] = "longitudes de truth distintas"
            return out
        diff = ~(np.isclose(v_a, v_b, rtol=0.0, atol=1e-9)
                 | (np.isnan(v_a) & np.isnan(v_b)))
        if not diff.any():
            out["t_act"] = None
            out["t_act_source"] = "truth_applied"
            out["no_divergence"] = True
            return out
        i0 = int(np.argmax(diff))
        out["t_act"] = float(t_a[i0])
        out["t_act_source"] = "truth_applied"
        out["lead_s"] = float(t_a[i0] - split_t)
        out["applied_delta_at_act"] = float(v_a[i0] - v_b[i0])
    else:
        # ``intervention_a`` son las CADENAS de ``Action.summary()``
        # ("t=6777.8s BOLUS propofol 64.8mg"), no diccionarios.
        acts = pair.get("intervention_a") or []
        times: list[float] = []
        for a in acts:
            if isinstance(a, str):
                mm = _PLAN_T_RE.match(a)
                if mm:
                    times.append(float(mm.group(1)))
            elif isinstance(a, dict) and "t_s" in a:
                times.append(float(a["t_s"]))
        if not times:
            out["t_act"] = None
            out["t_act_source"] = "plan"
            out["no_action"] = True
            return out
        t_first = min(times)
        out["t_act"] = t_first
        out["t_act_source"] = "plan"
        out["lead_s"] = float(t_first - split_t)
    return out


def _p3_task(pair: dict) -> dict:
    """P3 sobre un par de consigna: registrado vs aplicado tras ``t_act``."""
    cid_a, cid_b = int(pair["caseid_a"]), int(pair["caseid_b"])
    lever = str(pair["lever"])
    applied_col, set_col, step, conv = LEVER_SPEC[lever]
    meta = json.loads((CF_V7_1 / "metadata" / f"cf_pair_{cid_a}.json")
                      .read_text(encoding="utf-8"))
    ov = meta.get("vent_override_a") or {}
    key = {"set_rr": "rr_delta", "set_tv": "tv_delta", "set_peep": "peep_delta",
           "set_fio2": "fio2_set", "peep_up": "peep_delta",
           "peep_down": "peep_delta", "fio2_down": "fio2_delta"}[lever]
    delta_raw = float(ov.get(key, 0.0))
    if lever == "set_fio2":          # override absoluto: δ = (fio2_set − base)
        phys = meta.get("physiology_override") or {}
        delta_raw = delta_raw - float(phys.get("fio2_baseline", 0.50))
    delta_applied = delta_raw      # ya en unidades de la columna aplicada

    res: dict = {"caseid_a": cid_a, "lever": lever, "delta": delta_applied,
                 "n_clipped": 0, "p3_ok": True, "n_samples": 0, "n_bad": 0}
    ta = _read(CF_V7_1 / "truth" / f"{cid_a:04d}_truth.parquet")
    tb = _read(CF_V7_1 / "truth" / f"{cid_b:04d}_truth.parquet")
    t = ta["time"].to_numpy(dtype=np.float64)
    va = ta[applied_col].to_numpy(dtype=np.float64)
    vb = tb[applied_col].to_numpy(dtype=np.float64)
    if len(va) != len(vb):
        res["p3_ok"] = False
        res["error"] = "longitudes distintas"
        return res
    diff_applied = va - vb
    changed = ~(np.isclose(diff_applied, 0.0, rtol=0.0, atol=1e-9)
                | (np.isnan(diff_applied)))
    if not changed.any():
        # Clase ``no_divergence`` ya documentada en 3c: el par no diverge en la
        # serie APLICADA (el plan base sobrescribe la intervención, o el recorte
        # la anula). NO es un fallo de resolución del registro: es un par sin
        # efecto, y se cuenta aparte.
        res["no_divergence"] = True
        return res
    t_act = float(t[int(np.argmax(changed))])
    res["t_act"] = t_act
    # "diferencia APLICADA distinta de δ" (recorte): en el primer instante
    res["n_clipped"] = int(not np.isclose(diff_applied[int(np.argmax(changed))],
                                          delta_applied, rtol=0.0, atol=1e-9))

    # consigna REGISTRADA en ambas ramas (del caso crudo; las ventanas no llevan
    # las columnas Primus/SET_*). Hay que pedir también ``time``.
    ra = _read(CF_V7_1 / "cases" / f"{cid_a:04d}.parquet", ["time", set_col])
    rb = _read(CF_V7_1 / "cases" / f"{cid_b:04d}.parquet", ["time", set_col])
    if set_col not in ra or set_col not in rb:
        res["p3_ok"] = False
        res["error"] = f"falta {set_col}"
        return res
    sa = ra[set_col].to_numpy(dtype=np.float64)
    sb = rb[set_col].to_numpy(dtype=np.float64)
    traw = ra["time"].to_numpy(dtype=np.float64) if "time" in ra else None
    if traw is None:
        res["p3_ok"] = False
        res["error"] = "falta time en el caso"
        return res
    # consigna en la rejilla del truth (forward-fill hold) para alinear
    idx = np.clip(np.searchsorted(traw, t, side="right") - 1, 0, len(sa) - 1)
    ok = (np.searchsorted(traw, t, side="right") - 1) >= 0
    ga = np.where(ok, sa[idx], np.nan)
    gb = np.where(ok, sb[idx], np.nan)
    mask = (t >= t_act) & np.isfinite(ga) & np.isfinite(gb)
    res["n_samples"] = int(mask.sum())
    if not mask.any():
        res["p3_ok"] = False
        res["error"] = "sin muestras tras t_act"
        return res
    reg = (ga[mask] - gb[mask]) * conv
    q = np.round(diff_applied[mask] / step) * step
    bad = np.abs(reg - q) > step / 2.0
    res["n_bad"] = int(bad.sum())
    res["max_abs_reg"] = float(np.nanmax(np.abs(reg))) if mask.any() else 0.0
    res["step"] = step
    res["p3_ok"] = bool(not bad.any())
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        res["first_bad"] = {"t": float(t[mask][i]), "registrado": float(reg[i]),
                            "aplicado_cuantizado": float(q[i])}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path,
                    default=paths.MANIFESTS_DIR / "paso3d_f4_t_act.json")
    args = ap.parse_args()
    t0 = _time.time()

    pairs: list[dict] = []
    for p in sorted((CF_V7_1 / "metadata").glob("cf_pair_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        m["pair_id"] = m.get("pair_id") or int(m["caseid_a"])
        pairs.append(m)
    print(f"pares: {len(pairs)}")

    # ── t_act de los 6345 pares ────────────────────────────────────────────
    res: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_task, m) for m in pairs]
        for i, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if i % 1000 == 0:
                print(f"  t_act: {i}/{len(pairs)} ({_time.time() - t0:.0f}s)", flush=True)
    by_source: dict[str, int] = {}
    for r in res:
        by_source[r.get("t_act_source", "?")] = by_source.get(r.get("t_act_source", "?"), 0) + 1
    no_div = [r for r in res if r.get("no_divergence")]
    leads = [r["lead_s"] for r in res if r.get("lead_s") is not None]

    # ── P3 y N1 sobre las tres palancas modificadas ────────────────────────
    mods = [m for m in pairs if m["lever"] in MODIFIED_LEVERS]
    print(f"pares modificados para P3/N1: {len(mods)}")
    p3: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_p3_task, m) for m in mods]
        for i, f in enumerate(as_completed(futs), 1):
            p3.append(f.result())
            if i % 300 == 0:
                print(f"  P3: {i}/{len(mods)} ({_time.time() - t0:.0f}s)", flush=True)

    with_div = [r for r in p3 if not r.get("no_divergence")]
    nodiv = [r for r in p3 if r.get("no_divergence")]
    p3_bad = [r for r in with_div if not r["p3_ok"]]
    n_bad_samples = sum(int(r.get("n_bad", 0)) for r in with_div)
    n_samples = sum(int(r.get("n_samples", 0)) for r in with_div)
    below_res = [r for r in with_div
                 if float(r.get("max_abs_reg", 0.0)) < float(r.get("step", 1.0)) / 2.0]
    by_lever: dict[str, dict] = {}
    for r in with_div:
        d = by_lever.setdefault(r["lever"], {"n": 0, "n_clipped": 0, "n_p3_bad": 0,
                                             "n_bad_samples": 0, "n_samples": 0})
        d["n"] += 1
        d["n_clipped"] += int(r.get("n_clipped", 0))
        d["n_p3_bad"] += int(not r["p3_ok"])
        d["n_bad_samples"] += int(r.get("n_bad", 0))
        d["n_samples"] += int(r.get("n_samples", 0))

    out = {
        "date": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(pairs),
        "n_modified_pairs": len(mods),
        "t_act": {
            "by_source": by_source,
            "n_no_divergence": len(no_div),
            "no_divergence_by_lever": _count(no_div),
            "lead_s_min": float(min(leads)) if leads else None,
            "lead_s_median": float(np.median(leads)) if leads else None,
            "lead_s_max": float(max(leads)) if leads else None,
        },
        "N1": {
            "n_tested": len(with_div),
            "n_below_resolution": len(below_res),
            "examples_below_resolution": [r["caseid_a"] for r in below_res[:10]],
            "pass": bool(not below_res),
        },
        "no_divergence_pares": {
            "n": len(nodiv),
            "by_lever": _count(nodiv),
            "nota": ("pares sin divergencia en la serie APLICADA (clase de 3c: el "
                     "plan base sobrescribe la intervención o el recorte la anula); "
                     "no son fallos de resolución del registro"),
        },
        "P3": {
            "n_tested": len(with_div),
            "n_ok": len(with_div) - len(p3_bad),
            "n_bad": len(p3_bad),
            "n_samples": n_samples,
            "n_bad_samples": n_bad_samples,
            "bad_sample_fraction": (n_bad_samples / n_samples) if n_samples else None,
            "by_lever": by_lever,
            "examples_bad": [{k: v for k, v in r.items() if k != "delta"}
                             for r in p3_bad[:5]],
            "pass": bool(not p3_bad),
        },
        "elapsed_s": round(_time.time() - t0, 1),
    }
    out["pass"] = bool(out["P3"]["pass"] and out["N1"]["pass"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_pairs", "N1", "P3")},
                     indent=2, ensure_ascii=False)[:1200])
    print(f"t_act: {json.dumps(out['t_act'], ensure_ascii=False)}")
    print(f"FASE 4 (t_act/P3/N1): {'PASA' if out['pass'] else 'FALLA'} -> {args.out}")
    return 0 if out["pass"] else 1


def _count(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["lever"]] = out.get(r["lever"], 0) + 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    raise SystemExit(main())
