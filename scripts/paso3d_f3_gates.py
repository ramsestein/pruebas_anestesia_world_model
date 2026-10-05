"""Paso 3d / Fase 3 — compuertas G3 sobre windows_v4_1 (y pk/tokens_v*_1).

G3a  el split es IDÉNTICO a windows_v4: ``split.parquet`` y ``registry.parquet``
     iguales bit a bit y ``cases.parquet`` igual salvo ``process_time_s`` (columna
     de tiempo de proceso, no dato).
G3b  todas las filas NO INTERVENIDAS son idénticas a windows_v4:
       - fuentes real / synthetic_v7 / vaso_reinf_v7: comparación completa
         (todas las columnas, todas las filas) partición a partición;
       - fuente cf_v7: las filas de los casos de CONTROL (caseid_b) iguales en
         todas las columnas, y las filas de los casos INTERVENIDOS (caseid_a)
         de ventanas que cierran antes del split (t0 + 60 <= split_t) iguales.
     Las filas posteriores al split de los casos intervenidos SÍ deben diferir
     (se informa el recuento, sin ser criterio).

Nota de procedencia (hallazgo de esta fase). ``window.py`` cambió DESPUÉS de
construir windows_v4: el manifiesto de windows_v4 registra
``window_py_sha256 = 9508ccd0...`` y windows_v4_1 registra ``e714476c...``
(refactor 22ba809 "Paso 0 C: repuntar tokens y autoencoder a paths.py", 30/09).
Las dos cohortes se construyeron por tanto con revisiones distintas del módulo,
así que G3b NO es una comparación de código idéntico: es la verificación
EMPÍRICA de que el refactor no cambió ninguna fila no intervenida. Si G3b pasa,
la comparación es válida; si fallara, habría que reconstruir windows_v4 con el
window.py actual para separar los dos efectos.

Uso:
  python scripts/paso3d_f3_gates.py --stage windows --workers 10
  python scripts/paso3d_f3_gates.py --stage pk --workers 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths  # noqa: E402
from paso3d_f0_diagnostico import _diff_columns, _read_parquet  # noqa: E402

W4 = ROOT / "data" / "windows_v4"
W41 = ROOT / "data" / "windows_v4_1"
P2 = ROOT / "data" / "pk_v2"
P21 = ROOT / "data" / "pk_v2_1"
T2 = ROOT / "data" / "tokens_v2"
T21 = ROOT / "data" / "tokens_v2_1"
CF_V71 = ROOT / "data" / "cf_v7_1"
WIN_LEN_S = 60.0          # t1 = t0 + 60 (convención de la fase 0 de 3c)
NON_CF = ("real", "synthetic_v7", "vaso_reinf_v7")


# ---------------------------------------------------------------------------
# G3a
# ---------------------------------------------------------------------------

def g3a() -> dict:
    out: dict = {"files": {}}
    for f in ("split.parquet", "registry.parquet", "cases.parquet"):
        a = _read_parquet(W4 / f)
        b = _read_parquet(W41 / f)
        diffs = _diff_columns(a, b)
        out["files"][f] = {"rows_v4": len(a), "rows_v4_1": len(b),
                           "diff_columns": diffs}
    split_ok = not out["files"]["split.parquet"]["diff_columns"]
    reg_ok = not out["files"]["registry.parquet"]["diff_columns"]
    cases_d = out["files"]["cases.parquet"]["diff_columns"]
    cases_ok = all(c == "process_time_s" for c in cases_d)
    out["split_identical"] = split_ok
    out["registry_identical"] = reg_ok
    out["cases_identical_except_process_time"] = cases_ok
    out["pass"] = bool(split_ok and reg_ok and cases_ok)
    return out


# ---------------------------------------------------------------------------
# G3b — ventanas
# ---------------------------------------------------------------------------

def _pair_maps() -> tuple[dict[int, float], set[int]]:
    """{caseid_a: split_t} y {caseid_b} de la cohorte cf_v7_1."""
    split_by_a: dict[int, float] = {}
    control: set[int] = set()
    for p in (CF_V71 / "metadata").glob("cf_pair_*.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        split_by_a[int(m["caseid_a"])] = float(m["split_t"])
        control.add(int(m["caseid_b"]))
    return split_by_a, control


def _sort_key(df: pd.DataFrame, time_col: str = "t") -> pd.DataFrame:
    return df.sort_values(["caseid", time_col], kind="stable").reset_index(drop=True)


# ``run_full`` de tokenize aplica ``apply_normalization`` a las features
# ``vent_*_t0`` / ``vent_*_t1`` con estadísticos ACUMULADOS sobre el corpus
# procesado, así que son z-scores: cambiar cualquier caso del corpus desplaza
# TODOS los valores del corpus (media y desviación) y ninguna fila —ni siquiera
# las de real— puede ser idéntica bit a bit entre dos corpus distintos. Para las
# filas NO intervenidas la relación debe ser EXACTAMENTE afín (misma x cruda,
# otra media/desv):
#     v_nuevo = a + b * v_viejo,  con a = (m1 - m2)/s2  y  b = s1/s2
# Eso es lo que se comprueba con el ajuste afín. Las demás columnas
# (drug_*, ctx_*, vent_*_proxy, vent_*_mask y los metadatos) no se normalizan
# así y se exigen idénticas.
AFFINE_ATOL = 2e-2   # z-scores en float32


def _is_affine_col(col: str) -> bool:
    return col.startswith("vent_") and (col.endswith("_t0") or col.endswith("_t1"))


def _affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """(a, b, r2, max_abs_resid) de ``y ~ a + b x``, ignorando NaN."""
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    xv = x[m].astype(np.float64)
    yv = y[m].astype(np.float64)
    if float(np.ptp(xv)) == 0.0:
        return (float(yv.mean()), 0.0, 1.0, float(np.abs(yv - yv.mean()).max()))
    b, a = np.polyfit(xv, yv, 1)
    resid = np.abs(yv - (a + b * xv))
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return (float(a), float(b), float(r2), float(resid.max()))


def _task_identity(task: dict) -> dict:
    """Compara una partición entre las dos cohortes.

    ``time_col`` es la columna de tiempo (``t`` en ventanas y pk, ``t1`` en
    tokens, donde cada fila es una ventana) y ``win_len`` lo que hay que sumarle
    para obtener el CIERRE de la fila (60 s en ventanas, 0 en pk/tokens). El
    prefijo es ``cierre < split_t``: una fila que cierre en ``split_t`` ya
    contiene el valor de ese instante, y el override está activo desde
    ``split_t`` (ver la nota de frontera de paso3d_f2_gates).
    """
    rel = task["rel"]
    source = task["source"]
    dir_a, dir_b = Path(task["dir_a"]), Path(task["dir_b"])
    time_col = task.get("time_col", "t")
    win_len = float(task.get("win_len", 0.0))
    a = _sort_key(_read_parquet(dir_a / rel), time_col)
    b = _sort_key(_read_parquet(dir_b / rel), time_col)
    affine = bool(task.get("affine", False))
    res: dict = {"rel": rel, "source": source, "rows": int(len(a))}
    if len(a) != len(b):
        res["diffs"] = ["<filas>"]
        return res

    if source != "cf_v7":
        m_ctrl = np.ones(len(a), dtype=bool)
        m_pref = np.zeros(len(a), dtype=bool)
        m_post = np.zeros(len(a), dtype=bool)
    else:
        split_by_a, control = _PAIRS if _PAIRS is not None else _pair_maps()
        cid = a["caseid"].to_numpy()
        s = pd.Series(cid).map(split_by_a)
        t_hi = a[time_col].to_numpy(dtype=np.float64) + win_len
        m_ctrl = s.isna().to_numpy()
        m_pref = (~s.isna()).to_numpy() & (t_hi < s.fillna(-1.0).to_numpy())
        m_post = (~s.isna()).to_numpy() & ~m_pref
        unknown = set(np.unique(cid)) - set(split_by_a) - control
        res["unknown_caseids"] = int(len(unknown))
        res["n_cases"] = int(len(np.unique(cid)))
    m_ni = m_ctrl | m_pref          # filas NO intervenidas
    res["rows_control"] = int(m_ctrl.sum())
    res["rows_intervened_prefix"] = int(m_pref.sum())
    res["rows_intervened_post"] = int(m_post.sum())

    def _diffs(mask: np.ndarray) -> list[str]:
        d = _diff_columns(a.loc[mask].reset_index(drop=True),
                          b.loc[mask].reset_index(drop=True))
        return [c for c in d if c.startswith("<") or not affine or not _is_affine_col(c)]

    res["diffs"] = _diffs(m_ni)
    res["post_rows_differ"] = bool(_diffs(m_post)) if m_post.any() else False

    if affine:
        cols = [c for c in a.columns if _is_affine_col(c)]
        r2_min, res_max = 1.0, 0.0
        res_max_post, cols_bad = 0.0, []
        for c in cols:
            xa = a[c].to_numpy(dtype=np.float64)
            yb = b[c].to_numpy(dtype=np.float64)
            a_fit, b_fit, r2, rmax = _affine_fit(xa[m_ni], yb[m_ni])
            if not np.isfinite(r2) or r2 < 1.0 - 1e-6 or (np.isfinite(rmax) and rmax > AFFINE_ATOL):
                cols_bad.append(c)
            if np.isfinite(r2):
                r2_min = min(r2_min, float(r2))
            if np.isfinite(rmax):
                res_max = max(res_max, float(rmax))
            if m_post.any() and np.isfinite(rmax):
                m = np.isfinite(xa) & np.isfinite(yb)
                mp = m_post & m
                if mp.any():
                    rp = np.abs(yb[mp] - (a_fit + b_fit * xa[mp]))
                    res_max_post = max(res_max_post, float(rp.max()))
        res["affine_cols"] = len(cols)
        res["affine_cols_bad"] = cols_bad
        res["affine_r2_min"] = r2_min
        res["affine_resid_max"] = res_max
        res["affine_resid_max_post"] = res_max_post
    return res


_PAIRS: dict | None = None


def _init_worker() -> None:
    """Carga los mapas de pares UNA vez por worker (spawn en Windows)."""
    global _PAIRS
    _PAIRS = _pair_maps()


def _task_missing(task: dict) -> dict:
    return {"rel": task["rel"], "source": task["source"],
            "diffs": ["<falta_particion>"]}


def identity_gate(dir_a: Path, dir_b: Path, workers: int, win_len: float,
                  label: str = "", time_col: str = "t",
                  affine: bool = False) -> dict:
    """Identidad de las filas NO intervenidas entre dos raíces de particiones
    (``<raíz>/source=*/split=*/part-*.parquet``). ``affine=True`` para tokens:
    las columnas ``vent_*_t0/t1`` son z-scores del corpus y se comprueban con un
    ajuste afín en vez de por igualdad exacta."""
    tasks: list[dict] = []
    for src in sorted(dir_a.glob("source=*")):
        source = src.name.split("=", 1)[1]
        for part in sorted(src.glob("split=*/part-*.parquet")):
            rel = str(part.relative_to(dir_a)).replace("\\", "/")
            tasks.append({"rel": rel, "source": source, "dir_a": str(dir_a),
                          "dir_b": str(dir_b), "win_len": win_len,
                          "time_col": time_col, "affine": affine,
                          "missing": not (dir_b / rel).exists()})
    print(f"{label}: {len(tasks)} particiones")
    res: list[dict] = []
    t0 = _time.time()
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as ex:
        futs = [ex.submit(_task_missing if t["missing"] else _task_identity, t)
                for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                res.append(f.result())
            except Exception as exc:  # noqa: BLE001
                res.append({"rel": "?", "source": "?", "diffs": [f"<error {exc!r}>"]})
            if i % 50 == 0 or i == len(futs):
                print(f"  {label}: {i}/{len(futs)}  ({_time.time() - t0:.0f}s)", flush=True)

    non_cf = [r for r in res if r["source"] != "cf_v7"]
    cf = [r for r in res if r["source"] == "cf_v7"]
    bad_non_cf = [r for r in non_cf if r.get("diffs")]
    bad_cf = [r for r in cf if r.get("diffs") or r.get("unknown_caseids")]
    affine_bad = [r for r in res if r.get("affine_cols_bad")]
    r2_vals = [r.get("affine_r2_min") for r in res if np.isfinite(r.get("affine_r2_min", np.nan))]
    resmax = [r.get("affine_resid_max") for r in res if np.isfinite(r.get("affine_resid_max", np.nan))]
    respost = [r.get("affine_resid_max_post") for r in res
               if np.isfinite(r.get("affine_resid_max_post", np.nan))]
    out: dict = {
        "n_partitions": len(res),
        "criterio": ("columnas exactas identicas en filas no intervenidas"
                     + (" + ajuste afin (R2=1) en las z-scores vent_*_t0/t1"
                        if affine else "")),
        "non_cf": {
            "n_partitions": len(non_cf),
            "rows": int(sum(r.get("rows", 0) for r in non_cf)),
            "n_partitions_with_diffs": len(bad_non_cf),
            "examples": [{"rel": r["rel"], "diffs": r.get("diffs")}
                         for r in bad_non_cf[:5]],
        },
        "cf": {
            "n_partitions": len(cf),
            "rows": int(sum(r.get("rows", 0) for r in cf)),
            "rows_control": int(sum(r.get("rows_control", 0) for r in cf)),
            "rows_intervened_prefix": int(sum(r.get("rows_intervened_prefix", 0) for r in cf)),
            "rows_intervened_post": int(sum(r.get("rows_intervened_post", 0) for r in cf)),
            "n_partitions_with_diffs": len(bad_cf),
            "n_partitions_post_differ": sum(1 for r in cf if r.get("post_rows_differ")),
            "unknown_caseids": int(sum(r.get("unknown_caseids", 0) for r in cf)),
            "n_cases": int(sum(r.get("n_cases", 0) for r in cf)),
            "examples": [{"rel": r["rel"], "diffs": r.get("diffs")}
                         for r in bad_cf[:5]],
        },
        "elapsed_s": round(_time.time() - t0, 1),
    }
    if affine:
        out["affine"] = {
            "n_cols_per_partition": int(sum(r.get("affine_cols", 0) for r in res)),
            "n_partitions_all_cols_affine": len(res) - len(affine_bad),
            "r2_min": float(min(r2_vals)) if r2_vals else None,
            "resid_max": float(max(resmax)) if resmax else None,
            "resid_max_post_split": float(max(respost)) if respost else None,
            "examples_bad": [{"rel": r["rel"], "cols": r["affine_cols_bad"]}
                             for r in affine_bad[:5]],
        }
    out["pass"] = bool(not bad_non_cf and not bad_cf and not affine_bad
                       and out["cf"]["unknown_caseids"] == 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["windows", "pk", "tokens"], default="windows")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    t0 = _time.time()
    result: dict = {"date": datetime.now(timezone.utc).isoformat()}

    if args.stage == "windows":
        out_path = args.out or (paths.MANIFESTS_DIR / "paso3d_f3_windows.json")
        result.update({"artifact": "windows", "v4": str(W4), "v4_1": str(W41),
                       "window_py_sha256_v4": json.loads((W4 / "manifest.json")
                                                         .read_text("utf-8"))["window_py_sha256"],
                       "window_py_sha256_v4_1": json.loads((W41 / "manifest.json")
                                                           .read_text("utf-8"))["window_py_sha256"]})
        print("G3a — split / registry / cases")
        result["g3a"] = g3a()
        print(f"  G3a: {'PASA' if result['g3a']['pass'] else 'FALLA'}  "
              f"{json.dumps(result['g3a']['files'], ensure_ascii=False)[:200]}")
        print("G3b — identidad de las filas no intervenidas (ventanas)")
        result["g3b"] = identity_gate(W4 / "windows", W41 / "windows",
                                       args.workers, WIN_LEN_S, "G3b")
        result["pass"] = bool(result["g3a"]["pass"] and result["g3b"]["pass"])
    elif args.stage == "pk":
        out_path = args.out or (paths.MANIFESTS_DIR / "paso3d_f3_pk.json")
        m2 = json.loads((P2 / "manifest_pk.json").read_text("utf-8"))
        m21 = json.loads((P21 / "manifest_pk.json").read_text("utf-8"))
        result.update({
            "artifact": "pk", "pk_v2": str(P2), "pk_v2_1": str(P21),
            "counts_identical": m2["n_rows_by_source_split"] == m21["n_rows_by_source_split"],
            "n_rows_v2": m2["n_rows"], "n_rows_v2_1": m21["n_rows"],
            "n_partitions_v2": m2["n_partitions"], "n_partitions_v2_1": m21["n_partitions"],
            "pk_tokens_py_sha256_v2": m2["pk_tokens_py_sha256"],
            "pk_tokens_py_sha256_v2_1": m21["pk_tokens_py_sha256"],
        })
        print(f"  pk_v2: {m2['n_rows']} filas / {m2['n_partitions']} particiones | "
              f"pk_v2_1: {m21['n_rows']} / {m21['n_partitions']} | "
              f"conteos identicos: {result['counts_identical']}")
        print(f"  pk_tokens.py sha v2={m2['pk_tokens_py_sha256'][:16]}... "
              f"v2_1={m21['pk_tokens_py_sha256'][:16]}...")
        print("G3b — identidad de las filas no intervenidas (pk)")
        result["g3b"] = identity_gate(P2 / "windows", P21 / "windows",
                                       args.workers, 0.0, "G3b-pk")
        result["pass"] = bool(result["g3b"]["pass"] and result["counts_identical"])

    elif args.stage == "tokens":
        out_path = args.out or (paths.MANIFESTS_DIR / "paso3d_f3_tokens.json")
        m2 = json.loads((T2 / "manifest_tokens.json").read_text("utf-8"))
        m21 = json.loads((T21 / "manifest_tokens.json").read_text("utf-8"))
        result.update({
            "artifact": "tokens", "tokens_v2": str(T2), "tokens_v2_1": str(T21),
            "counts_identical": m2["n_rows_by_source_split"] == m21["n_rows_by_source_split"],
            "dense_counts_identical": (m2.get("n_dense_by_source_split")
                                       == m21.get("n_dense_by_source_split")),
            "discards_identical": (m2.get("discards_by_cause")
                                   == m21.get("discards_by_cause")),
            "pairs_identical": m2.get("pairs") == m21.get("pairs"),
            "n_rows_v2": m2["n_rows"], "n_rows_v2_1": m21["n_rows"],
            "counts": m21["n_rows_by_source_split"],
            "dense_counts": m21.get("n_dense_by_source_split"),
            "pairs": m21.get("pairs"),
            "context_dir": str(paths.CONTEXT_DIR),
            "context_reused": True,
        })
        print(f"  tokens_v2: {m2['n_rows']} filas | tokens_v2_1: {m21['n_rows']} | "
              f"conteos identicos: {result['counts_identical']} | "
              f"densos identicos: {result['dense_counts_identical']} | "
              f"pares identicos: {result['pairs_identical']}")
        print("G3b — identidad/afinidad de las filas no intervenidas (tokens)")
        result["g3b"] = identity_gate(T2 / "windows", T21 / "windows",
                                       args.workers, 0.0, "G3b-tokens",
                                       time_col="t1", affine=True)
        # G3c — el contexto (context_v2) se reutiliza; la verificación es
        # indirecta pero concluyente: si el vocabulario o los tokens de contexto
        # hubieran cambiado, TODAS las filas (incluidas las de real, las de
        # control y las del prefijo) habrian cambiado, y G3b lo detecta.
        result["g3c"] = {
            "contexto": str(paths.CONTEXT_DIR),
            "reutilizado": True,
            "verificacion": ("G3b: las filas no intervenidas (real/synthetic_v7/"
                             "vaso, control de CF y prefijo intervenido) son identicas; "
                             "un desplazamiento del vocabulario de contexto habria "
                             "cambiado todas ellas"),
            "no_intervenidas_identicas": bool(result["g3b"]["pass"]),
        }
        result["pass"] = bool(result["g3b"]["pass"] and result["counts_identical"]
                              and result["dense_counts_identical"])

    print(f"  G3b: {'PASA' if result['g3b']['pass'] else 'FALLA'}")
    print(f"    no-CF: {result['g3b']['non_cf']}")
    print(f"    CF: {result['g3b']['cf']}")
    result["elapsed_s"] = round(_time.time() - t0, 1)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"FASE 3 ({args.stage}): {'PASA' if result['pass'] else 'FALLA'} -> {out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
