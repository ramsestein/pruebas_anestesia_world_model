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


def _sort_key(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["caseid", "t"], kind="stable").reset_index(drop=True)


def _task_windows(task: dict) -> dict:
    rel = task["rel"]
    source = task["source"]
    a = _sort_key(_read_parquet(W4 / "windows" / rel))
    b = _sort_key(_read_parquet(W41 / "windows" / rel))
    res: dict = {"rel": rel, "source": source, "rows": int(len(a))}
    if len(a) != len(b):
        res["diffs"] = ["<filas>"]
        return res
    if source != "cf_v7":
        res["diffs"] = _diff_columns(a, b)
        return res

    split_by_a, control = _PAIRS if _PAIRS is not None else _pair_maps()
    cid = a["caseid"].to_numpy()
    s = pd.Series(cid).map(split_by_a)
    m_ctrl = s.isna().to_numpy()
    m_pref = (~s.isna()).to_numpy() & ((a["t"].to_numpy() + WIN_LEN_S)
                                       <= s.fillna(-1.0).to_numpy())
    m_post = (~s.isna()).to_numpy() & ~m_pref
    res["rows_control"] = int(m_ctrl.sum())
    res["rows_intervened_prefix"] = int(m_pref.sum())
    res["rows_intervened_post"] = int(m_post.sum())
    res["diffs_control"] = _diff_columns(a.loc[m_ctrl].reset_index(drop=True),
                                         b.loc[m_ctrl].reset_index(drop=True))
    res["diffs_prefix"] = _diff_columns(a.loc[m_pref].reset_index(drop=True),
                                        b.loc[m_pref].reset_index(drop=True))
    res["post_rows_differ"] = bool(
        _diff_columns(a.loc[m_post].reset_index(drop=True),
                      b.loc[m_post].reset_index(drop=True)))
    # Todos los casos de CF deben ser o bien un caso de control (caseid_b) o
    # bien un caso intervenido (caseid_a), y los dos conjuntos son disjuntos.
    unknown = set(np.unique(cid)) - set(split_by_a) - control
    res["unknown_caseids"] = int(len(unknown))
    res["n_cases"] = int(len(np.unique(cid)))
    return res


_PAIRS: dict | None = None


def _init_worker() -> None:
    """Carga los mapas de pares UNA vez por worker (spawn en Windows)."""
    global _PAIRS
    _PAIRS = _pair_maps()


def _task_missing(task: dict) -> dict:
    return {"rel": task["rel"], "source": task["source"],
            "diffs": ["<falta_particion>"]}


def g3b_windows(workers: int) -> dict:
    tasks: list[dict] = []
    for src in sorted((W4 / "windows").glob("source=*")):
        source = src.name.split("=", 1)[1]
        for part in sorted(src.glob("split=*/part-*.parquet")):
            rel = str(part.relative_to(W4 / "windows")).replace("\\", "/")
            tasks.append({"rel": rel, "source": source,
                          "missing": not (W41 / "windows" / rel).exists()})
    print(f"G3b: {len(tasks)} particiones")
    res: list[dict] = []
    t0 = _time.time()
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as ex:
        futs = [ex.submit(_task_missing if t["missing"] else _task_windows, t)
                for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                res.append(f.result())
            except Exception as exc:  # noqa: BLE001
                res.append({"rel": "?", "source": "?", "diffs": [f"<error {exc!r}>"]})
            if i % 50 == 0 or i == len(futs):
                print(f"  G3b: {i}/{len(futs)}  ({_time.time() - t0:.0f}s)", flush=True)

    non_cf = [r for r in res if r["source"] != "cf_v7"]
    cf = [r for r in res if r["source"] == "cf_v7"]
    bad_non_cf = [r for r in non_cf if r.get("diffs")]
    bad_ctrl = [r for r in cf if r.get("diffs") or r.get("diffs_control")]
    bad_pref = [r for r in cf if r.get("diffs_prefix")]
    out = {
        "n_partitions": len(res),
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
            "n_partitions_with_ctrl_diffs": len(bad_ctrl),
            "n_partitions_with_prefix_diffs": len(bad_pref),
            "n_partitions_post_differ": sum(1 for r in cf if r.get("post_rows_differ")),
            "unknown_caseids": int(sum(r.get("unknown_caseids", 0) for r in cf)),
            "n_cases": int(sum(r.get("n_cases", 0) for r in cf)),
            "examples_ctrl": [{"rel": r["rel"], "diffs": r.get("diffs") or r.get("diffs_control")}
                              for r in bad_ctrl[:5]],
            "examples_prefix": [{"rel": r["rel"], "diffs": r.get("diffs_prefix")}
                                for r in bad_pref[:5]],
        },
        "elapsed_s": round(_time.time() - t0, 1),
    }
    out["pass"] = bool(not bad_non_cf and not bad_ctrl and not bad_pref
                       and out["cf"]["unknown_caseids"] == 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["windows"], default="windows")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, default=paths.MANIFESTS_DIR / "paso3d_f3_windows.json")
    args = ap.parse_args()
    t0 = _time.time()
    result = {"date": datetime.now(timezone.utc).isoformat(),
              "windows_v4": str(W4), "windows_v4_1": str(W41),
              "window_py_sha256_v4": json.loads((W4 / "manifest.json").read_text("utf-8"))["window_py_sha256"],
              "window_py_sha256_v4_1": json.loads((W41 / "manifest.json").read_text("utf-8"))["window_py_sha256"]}
    print("G3a — split / registry / cases")
    result["g3a"] = g3a()
    print(f"  G3a: {'PASA' if result['g3a']['pass'] else 'FALLA'}  "
          f"{json.dumps(result['g3a']['files'], ensure_ascii=False)[:200]}")
    print("G3b — identidad de las filas no intervenidas")
    result["g3b"] = g3b_windows(args.workers)
    print(f"  G3b: {'PASA' if result['g3b']['pass'] else 'FALLA'}")
    print(f"    no-CF: {result['g3b']['non_cf']}")
    print(f"    CF: {result['g3b']['cf']}")
    result["pass"] = bool(result["g3a"]["pass"] and result["g3b"]["pass"])
    result["elapsed_s"] = round(_time.time() - t0, 1)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"FASE 3 (ventanas): {'PASA' if result['pass'] else 'FALLA'} -> {args.out}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
