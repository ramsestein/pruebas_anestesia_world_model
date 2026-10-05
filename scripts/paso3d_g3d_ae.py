"""Paso 3d / G3d — no regresión del autoencoder (ae_v2 CONGELADO, k_def = 12).

Compara el MAE de reconstrucción —enmascarado, por variable y en el espacio
normalizado— de las ventanas de las RAMAS INTERVENIDAS de ``set_rr``, ``set_tv``
y ``set_peep`` en ``cf_v7_1`` frente a las MISMAS ventanas en ``cf_v7``. Es la
misma definición de error que usa ``physio_ae.evaluate``: entradas
``concat(valores_normalizados, máscaras)``, ``model(x, k=12)`` y
``|recon - valor| * máscara`` promediado por variable.

Criterio BLOQUEANTE: cociente ``nuevo/viejo <= 1.25`` en
``Solar8000/HR``, ``Solar8000/ART_MBP``, ``BIS/BIS`` y ``Primus/ETCO2``. El resto
de variables se informa.

Se ejecuta dos veces (una por cohorte de ventanas) porque ``physio_ae`` captura
``paths.WINDOWS_DIR`` al importar, y luego compara:

  python scripts/paso3d_g3d_ae.py --windows-dir data/windows_v4   --out reports/_g3d_v4.json
  python scripts/paso3d_g3d_ae.py --windows-dir data/windows_v4_1 --out reports/_g3d_v41.json
  python scripts/paso3d_g3d_ae.py --compare reports/_g3d_v4.json reports/_g3d_v41.json \\
      --out manifests/paso3d_g3d.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time as _time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths  # noqa: E402

CF_V7_1 = ROOT / "data" / "cf_v7_1"
AE_V2_DIR = ROOT / "data" / "ae_v2"
MODIFIED_LEVERS = ("set_rr", "set_tv", "set_peep")

# Variables clínicas cuyo cociente de MAE es bloqueante.
BLOCKING_VARS = ("Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS", "Primus/ETCO2")
RATIO_MAX = 1.25
BATCH = 65536


def intervened_caseids() -> np.ndarray:
    """caseid_a (rama INTERVENIDA) de los pares de set_rr/set_tv/set_peep."""
    out: list[int] = []
    for p in (CF_V7_1 / "metadata").glob("cf_pair_*.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        if m["lever"] in MODIFIED_LEVERS:
            out.append(int(m["caseid_a"]))
    return np.array(sorted(out), dtype=np.int64)


def measure(windows_dir: Path) -> dict:
    t0 = _time.time()
    # ``physio_ae`` captura paths.WINDOWS_DIR al importar: hay que fijarlo antes.
    os.environ["ANESTESIA_WINDOWS_DIR"] = str(windows_dir)
    importlib.reload(paths)
    import ae.physio_ae as pa
    importlib.reload(pa)
    import torch

    k_def = int(json.loads((ROOT / "manifests" / "ae_v2_manifest.json")
                           .read_text(encoding="utf-8"))["k_def_final"])
    assert k_def == 12, f"k_def inesperado: {k_def}"
    device = torch.device("cpu")
    model = pa.make_model(pa.SEED)
    model.encoder.load_state_dict(torch.load(AE_V2_DIR / "encoder.pt",
                                             map_location="cpu"))
    model.decoder.load_state_dict(torch.load(AE_V2_DIR / "decoder.pt",
                                             map_location="cpu"))
    model = model.to(device).eval()
    stats = json.loads((AE_V2_DIR / "norm_stats.json").read_text(encoding="utf-8"))
    excluded = pa.excluded_caseids()
    wanted = intervened_caseids()

    vs: list[np.ndarray] = []
    ms: list[np.ndarray] = []
    for split in ("train", "val"):
        val = pa.load_cells(["cf_v7"], split, excluded)
        sel = np.isin(val["caseid"], wanted)
        print(f"  split={split}: {int(sel.sum())} celdas de casos intervenidos "
              f"(de {len(val['caseid'])})", flush=True)
        if sel.any():
            vs.append(val["values"][sel])
            ms.append(val["masks"][sel])
    if not vs:
        raise SystemExit("no se han encontrado celdas de casos intervenidos")
    values = np.concatenate(vs, axis=0)
    masks = np.concatenate(ms, axis=0)
    vn = pa.normalize(values, stats, pa.IMAGE_TRACKS).astype(np.float32)
    # Misma convención que ``_normalize_train_data``: las celdas enmascaradas
    # valen 0 (los valores crudos traen NaN donde la máscara es 0 y ``NaN * 0``
    # seguiría siendo NaN en el error).
    vn[masks == 0] = 0.0

    n = len(vn)
    acc = np.zeros(pa.N_VARS, dtype=np.float64)
    cnt = np.zeros(pa.N_VARS, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n, BATCH):
            idx = np.arange(i, min(i + BATCH, n))
            x = pa._make_input(vn, masks, idx, device)
            recon = model(x, k=k_def)[:, :pa.N_VARS].cpu().numpy()
            err = np.abs(recon - vn[idx]) * masks[idx].astype(np.float64)
            acc += err.sum(axis=0)
            cnt += masks[idx].sum(axis=0)
    mae = acc / np.maximum(cnt, 1.0)
    out = {
        "windows_dir": str(windows_dir),
        "k_def": k_def,
        "n_cells": int(n),
        "n_intervened_cases": int(len(wanted)),
        "n_obs_by_var": {v: int(c) for v, c in zip(pa.IMAGE_TRACKS, cnt)},
        "mae_by_var": {v: float(m) for v, m in zip(pa.IMAGE_TRACKS, mae)},
        "elapsed_s": round(_time.time() - t0, 1),
    }
    print(f"  {n} celdas, k={k_def}, {out['elapsed_s']}s", flush=True)
    return out


def compare(old: dict, new: dict) -> dict:
    ratios = {v: (new["mae_by_var"][v] / old["mae_by_var"][v]
                  if old["mae_by_var"][v] > 0 else float("inf"))
              for v in old["mae_by_var"]}
    blocking = {v: ratios[v] for v in BLOCKING_VARS}
    result = {
        "old": {"windows_dir": old["windows_dir"], "n_cells": old["n_cells"],
                "mae_by_var": old["mae_by_var"]},
        "new": {"windows_dir": new["windows_dir"], "n_cells": new["n_cells"],
                "mae_by_var": new["mae_by_var"]},
        "ratio_new_over_old": ratios,
        "blocking_vars": list(BLOCKING_VARS),
        "ratio_max": RATIO_MAX,
        "n_obs_by_var": new["n_obs_by_var"],
        "k_def": new["k_def"],
        "pass": bool(all(r <= RATIO_MAX for r in blocking.values())),
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", nargs=2, type=Path, default=None,
                    metavar=("OLD_JSON", "NEW_JSON"))
    args = ap.parse_args()

    if args.compare:
        old = json.loads(args.compare[0].read_text(encoding="utf-8"))
        new = json.loads(args.compare[1].read_text(encoding="utf-8"))
        res = compare(old, new)
        out = args.out or (paths.MANIFESTS_DIR / "paso3d_g3d.json")
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"{'variable':28s} {'viejo':>10s} {'nuevo':>10s} {'cociente':>9s}")
        for v in res["ratio_new_over_old"]:
            mark = " *" if v in BLOCKING_VARS else ""
            print(f"{v:28s} {old['mae_by_var'][v]:10.5f} {new['mae_by_var'][v]:10.5f} "
                  f"{res['ratio_new_over_old'][v]:9.4f}{mark}")
        print(f"\nBLOQUEANTES (cociente <= {RATIO_MAX}): "
              f"{ {v: round(res['ratio_new_over_old'][v], 4) for v in BLOCKING_VARS} }")
        print(f"G3d: {'PASA' if res['pass'] else 'FALLA'} -> {out}")
        return 0 if res["pass"] else 1

    if args.windows_dir is None:
        raise SystemExit("falta --windows-dir (o --compare)")
    res = measure(args.windows_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"escrito {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
