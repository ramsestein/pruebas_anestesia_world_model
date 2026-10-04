"""generate_v7.py — regeneración de las cohortes v7 (atribución no lineal).

Uso:
  python scripts/generate_v7.py --base | --vaso | --cf | --all
  python scripts/generate_v7.py --smoke N   # cohorte base reducida para traza
  python scripts/generate_v7.py --cf --cf-out data/cf_v7_1

Genera (NO sobrescribe las v5 ni las v6):
  - data/synthetic_v7/            10000 casos base (offset 150001)
  - data/synthetic_vaso_reinf_v7/ 500 casos vaso (offset 160001)
  - data/cf_v7/                   pares CF (pharma 870, vent 600, learning 4875)

Paso 3d (Fase 2): ``--cf-out`` permite escribir la cohorte CF en un directorio
nuevo (por defecto ``data/cf_v7_1``) y escribir además ``manifest_cohorte.json``
con el commit de git y los sha256 del núcleo del generador (el hueco de
procedencia de cf_v7, ver provenance_gaps.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": "src"}

# Núcleo del generador cuyo sha256 queda registrado en el manifiesto.
CORE_FILES = [
    "src/anessim/scripts/generate_f2_cf.py",
    "src/anessim/simulate.py",
    "src/anessim/sensors.py",
    "src/anessim/respiratory.py",
    "src/anessim/render.py",
    "src/anessim/configs/synthetic_v7.yaml",
]
DEFAULT_CF_OUT = ROOT / "data" / "cf_v7_1"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=ENV)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return (p.stdout or "").strip()


def _disk_free_gb(path: Path) -> float:
    import shutil
    target = path if path.exists() else path.parent
    return round(shutil.disk_usage(str(target)).free / (1024 ** 3), 1)


def _write_cohort_manifest(out: Path) -> None:
    base_count = 10000
    vaso_count = 500
    cf_counts = {"pharma": 870, "vent": 600, "learning": 4875}
    manifest = {
        "artifact": "cf cohort",
        "output_dir": str(out),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "core_sha256": {f: _sha256(ROOT / f) for f in CORE_FILES},
        "n_pairs": {"pharma": cf_counts["pharma"], "vent": cf_counts["vent"],
                    "learning": cf_counts["learning"],
                    "total": sum(cf_counts.values())},
        "base_cases": base_count,
        "vaso_cases": vaso_count,
        "generator": "scripts/generate_v7.py --cf",
        "notes": ("Paso 3d Fase 2: cohorte cf_v7_1 (muestreo v7.1 de las palancas "
                  "de consigna y bt_start_model='v7_normal'). Este manifiesto "
                  "cierra hacia adelante el hueco de procedencia de cf_v7."),
    }
    (out / "manifest_cohorte.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifiesto de cohorte escrito en {out / 'manifest_cohorte.json'}")


def base(n_cases: int = 10000) -> None:
    _run([PY, "-m", "anessim.cli", "--config",
          "src/anessim/configs/synthetic_v7.yaml",
          "--n-cases", str(n_cases), "--case-offset", "150001", "--workers", "8"])


def vaso() -> None:
    _run([PY, "src/anessim/scripts/generate_vaso_reinforcement.py",
          "--cases", "500", "--workers", "8", "--start-caseid", "160001",
          "--output-dir", str(ROOT / "data" / "synthetic_vaso_reinf_v7")])


def cf(out: Path | None = None, workers: int = 8) -> None:
    out = Path(out) if out is not None else DEFAULT_CF_OUT
    print(f"espacio libre en disco antes de generar: {_disk_free_gb(out)} GiB")
    _run([PY, "-m", "anessim.scripts.generate_f2_cf", "pharma",
          "--n-pairs", "250", "--start-caseid", "170001",
          "--output-dir", str(out), "--workers", str(workers),
          "--config", "src/anessim/configs/synthetic_v7.yaml"])
    _run([PY, "-m", "anessim.scripts.generate_f2_cf", "vent",
          "--n-pairs", "150", "--start-caseid", "180001",
          "--output-dir", str(out), "--workers", str(workers),
          "--config", "src/anessim/configs/synthetic_v7.yaml"])
    _run([PY, "-m", "anessim.scripts.generate_f2_cf", "learning",
          "--n-pairs", "375", "--start-caseid", "190001",
          "--output-dir", str(out), "--workers", str(workers),
          "--config", "src/anessim/configs/synthetic_v7.yaml"])
    _write_cohort_manifest(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--vaso", action="store_true")
    ap.add_argument("--cf", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cf-out", type=Path, default=None,
                    help="directorio de salida de la cohorte CF (paso 3d)")
    ap.add_argument("--workers", type=int, default=8,
                    help="workers por colección en --cf")
    ap.add_argument("--smoke", type=int, default=0,
                    help="genera solo N casos de la base (traza preliminar)")
    args = ap.parse_args()
    if args.smoke:
        base(args.smoke)
        return
    do_all = args.all or not (args.base or args.vaso or args.cf)
    if do_all or args.base:
        base()
    if do_all or args.vaso:
        vaso()
    if do_all or args.cf:
        cf(args.cf_out, args.workers)


if __name__ == "__main__":
    main()
