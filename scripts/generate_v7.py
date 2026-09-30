"""generate_v7.py — regeneración de las cohortes v7 (atribución no lineal).

Uso:
  python scripts/generate_v7.py --base | --vaso | --cf | --all
  python scripts/generate_v7.py --smoke N   # cohorte base reducida para traza

Genera (NO sobrescribe las v5 ni las v6):
  - data/synthetic_v7/            10000 casos base (offset 150001)
  - data/synthetic_vaso_reinf_v7/ 500 casos vaso (offset 160001)
  - data/cf_v7/                   pares CF (pharma 870, vent 600, learning 4875)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": "src"}


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=ENV)


def base(n_cases: int = 10000) -> None:
    _run([PY, "-m", "anessim.cli", "--config",
          "src/anessim/configs/synthetic_v7.yaml",
          "--n-cases", str(n_cases), "--case-offset", "150001", "--workers", "8"])


def vaso() -> None:
    _run([PY, "src/anessim/scripts/generate_vaso_reinforcement.py",
          "--cases", "500", "--workers", "8", "--start-caseid", "160001",
          "--output-dir", str(ROOT / "data" / "synthetic_vaso_reinf_v7")])


def cf() -> None:
    out = ROOT / "data" / "cf_v7"
    _run([PY, "-m", "anessim.scripts.generate_f2_cf", "pharma",
          "--n-pairs", "250", "--start-caseid", "170001",
          "--output-dir", str(out), "--workers", "8",
          "--config", "src/anessim/configs/synthetic_v7.yaml"])
    _run([PY, "-m", "anessim.scripts.generate_f2_cf", "vent",
          "--n-pairs", "150", "--start-caseid", "180001",
          "--output-dir", str(out), "--workers", "8",
          "--config", "src/anessim/configs/synthetic_v7.yaml"])
    _run([PY, "-m", "anessim.scripts.generate_f2_cf", "learning",
          "--n-pairs", "375", "--start-caseid", "190001",
          "--output-dir", str(out), "--workers", "8",
          "--config", "src/anessim/configs/synthetic_v7.yaml"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--vaso", action="store_true")
    ap.add_argument("--cf", action="store_true")
    ap.add_argument("--all", action="store_true")
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
        cf()


if __name__ == "__main__":
    main()
