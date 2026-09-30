"""Fase B: regeneración de las cohortes v5 completas.

Uso:
  python -m anessim.scripts.fase_b [--base] [--vaso] [--cf] [--all]

Genera:
  - data/synthetic_v5/            10000 casos base (offset 40001)
  - data/synthetic_vaso_reinf_v5/ 500 casos vaso (offset 61001)
  - data/cf_v5/                   pares CF (pharma 250/70001, vent 150/80001,
                                  learning 375/90001; sin ephedrine duplicado:
                                  pharma ya incluye 'ephedrine' y learning 'eph_bolus')

Nota: efedrina ya está cubierta por pharma (palanca 'ephedrine', 120 pares) y por
learning ('eph_bolus'); generate_cf_ephedrine.py NO se ejecuta para no duplicar.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env={**__import__("os").environ, "PYTHONPATH": "src"})


def base() -> None:
    _run([PY, "-m", "anessim.cli", "--config", "src/anessim/configs/synthetic_v5.yaml",
          "--n-cases", "10000", "--case-offset", "40001", "--workers", "8"])


def vaso() -> None:
    _run([PY, "src/anessim/scripts/generate_vaso_reinforcement.py",
          "--cases", "500", "--workers", "8", "--start-caseid", "61001",
          "--output-dir", str(ROOT / "data" / "synthetic_vaso_reinf_v5")])


def cf() -> None:
    out = ROOT / "data" / "cf_v5"
    _run([PY, "-m", "anessim.scripts.generate_f2_cf", "pharma",
          "--n-pairs", "250", "--start-caseid", "70001",
          "--output-dir", str(out), "--workers", "8",
          "--config", "src/anessim/configs/synthetic_v5.yaml"])
    _run([PY, "-m", "anessim.scripts.generate_f2_cf", "vent",
          "--n-pairs", "150", "--start-caseid", "80001",
          "--output-dir", str(out), "--workers", "8",
          "--config", "src/anessim/configs/synthetic_v5.yaml"])
    _run([PY, "-m", "anessim.scripts.generate_f2_cf", "learning",
          "--n-pairs", "375", "--start-caseid", "90001",
          "--output-dir", str(out), "--workers", "8",
          "--config", "src/anessim/configs/synthetic_v5.yaml"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--vaso", action="store_true")
    ap.add_argument("--cf", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    do_all = args.all or not (args.base or args.vaso or args.cf)
    if do_all or args.base:
        base()
    if do_all or args.vaso:
        vaso()
    if do_all or args.cf:
        cf()


if __name__ == "__main__":
    main()
