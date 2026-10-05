"""Paso 3d / Fase 3 — aguas abajo de cf_v7_1.

Genera, reutilizando el código existente SIN modificarlo (solo se re-apuntan sus
constantes de módulo, igual que hace ``scripts/build_windows_v4.py``):

  windows_v4_1   <- build_windows_v4.generate_v3() con cf_v7 -> data/cf_v7_1
  pk_v2_1        <- tokens.pk_tokens.run_all() sobre windows_v4_1
  tokens_v2_1    <- tokens.tokenize.run_all() sobre pk_v2_1 + context_v2

Regla de nombres. Al construir ``windows_v4_1`` la cohorte canónica ``cf_v7`` se
APUNTA a ``data/cf_v7_1``, de modo que las etiquetas de fuente, el split y las
particiones son idénticas a ``windows_v4`` y las comparaciones de G3a/G3b son
directas (mismo ``source``, misma ruta relativa). El directorio histórico
``data/cf_v7`` NO se toca.

El contexto (``context_v2``) se REUTILIZA por defecto: G3c comprueba si los
tokens de contexto cambian; si cambiasen, hay que rehacerlo (``--ctx rebuild``).

Uso:
  python scripts/paso3d_f3_downstream.py --stage windows --workers 14
  python scripts/paso3d_f3_downstream.py --stage pk
  python scripts/paso3d_f3_downstream.py --stage tokens
  python scripts/paso3d_f3_downstream.py --stage all --workers 14
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths  # noqa: E402

CF_V7_1 = ROOT / "data" / "cf_v7_1"
W41 = ROOT / "data" / "windows_v4_1"
P41 = ROOT / "data" / "pk_v2_1"
T41 = ROOT / "data" / "tokens_v2_1"


def _require_cf() -> None:
    if not (CF_V7_1 / "metadata").is_dir():
        raise SystemExit(f"ERROR: falta {CF_V7_1}/metadata (¿generación de "
                         f"Fase 2 incompleta?)")


def stage_windows(workers: int) -> None:
    _require_cf()
    # Las variables de entorno son la vía que SÍ llega a los workers: con spawn
    # (Windows) los hijos re-importan build_windows_v4 y no heredan el
    # monkeypatch del padre. Sin esto, los hijos leen cf_v7 en vez de cf_v7_1 y
    # las ventanas de CF salen idénticas a windows_v4.
    os.environ["ANESTESIA_CF_V7_DIR"] = str(CF_V7_1)
    os.environ["ANESTESIA_WINDOWS_OUT"] = str(W41)
    import build_windows_v4 as bw  # scripts/ está en sys.path

    srcs = dict(paths.COHORTS)
    srcs["cf_v7"] = CF_V7_1          # única diferencia con windows_v4
    bw.SOURCES_V7 = srcs
    bw.OUT_DIR_V7 = W41
    W41.mkdir(parents=True, exist_ok=True)
    print(f"windows_v4_1 -> {W41}  (cf_v7 := {CF_V7_1})")
    print(f"  comprobación: SOURCES_V7['cf_v7'] = {bw.SOURCES_V7['cf_v7']}")
    print(f"                OUT_DIR_V7           = {bw.OUT_DIR_V7}")
    bw.generate_v3(max_workers=workers)


def stage_pk() -> None:
    import tokens.pk_tokens as PK

    paths.WINDOWS_DIR = W41
    paths.PK_DIR = P41
    PK.WINDOWS_DIR = W41 / "windows"
    PK.OUT_DIR = P41 / "windows"
    PK.OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"pk_v2_1 -> {P41}  (windows_v4_1 = {W41})")
    PK.main(["run", "--out-root", str(P41)])


def stage_tokens(ctx: str) -> None:
    import tokens.cf_pairs as CP
    import tokens.tokenize as TK

    paths.WINDOWS_DIR = W41
    paths.PK_DIR = P41
    paths.TOKENS_DIR = T41
    # CLAVE: las features vent_* (consignas de ventilación) NO salen de las
    # ventanas sino del parquet de CASO CRUDO, vía ``cf_pairs.CF_CASES_DIR``,
    # que se captura en el import desde ``paths.COHORTS["cf_v7"]``. Sin esta
    # redirección el tokenizador lee los setpoints de la cohorte vieja y las
    # filas de CF salen idénticas a tokens_v2 (lo detectó G3b: 0 de 163
    # particiones con diferencias tras el split, cuando en las ventanas sí
    # difieren 30 de 163).
    paths.COHORTS["cf_v7"] = CF_V7_1
    CP.CF_CASES_DIR = CF_V7_1 / "cases"
    TK.WINDOWS_ROOT = W41
    TK.WINDOWS_DIR = W41 / "windows"
    if ctx == "rebuild":
        TK.CTX_TOKENS = paths.CONTEXT_DIR / "tokens.parquet"
        TK.CTX_VOCAB = paths.CONTEXT_DIR / "vocab.json"
    (T41 / "windows").mkdir(parents=True, exist_ok=True)
    print(f"tokens_v2_1 -> {T41}  (pk_v2_1 = {P41}, windows = {W41}, "
          f"casos crudos = {CP.CF_CASES_DIR}, contexto = {paths.CONTEXT_DIR})")
    # ``main`` declara global PK_DIR/CTX_*/OUT_DIR/OUT_ROOT/MANIFEST_PATH y solo
    # las reescribe si se pasan por CLI; pasarlas es más explícito que confiar
    # en el re-apuntado previo. El directorio de ventanas no tiene opción de CLI
    # y va por el global parcheado (tokenize no usa multiprocessing).
    TK.main(["run", "--pk-root", str(P41), "--out-root", str(T41)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["windows", "pk", "tokens", "all"],
                    default="all")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--ctx", choices=["reuse", "rebuild"], default="reuse")
    args = ap.parse_args()
    if args.stage in ("windows", "all"):
        stage_windows(args.workers)
    if args.stage in ("pk", "all"):
        stage_pk()
    if args.stage in ("tokens", "all"):
        stage_tokens(args.ctx)
    print("Fase 3: hecho")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
