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
import importlib
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
# Manifiesto de tokens del que se CONGELAN los estadísticos de normalización
# (Corrección A): sin esto, tokenize los acumula sobre el train del corpus y
# regenerar cf_v7_1 cambiaría los tokens de real.
STATS_FROM = paths.MANIFESTS_DIR / "tokens_v2_manifest.json"


def _require_cf() -> None:
    if not (CF_V7_1 / "metadata").is_dir():
        raise SystemExit(f"ERROR: falta {CF_V7_1}/metadata (¿generación de "
                         f"Fase 2 incompleta?)")


def _point_paths(windows: Path, pk: Path, tokens: Path) -> None:
    """Fija los DIRECTORIOS DE SALIDA por variable de entorno y recarga paths.

    Nada de monkeypatch ni de redirección de cohortes: la cohorte de CF la elige
    ``paths`` por CONSTANTE (``CF_COHORT_ACTIVE``, hoy cf_v7_1) y aquí solo se
    redirigen los directorios de artefacto, para no pisar los adoptados. El
    entorno se hereda entre procesos, así que los workers spawn de las ventanas
    también lo ven; y ``importlib.reload(paths)`` propaga los valores a todos los
    módulos que leen ``paths`` de forma diferida.
    """
    os.environ["ANESTESIA_WINDOWS_DIR"] = str(windows)
    os.environ["ANESTESIA_PK_DIR"] = str(pk)
    os.environ["ANESTESIA_TOKENS_DIR"] = str(tokens)
    importlib.reload(paths)    print(f"  paths: WINDOWS_DIR={paths.WINDOWS_DIR}")
    print(f"         PK_DIR={paths.PK_DIR}  TOKENS_DIR={paths.TOKENS_DIR}")
    print(f"         cohorte CF activa={paths.CF_COHORT_ACTIVE} "
          f"({paths.cohort_dir('cf_v7')})")


def stage_windows(workers: int) -> None:
    _require_cf()
    _point_paths(W41, P41, T41)
    import build_windows_v4 as bw  # scripts/ está en sys.path

    W41.mkdir(parents=True, exist_ok=True)
    print(f"windows_v4_1 -> {W41}  (etiqueta source=cf_v7 <- "
          f"{bw.SOURCES_V7['cf_v7']})")
    bw.generate_v3(max_workers=workers)


def stage_pk() -> None:
    _point_paths(W41, P41, T41)
    import tokens.pk_tokens as PK

    (P41 / "windows").mkdir(parents=True, exist_ok=True)
    print(f"pk_v2_1 -> {P41}  (ventanas = {PK.WINDOWS_DIR})")
    PK.main(["run", "--out-root", str(P41)])


def stage_tokens(ctx: str, stats_from: Path | None = None) -> None:
    _point_paths(W41, P41, T41)
    import tokens.cf_pairs as CP
    import tokens.tokenize as TK

    # ``ctx`` se conserva por compatibilidad de la interfaz: el contexto
    # (context_v2) se reutiliza y su directorio sale de ``paths.CONTEXT_DIR``,
    # redirigible con ANESTESIA_CONTEXT_DIR si alguna vez hay que rehacerlo.
    assert ctx in ("reuse", "rebuild")

    (T41 / "windows").mkdir(parents=True, exist_ok=True)
    print(f"tokens_v2_1 -> {T41}  (ventanas = {TK.WINDOWS_DIR}, pk = {TK.PK_DIR}, "
          f"casos crudos = {CP.CF_CASES_DIR}, contexto = {paths.CONTEXT_DIR})")
    TK.main(["run", "--pk-root", str(P41), "--out-root", str(T41),
             "--stats-from", str(stats_from or STATS_FROM)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["windows", "pk", "tokens", "all"],
                    default="all")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--ctx", choices=["reuse", "rebuild"], default="reuse")
    ap.add_argument("--stats-from", type=Path, default=None,
                    help=f"manifiesto con normalization_stats (por defecto {STATS_FROM.name})")
    args = ap.parse_args()
    if args.stage in ("windows", "all"):
        stage_windows(args.workers)
    if args.stage in ("pk", "all"):
        stage_pk()
    if args.stage in ("tokens", "all"):
        stage_tokens(args.ctx, args.stats_from)
    print("Fase 3: hecho")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
