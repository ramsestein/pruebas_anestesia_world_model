"""Paso 3d / Fase 3 — regresión del fallo de propagación a los workers (spawn).

``build_windows_v4.generate_v3`` procesa los casos en un pool de procesos NUEVOS
(spawn en Windows). Los hijos re-importan el módulo y NO heredan ningún
monkeypatch hecho en el padre, así que el parche de ``SOURCES_V7`` que hacía el
orquestador de la Fase 3 no llegaba a los workers: leían ``paths.COHORTS["cf_v7"]``
(la cohorte VIEJA) y las ventanas de CF salían idénticas a windows_v4.

Arreglo: ``build_windows_v4`` resuelve ``SOURCES_V7`` y ``OUT_DIR_V7`` a partir de
``ANESTESIA_CF_V7_DIR`` y ``ANESTESIA_WINDOWS_OUT`` al IMPORTAR el módulo; el
entorno SÍ se hereda entre procesos. Estos tests comprueban (a) el
comportamiento por defecto intacto y (b) que un proceso nuevo —equivalente a un
worker spawn— ve la cohorte redirigida.

Nota: ``importlib.reload`` re-ejecuta el módulo EN SITIO y devuelve el mismo
objeto, así que la instantánea hay que tomarla dentro del contexto con las
variables puestas; leer el módulo después vería el estado restaurado.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths  # noqa: E402
import build_windows_v4 as bw  # noqa: E402

_VARS = ("ANESTESIA_CF_V7_DIR", "ANESTESIA_WINDOWS_OUT")


def _reload_snapshot(env: dict | None) -> dict:
    """Recarga build_windows_v4 con (o sin) las dos variables y devuelve una
    instantánea de lo que resolvió, dejando el módulo como estaba."""
    saved = {k: os.environ.get(k) for k in _VARS}
    try:
        for k in _VARS:
            os.environ.pop(k, None)
        if env:
            os.environ.update(env)
        m = importlib.reload(bw)
        snap = {
            "cf_v7": m.SOURCES_V7["cf_v7"],
            "out": m.OUT_DIR_V7,
            "real": m.SOURCES_V7["real"],
            "synthetic_v7": m.SOURCES_V7["synthetic_v7"],
            "vaso": m.SOURCES_V7["vaso_reinf_v7"],
        }
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(bw)
    return snap


def test_sin_variables_usa_los_valores_vigentes() -> None:
    snap = _reload_snapshot(None)
    assert snap["cf_v7"] == paths.COHORTS["cf_v7"]
    assert snap["out"] == paths.WINDOWS_DIR


def test_variables_de_entorno_redirigen_la_cohorte(tmp_path: Path) -> None:
    cf = tmp_path / "cf_x"
    out = tmp_path / "win_x"
    snap = _reload_snapshot({"ANESTESIA_CF_V7_DIR": str(cf),
                             "ANESTESIA_WINDOWS_OUT": str(out)})
    assert snap["cf_v7"] == cf
    assert snap["out"] == out
    # las otras cohortes no se tocan
    assert snap["real"] == paths.COHORTS["real"]
    assert snap["synthetic_v7"] == paths.COHORTS["synthetic_v7"]
    assert snap["vaso"] == paths.COHORTS["vaso_reinf_v7"]


def test_proceso_nuevo_ve_la_cohorte_redirigida(tmp_path: Path) -> None:
    """El caso real del fallo: un proceso NUEVO (worker spawn) debe ver la
    cohorte redirigida por el entorno.

    El hijo imprime sólo el NOMBRE del directorio: la consola de Windows usa
    cp1252 y no puede codificar «Ramsés», así que comparar rutas completas daría
    un falso negativo por la decodificación, no por el comportamiento.
    """
    cf = tmp_path / "cf_y"
    out = tmp_path / "win_y"
    env = {**os.environ,
           "PYTHONPATH": str(ROOT / "src"),
           "ANESTESIA_CF_V7_DIR": str(cf),
           "ANESTESIA_WINDOWS_OUT": str(out)}
    code = ("import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s');"
            " import build_windows_v4 as b;"
            " print(b.SOURCES_V7['cf_v7'].name); print(b.OUT_DIR_V7.name)"
            % (ROOT, ROOT / "scripts"))
    p = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, encoding="ascii",
                       errors="replace")
    assert p.returncode == 0, p.stderr
    lines = [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
    assert lines[-2] == cf.name
    assert lines[-1] == out.name
