"""Paso 3d / Corrección C — cohorte explícita y lecturas diferidas.

Dos bugs de la MISMA clase aparecieron en la Fase 3:

  1. los workers SPAWN de la construcción de ventanas re-importan el módulo y no
     heredan ningún monkeypatch del padre, así que leían ``paths.COHORTS["cf_v7"]``
     (la cohorte antigua) y las ventanas de CF salían idénticas a windows_v4;
  2. ``cf_pairs.CF_CASES_DIR`` capturaba la cohorte en el import, así que el
     tokenizador seguía leyendo los casos CRUDOS de la cohorte antigua y las
     features ``vent_*`` no veían la intervención.

Ambos se arreglan igual: nada de capturas en el import (lectura diferida) y la
cohorte de CF elegida por CONSTANTE en ``paths`` (``CF_COHORT_ACTIVE``), con los
directorios de artefacto redirigibles por variable de entorno, que sí se hereda
entre procesos.
"""
from __future__ import annotations

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
from tokens import cf_pairs as cp  # noqa: E402
from tokens import tokenize as tk  # noqa: E402


def test_no_hay_capturas_de_cohorte_en_el_import() -> None:
    """Los nombres históricos siguen accesibles, pero NO como variables de
    módulo: ese era el bug (una constante capturada en el import)."""
    for mod, names in ((cp, ("CF_CASES_DIR", "CF_META_DIR",
                             "PAIRS_PARQUET", "PAIRS_ANNOTATED_V2_PARQUET")),
                       (tk, ("CF_META_DIR",))):
        for n in names:
            assert n not in vars(mod), f"{mod.__name__}.{n} es una captura en import"


def test_cohorte_activa_por_defecto_es_cf_v7() -> None:
    """Hasta la adopción de la Fase 5 la cohorte activa es la ADOPTADA (cf_v7):
    si fuese cf_v7_1 mientras las ventanas/pk/tokens siguen siendo los de cf_v7,
    las rutinas que leen casos crudos vía ``cf_pairs`` mezclarían cohorte y
    anotaciones sin avisar."""
    assert paths.CF_COHORT_DEFAULT == "cf_v7"
    assert paths.CF_COHORT_ACTIVE == "cf_v7"
    assert paths.cohort_dir("cf_v7") == paths.COHORTS["cf_v7"]
    assert paths.COHORT_LABELS == {"cf_v7": paths.CF_COHORT_ACTIVE}


def test_cohorte_activa_se_fija_solo_durante_la_ejecucion(tmp_path: Path) -> None:
    """Con ANESTESIA_CF_COHORT puesta, la etiqueta lógica lleva a la cohorte
    nueva y el mapa lo registra; sin ella, a la adoptada."""
    code = ("import sys; sys.path.insert(0, r'%s'); import paths;"
            " print(paths.CF_COHORT_ACTIVE);"
            " print(paths.cohort_dir('cf_v7').name);"
            " print(paths.cohort_label_map()['cf_v7']['cohort'])"
            % (ROOT / "src"))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "ANESTESIA_CF_COHORT": "cf_v7_1"}
    pr = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                        capture_output=True, text=True, encoding="ascii",
                        errors="replace")
    assert pr.returncode == 0, pr.stderr
    lines = [ln.strip() for ln in (pr.stdout or "").splitlines() if ln.strip()]
    assert lines[-3:] == ["cf_v7_1", "cf_v7_1", "cf_v7_1"]
    # y una cohorte inexistente falla con mensaje claro en vez de pasar en silencio
    env2 = {**env, "ANESTESIA_CF_COHORT": "cf_que_no_existe"}
    pr2 = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env2,
                         capture_output=True, text=True, encoding="ascii",
                         errors="replace")
    assert pr2.returncode != 0
    assert "no es una cohorte conocida" in (pr2.stderr or "")


def test_lectura_diferida_tras_cambiar_la_cohorte(tmp_path: Path) -> None:
    """Un módulo YA IMPORTADO ve la cohorte nueva, tanto si cambia el mapa de
    etiquetas como el directorio de la cohorte activa."""
    saved_dir = paths.COHORTS[paths.CF_COHORT_ACTIVE]
    saved_label = paths.COHORT_LABELS["cf_v7"]
    try:
        paths.COHORT_LABELS["cf_v7"] = "cf_v7_1"
        paths.COHORTS["cf_v7_1"] = tmp_path
        assert paths.cohort_dir("cf_v7") == tmp_path
        assert cp.CF_CASES_DIR == tmp_path / "cases"
        assert cp.CF_META_DIR == tmp_path / "metadata"
        assert tk.cf_cases_dir() == tmp_path / "cases"
        assert tk.cf_meta_dir() == tmp_path / "metadata"
    finally:
        paths.COHORT_LABELS["cf_v7"] = saved_label
        paths.COHORTS["cf_v7_1"] = paths.CF_V7_1_DIR
        paths.COHORTS[paths.CF_COHORT_ACTIVE] = saved_dir
    assert cp.CF_CASES_DIR == saved_dir / "cases"      # vuelve al valor real


def test_dataset_sources_usa_la_etiqueta_logica() -> None:
    d = paths.dataset_sources()
    assert set(d) == {"real", "synthetic_v7", "vaso_reinf_v7", "cf_v7"}
    assert d["cf_v7"] == paths.COHORTS[paths.CF_COHORT_ACTIVE]
    assert "cf_v7_1" not in d               # la cohorte nueva NO es una fuente
    assert bw.SOURCES_V7 == paths.dataset_sources()
    assert paths.cohort_label_map() == {
        "cf_v7": {"cohort": paths.CF_COHORT_ACTIVE,
                  "dir": str(paths.COHORTS[paths.CF_COHORT_ACTIVE])}}


def test_proceso_nuevo_ve_los_directorios_redirigidos(tmp_path: Path) -> None:
    """El caso real del fallo de spawn: un proceso NUEVO ve lo que dice el
    entorno. Se comparan sólo los nombres de directorio: la consola de Windows
    usa cp1252 y la ruta lleva «Ramsés»."""
    w, p, t = tmp_path / "w_x", tmp_path / "p_x", tmp_path / "t_x"
    code = ("import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s');"
            " import paths;"
            " print(paths.WINDOWS_DIR.name); print(paths.PK_DIR.name);"
            " print(paths.TOKENS_DIR.name); print(paths.cohort_dir('cf_v7').name)"
            % (ROOT, ROOT / "scripts"))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "ANESTESIA_WINDOWS_DIR": str(w), "ANESTESIA_PK_DIR": str(p),
           "ANESTESIA_TOKENS_DIR": str(t), "ANESTESIA_CF_COHORT": "cf_v7_1"}
    pr = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                        capture_output=True, text=True, encoding="ascii",
                        errors="replace")
    assert pr.returncode == 0, pr.stderr
    lines = [ln.strip() for ln in (pr.stdout or "").splitlines() if ln.strip()]
    assert lines[-4:] == [w.name, p.name, t.name, "cf_v7_1"]
