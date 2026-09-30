"""Tests de src/paths.py (módulo único de rutas y cohortes)."""

from __future__ import annotations

import importlib

import paths


def test_vigent_dirs_exist_under_data_root():
    """Todas las rutas vigentes existen bajo DATA_ROOT."""
    for d in (paths.WINDOWS_DIR, paths.TOKENS_DIR, paths.PK_DIR,
              paths.CONTEXT_DIR, paths.AE_DIR, paths.DIAGNOSTICS_DIR,
              paths.AUDIT_DIR):
        assert d.is_dir(), f"{d} no existe (DATA_ROOT={paths.DATA_ROOT})"
    for name, d in paths.COHORTS.items():
        assert d.is_dir(), f"cohorte {name}: {d} no existe"


def test_cohorts_and_synth_lists_consistent():
    """COHORTS y SYNTH_COHORTS/ALL_COHORTS describen las mismas cohortes."""
    assert set(paths.COHORTS) == set(paths.ALL_COHORTS)
    assert paths.SYNTH_COHORTS == ["synthetic_v7", "vaso_reinf_v7", "cf_v7"]
    assert paths.ALL_COHORTS == ["real"] + paths.SYNTH_COHORTS
    assert set(paths.SYNTH_COHORTS) <= set(paths.COHORTS)


def test_lost_cohorts_are_not_vigent():
    """LOST_COHORTS no se solapa con las cohortes vigentes."""
    assert set(paths.LOST_COHORTS).isdisjoint(set(paths.COHORTS))
    assert "windows_v2" in paths.LOST_COHORTS
    assert "windows_v3" in paths.LOST_COHORTS


def test_env_overrides_data_root(tmp_path, monkeypatch):
    """ANESTESIA_DATA_ROOT sobrescribe DATA_ROOT (y las rutas derivadas)."""
    monkeypatch.setenv("ANESTESIA_DATA_ROOT", str(tmp_path))
    try:
        importlib.reload(paths)
        assert paths.DATA_ROOT == tmp_path
        assert paths.WINDOWS_DIR == tmp_path / "windows_v4"
        assert paths.COHORTS["cf_v7"] == tmp_path / "cf_v7"
        assert paths.COHORTS["vaso_reinf_v7"] == tmp_path / "synthetic_vaso_reinf_v7"
    finally:
        monkeypatch.delenv("ANESTESIA_DATA_ROOT", raising=False)
        importlib.reload(paths)


def test_default_data_root_is_repo_data():
    """Sin la variable de entorno, DATA_ROOT es <repo>/data."""
    assert paths.DATA_ROOT == paths.REPO_ROOT / "data"
