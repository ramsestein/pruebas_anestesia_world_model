"""PASO 3d Fase 0' — test del comparador nulo-seguro (_diff_columns/frames_equal).

Regresión del artefacto ``truth/phase``: el comparador anterior usaba
``np.array_equal`` sobre columnas object, donde ``nan != nan``, y por eso una
columna con un nulo en la MISMA posición en ambos lados aparecía como
diferencia.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "paso3d_f0_diag", ROOT / "scripts" / "paso3d_f0_diagnostico.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


def test_object_column_with_null_same_position_is_not_a_diff():
    a = pd.DataFrame({"phase": ["maintenance", None, "emergence"],
                      "x": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"phase": ["maintenance", None, "emergence"],
                      "x": [1.0, 2.0, 3.0]})
    assert M._diff_columns(a, b) == []
    assert M.frames_equal(a, b)


def test_object_column_null_vs_value_is_a_diff():
    a = pd.DataFrame({"phase": ["maintenance", None]})
    b = pd.DataFrame({"phase": ["maintenance", "emergence"]})
    assert M._diff_columns(a, b) == ["phase"]


def test_float_nan_same_position_is_not_a_diff():
    a = pd.DataFrame({"v": [1.0, np.nan, 3.0]})
    b = pd.DataFrame({"v": [1.0, np.nan, 3.0]})
    assert M._diff_columns(a, b) == []


def test_float_nan_vs_value_is_a_diff():
    a = pd.DataFrame({"v": [1.0, np.nan]})
    b = pd.DataFrame({"v": [1.0, 2.0]})
    assert M._diff_columns(a, b) == ["v"]


def test_categorical_column_nulls_same_position():
    a = pd.DataFrame({"phase": pd.Categorical(["a", None, "b"])})
    b = pd.DataFrame({"phase": pd.Categorical(["a", None, "b"])})
    assert M._diff_columns(a, b) == []


def test_column_order_differs():
    a = pd.DataFrame({"x": [1], "y": [2]})
    b = pd.DataFrame({"y": [2], "x": [1]})
    assert M._diff_columns(a, b) == ["<columnas>"]


def test_row_count_differs():
    a = pd.DataFrame({"x": [1, 2]})
    b = pd.DataFrame({"x": [1]})
    assert M._diff_columns(a, b) == ["<filas>"]


def test_numeric_value_diff_is_reported():
    a = pd.DataFrame({"v": [1.0, 2.0]})
    b = pd.DataFrame({"v": [1.0, 2.5]})
    assert M._diff_columns(a, b) == ["v"]
