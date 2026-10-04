"""PASO 3d Fase 0' — test del modelo de BT configurable (bt_start_model).

El código debe poder reproducir los datos v7 vigentes (``v7_normal``) y ofrecer
explícitamente la corrección pendiente (``uniform_c3_revert``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anessim.config import SimulatorConfig  # noqa: E402
from anessim.simulate import BT_START_MODELS, sample_bt_start  # noqa: E402

CONFIG_V7 = ROOT / "src" / "anessim" / "configs" / "synthetic_v7.yaml"


def test_default_model_is_v7_normal():
    assert SimulatorConfig().bt_start_model == "v7_normal"


def test_v7_normal_uses_normal_distribution_with_clip():
    got = sample_bt_start(np.random.default_rng(7), "v7_normal")
    exp = float(np.clip(np.random.default_rng(7).normal(36.0, 0.9), 33.5, 37.3))
    assert got == pytest.approx(exp, rel=0.0, abs=0.0)


def test_v7_normal_clip_bounds_are_respected():
    vals = np.array([sample_bt_start(np.random.default_rng(s), "v7_normal")
                     for s in range(400)])
    assert vals.min() >= 33.5
    assert vals.max() <= 37.3
    assert (vals == 37.3).any(), "el clip superior debe activarse en 400 sorteos"


def test_v7_normal_clip_lower_bound():
    for s in range(3000):
        raw = float(np.random.default_rng(s).normal(36.0, 0.9))
        if raw < 33.5:
            assert sample_bt_start(np.random.default_rng(s), "v7_normal") == 33.5
            return
    pytest.skip("ningún seed con normal < 33.5 en 3000 intentos")


def test_uniform_c3_revert_uses_uniform_distribution():
    got = sample_bt_start(np.random.default_rng(7), "uniform_c3_revert")
    exp = float(np.random.default_rng(7).uniform(36.5, 37.4))
    assert got == pytest.approx(exp, rel=0.0, abs=0.0)


def test_distributions_differ():
    a = np.array([sample_bt_start(np.random.default_rng(s), "v7_normal")
                  for s in range(200)])
    b = np.array([sample_bt_start(np.random.default_rng(s), "uniform_c3_revert")
                  for s in range(200)])
    assert a.mean() < b.mean() - 0.5   # 36.0 vs 36.95
    assert a.std() > b.std()           # 0.90 vs 0.26


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        sample_bt_start(np.random.default_rng(0), "nope")


def test_config_roundtrip_keeps_bt_start_model():
    d = SimulatorConfig().to_dict()
    assert d["bt_start_model"] == "v7_normal"
    assert SimulatorConfig.from_dict(d).bt_start_model == "v7_normal"


def test_synthetic_v7_yaml_sets_v7_normal():
    assert SimulatorConfig.from_yaml(CONFIG_V7).bt_start_model == "v7_normal"


def test_model_list_is_the_two_documented_models():
    assert set(BT_START_MODELS) == {"v7_normal", "uniform_c3_revert"}
