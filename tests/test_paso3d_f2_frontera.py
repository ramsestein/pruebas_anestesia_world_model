"""Paso 3d / Fase 2 — regresión de la regla de FRONTERA de G2c y de G2d.

Contexto (ver ``scripts/paso3d_f2_gates.py``): el override contrafactual está
activo desde ``split_t``, así que el prefijo pre-intervención es ``t < split_t``
(estricto). Si se compara con ``<=`` y el ``time`` del sidecar ``truth`` es
float32, el redondeo del escalar float64 ``split_t`` a float32 puede colar la
primera fila post-intervención (caso real: par 198303 con
``split_t = 9500.49995589281`` redondeado a 9500.5).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from paso3d_f2_gates import _cmp_parquet, _delta_on_grid, _read_parquet  # noqa: E402

# split_t real del único par que falló G2c (redondea a 9500.5 en float32).
SPLIT_T = 9500.49995589281


def _write_truth(dir_: Path, cid: int, times: np.ndarray, peep: np.ndarray,
                 float32_time: bool = True) -> None:
    (dir_ / "truth").mkdir(parents=True, exist_ok=True)
    t_dtype = pa.float32() if float32_time else pa.float64()
    tbl = pa.table({
        "time": pa.array(times.astype(np.float32 if float32_time else np.float64),
                         type=t_dtype),
        "peep_applied": pa.array(peep.astype(np.float32), type=pa.float32()),
    })
    pq.write_table(tbl, dir_ / "truth" / f"{cid:04d}_truth.parquet")


def test_frontera_excluye_la_primera_fila_post_intervencion(tmp_path: Path) -> None:
    """La fila en el primer instante >= split_t NO pertenece al prefijo."""
    times = np.array([9499.5, 9500.0, 9500.5, 9501.0])
    a = tmp_path / "a"
    b = tmp_path / "b"
    # A: PEEP antiguo hasta 9500.0; B: ya aplicado en 9500.5.
    _write_truth(a, 1, times, np.array([2.0, 2.0, 1.888, 1.888]))
    _write_truth(b, 1, times, np.array([2.0, 2.0, 0.0, 0.0]))

    assert _cmp_parquet(a, b, "truth", 1, prefix_t=SPLIT_T) == []
    # ... y sin embargo difieren en la cohorte completa.
    assert _cmp_parquet(a, b, "truth", 1) == ["peep_applied"]


def test_frontera_detecta_diferencias_anteriores_al_split(tmp_path: Path) -> None:
    """El prefijo no es vacío ni trivialmente igual: una diferencia previa al
    split SÍ se reporta."""
    times = np.array([9000.0, 9499.5, 9500.0, 9500.5])
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_truth(a, 2, times, np.array([2.0, 2.0, 1.7, 1.7]))
    _write_truth(b, 2, times, np.array([2.0, 2.0, 2.0, 1.0]))

    assert _cmp_parquet(a, b, "truth", 2, prefix_t=SPLIT_T) == ["peep_applied"]


def test_frontera_float32_vs_float64_mismo_resultado(tmp_path: Path) -> None:
    """El resultado no depende del tipo de la columna ``time``."""
    times = np.array([9500.0, 9500.5])
    peep = np.array([2.0, 0.0])
    a32, b32 = tmp_path / "a32", tmp_path / "b32"
    a64, b64 = tmp_path / "a64", tmp_path / "b64"
    _write_truth(a32, 3, times, peep, float32_time=True)
    _write_truth(b32, 3, times, peep, float32_time=True)
    _write_truth(a64, 3, times, peep, float32_time=False)
    _write_truth(b64, 3, times, peep, float32_time=False)

    r32 = _cmp_parquet(a32, b32, "truth", 3, prefix_t=SPLIT_T)
    r64 = _cmp_parquet(a64, b64, "truth", 3, prefix_t=SPLIT_T)
    assert r32 == r64 == []
    # el prefijo conserva la fila previa (no queda vacío)
    df = _read_parquet(a32 / "truth" / "0003_truth.parquet")
    t = df["time"].to_numpy(dtype=np.float64)
    assert int((t < SPLIT_T).sum()) == 1


def test_frontera_time_float32_realmente_redondea(tmp_path: Path) -> None:
    """Verifica la premisa del artefacto: float32(split_t) == 9500.5."""
    assert np.float32(SPLIT_T) == np.float32(9500.5)


# ---------------------------------------------------------------------------
# G2d — δ en la rejilla de registro
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lever,key,value,k", [
    ("set_rr", "rr_delta", 3.0, 3),
    ("set_rr", "rr_delta", -4.0, -4),
    ("set_tv", "tv_delta", 30.0, 3),
    ("set_tv", "tv_delta", -150.0, -15),
    ("set_peep", "peep_delta", 5.0, 5),
    ("set_peep", "peep_delta", -1.0, -1),
])
def test_g2d_acepta_multiplos_de_la_rejilla(lever: str, key: str, value: float,
                                            k: int) -> None:
    ok, kk, why = _delta_on_grid({"lever": lever, "vent_override_a": {key: value}})
    assert ok and why == "ok" and int(round(kk)) == k


@pytest.mark.parametrize("lever,key,value,why", [
    ("set_rr", "rr_delta", 2.5, "sub_escalon"),      # 0.5 rpm
    ("set_tv", "tv_delta", 5.0, "sub_escalon"),      # 0.5 escalones de 10 mL
    ("set_peep", "peep_delta", 0.0, "delta_nulo"),
    ("set_tv", "tv_delta", 200.0, "fuera_de_rango"),
    ("set_peep", "peep_delta", 9.0, "fuera_de_rango"),
    ("set_rr", "otra_clave", 3.0, "sin_delta"),
    ("no_existe", "rr_delta", 3.0, "palanca_desconocida"),
])
def test_g2d_rechaza_lo_que_no_esta_en_la_rejilla(lever: str, key: str,
                                                  value: float, why: str) -> None:
    ok, _, motivo = _delta_on_grid({"lever": lever, "vent_override_a": {key: value}})
    assert not ok and motivo == why


def test_g2d_lee_el_delta_de_la_rama_intervenida() -> None:
    """El δ se lee de ``vent_override_a`` (rama intervenida), no de la de
    control."""
    meta = {"lever": "set_peep", "vent_override_a": {"peep_delta": 3.0},
            "vent_override_b": None}
    assert _delta_on_grid(meta)[0]
    meta2 = {"lever": "set_peep", "vent_override_a": None,
             "vent_override_b": {"peep_delta": 3.0}}
    assert not _delta_on_grid(meta2)[0]
