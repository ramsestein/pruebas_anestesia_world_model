"""Tests de validación de la reconstrucción de v6_validate.py (PARTE 1.3).

Verifica que la reconstrucción (desde .pyc) reproduce numéricamente los valores
de V1, V2, V3 y V4 registrados en data/diagnostics/v7_validate_results.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diagnostics import v6_validate as vv  # noqa: E402
from diagnostics import cohort_gap as cg  # noqa: E402

CACHE_PATH = Path("data/diagnostics/v7_validate_results.json")
CHECK_PATH = Path("data/diagnostics/v9_reconstruction_check.json")


def _holdout_sha() -> tuple[str, int, int]:
    excluded = cg.load_excluded_caseids()
    real_ids = []
    import pyarrow.parquet as pq
    d = Path("data/windows_v4/windows/source=real/split=val")
    for part in sorted(d.glob("part-*.parquet")):
        df = pq.read_table(part, columns=["caseid", "phase_from_clinical"]).to_pandas()
        df = cg.filter_cells(df, excluded)
        real_ids.extend(int(c) for c in df["caseid"].unique())
    real_ids = sorted(set(real_ids))
    rng = np.random.default_rng(20260922)
    perm = rng.permutation(real_ids)
    k = int(round(len(perm) * 0.40))
    holdout = sorted(int(c) for c in perm[:k])
    calib = sorted(int(c) for c in perm[k:])
    sha = hashlib.sha256(",".join(str(c) for c in holdout).encode()).hexdigest()
    return sha, len(holdout), len(calib)


def test_a_split_holdout_sha_matches_cache():
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    sha, n_hold, n_cal = _holdout_sha()
    assert n_hold == cache["meta"]["n_holdout_caseids"]
    assert n_cal == cache["meta"]["n_calib_caseids"]
    assert sha == cache["meta"]["holdout_caseids_sha256"]


def test_b_reconstruction_numeric_check_all_match():
    # El re-cómputo completo (V1-V4) se guarda en v9_reconstruction_check.json;
    # este test valida que TODOS los valores reproducen la cache.
    if not CHECK_PATH.exists():
        pytest.skip("v9_reconstruction_check.json no generado (ejecuta "
                    "scripts/v9_check_reconstruction.py)")
    chk = json.loads(CHECK_PATH.read_text(encoding="utf-8"))
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    assert chk["all_match"] is True
    # Doble verificación directa contra la cache.
    assert chk["v1_linear_14_auc"] == pytest.approx(
        cache["v1v2"]["v1_linear_14"]["auc"], abs=1e-9)
    assert chk["v2_hgb_14_auc"] == pytest.approx(
        cache["v1v2"]["v2_hgb_14"]["auc"], abs=1e-9)
    assert chk["v3_v6_mean"] == pytest.approx(cache["v3"]["v6"]["mean"], abs=1e-6)


@pytest.mark.integration
def test_c_v1_v4_recompute_matches_cache():
    """Re-cómputo completo (lento) de V1-V4 contra la cache."""
    excluded = cg.load_excluded_caseids()
    sha, holdout_ids, _ = _holdout_sha()
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    assert sha == cache["meta"]["holdout_caseids_sha256"]

    # Reconstruir holdout_ids completos (no solo el sha).
    real_ids = []
    import pyarrow.parquet as pq
    d = Path("data/windows_v4/windows/source=real/split=val")
    for part in sorted(d.glob("part-*.parquet")):
        df = pq.read_table(part, columns=["caseid", "phase_from_clinical"]).to_pandas()
        df = cg.filter_cells(df, excluded)
        real_ids.extend(int(c) for c in df["caseid"].unique())
    real_ids = sorted(set(real_ids))
    rng = np.random.default_rng(20260922)
    perm = rng.permutation(real_ids)
    k = int(round(len(perm) * 0.40))
    holdout_ids = sorted(int(c) for c in perm[:k])

    real_all = vv.load_cells_from(Path("data/windows_v4"), ["real"], "val", excluded)
    hmask = np.isin(real_all["caseid"], holdout_ids)
    holdout = {k: real_all[k][hmask] for k in ("values", "masks", "caseid")}
    synth = vv.load_cells_from(Path("data/windows_v4"),
                               ["synthetic_v7", "vaso_reinf_v7", "cf_v7"],
                               "val", excluded)
    v1v2 = vv._run_v1_v2(holdout, synth)
    assert v1v2["v1_linear_14"]["auc"] == pytest.approx(
        cache["v1v2"]["v1_linear_14"]["auc"], abs=1e-6)
    assert v1v2["v2_hgb_14"]["auc"] == pytest.approx(
        cache["v1v2"]["v2_hgb_14"]["auc"], abs=1e-6)
    v3 = vv._run_v3(holdout, synth, None)
    assert v3["v6"]["mean"] == pytest.approx(cache["v3"]["v6"]["mean"], abs=1e-6)
    v4 = vv._run_v4(holdout, synth)
    assert v4["w1_norm"]["Solar8000/BT"]["w1_norm"] == pytest.approx(
        cache["v4"]["w1_norm"]["Solar8000/BT"]["w1_norm"], abs=1e-6)
