"""PASO 3d Fase 1 — tests del generador v7.1 (muestreo de las palancas de consigna).

Cambio bajo prueba (``generate_f2_cf.build_intervention``): set_rr, set_tv y
set_peep muestrean δ en la REJILLA del escalón de registro (1 rpm, 10 mL,
1 cmH2O) en lugar de un uniforme continuo, para que la diferencia registrada
sea exactamente δ y no 0 o un escalón al azar.

  a) test_vent_delta_en_rejilla_de_registro   δ de la rejilla y en rango.
  b) test_palancas_no_modificadas_identicas   las otras 18 palancas no cambian.
  c) test_nucleo_generador_intacto            simulate/sensors/respiratory/render
                                              intactos frente a paso3d_f0prime.json.
  d) test_registro_igual_a_aplicado           SET_b - SET_a == δ_reg (simula).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import paths  # noqa: E402
from anessim.config import SimulatorConfig  # noqa: E402
from anessim.scripts import generate_f2_cf as gcf  # noqa: E402

PAIRS = paths.TOKENS_V2_DIR / "pairs.parquet"
CF_DIR = paths.COHORTS["cf_v7"]
CF_META = CF_DIR / "metadata"
CONFIG_V7 = ROOT / "src" / "anessim" / "configs" / "synthetic_v7.yaml"
F0PRIME = paths.MANIFESTS_DIR / "paso3d_f0prime.json"

REJILLA = {"set_rr": (1.0, 4.0), "set_tv": (10.0, 150.0), "set_peep": (1.0, 5.0)}
NUCLEO = ["src/anessim/simulate.py", "src/anessim/sensors.py",
          "src/anessim/respiratory.py", "src/anessim/render.py"]


class _P:
    """Paciente mínimo para build_intervention (sólo necesita .weight)."""

    def __init__(self, weight: float = 70.0):
        self.weight = float(weight)


def _sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _meta(pid: int) -> dict:
    return json.loads((CF_META / f"cf_pair_{pid}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# a) δ en la rejilla
# ---------------------------------------------------------------------------

def test_vent_delta_en_rejilla_de_registro():
    pairs = pq.read_table(PAIRS, columns=["pair_id", "lever"]).to_pandas()
    for lever, (step, hi) in REJILLA.items():
        sub = pairs[pairs.lever == lever]
        assert len(sub) > 0, f"no hay pares para {lever}"
        for pid in sub.pair_id.to_numpy():
            m = _meta(int(pid))
            rng = np.random.default_rng(int(m["seed"]))
            _, _, ov, _ = gcf.build_intervention(
                lever, _P(), float(m["split_t"]), rng, None)
            d = ov["rr_delta"] if lever == "set_rr" else (
                ov["tv_delta"] if lever == "set_tv" else ov["peep_delta"])
            k = d / step
            assert abs(k - round(k)) < 1e-9, (pid, lever, d)
            assert round(k) != 0, (pid, lever, d)
            assert -hi <= d <= hi, (pid, lever, d)


# ---------------------------------------------------------------------------
# b) las otras 18 palancas no cambian
# ---------------------------------------------------------------------------

def _weight_of(pid: int) -> float:
    df = pq.read_table(CF_DIR / "clinical" / f"{pid:04d}_clinical.parquet",
                       columns=["weight"]).to_pandas()
    return float(df["weight"].iloc[0])


def _probe_truth_like(pid: int) -> dict:
    """Truth de referencia para las palancas que dependen de la línea base.

    Se lee de la rama BASE del par (``caseid_base``), que comparte plan con el
    probe (ambas ramas de un par comparten semilla y plan).
    """
    pairs = pq.read_table(PAIRS, columns=["pair_id", "caseid_base"]).to_pandas()
    base = int(pairs.loc[pairs.pair_id == pid, "caseid_base"].iloc[0])
    t = pq.read_table(CF_DIR / "truth" / f"{base:04d}_truth.parquet",
                      columns=["time", "remifentanil_rate",
                               "sevoflurane_mac"]).to_pandas()
    return {"time": t["time"].to_numpy(np.float64),
            "remifentanil_rate": t["remifentanil_rate"].to_numpy(np.float64),
            "sevoflurane_mac": t["sevoflurane_mac"].to_numpy(np.float64)}


def test_palancas_no_modificadas_identicas(tmp_path):
    pairs = pq.read_table(PAIRS).to_pandas()
    levers = sorted(set(gcf.PHARMA_LEVERS + gcf.VENT_LEVERS + gcf.LEARNING_LEVERS)
                    - set(REJILLA))
    assert len(levers) == 18
    failures = []
    for lever in levers:
        sub = pairs[pairs.lever == lever]
        assert len(sub) > 0, lever
        for row in sub.itertuples():
            pid = int(row.pair_id)
            m = _meta(pid)
            split_t = float(m["split_t"])
            rng = np.random.default_rng(int(m["seed"]))
            pt = _probe_truth_like(pid) if lever in ("remi_bolus", "sevo_mac") else None
            a, b, ov_a, ov_b = gcf.build_intervention(
                lever, _P(_weight_of(pid)), split_t, rng, pt)
            got_a = [x.summary() for x in a if x.t_s > split_t]
            if got_a != list(m["intervention_a"]) or ov_a != m["vent_override_a"] \
                    or ov_b != m["vent_override_b"]:
                failures.append({"pair_id": pid, "lever": lever,
                                 "got": got_a, "recorded": m["intervention_a"]})
                if len(failures) >= 5:
                    break
        if len(failures) >= 5:
            break
    assert not failures, f"palancas no modificadas divergen: {failures}"


# ---------------------------------------------------------------------------
# c) núcleo intacto
# ---------------------------------------------------------------------------

def test_nucleo_generador_intacto():
    assert F0PRIME.exists(), "falta manifests/paso3d_f0prime.json (Fase 0′)"
    shas = json.loads(F0PRIME.read_text(encoding="utf-8"))["core_shas"]
    for f in NUCLEO:
        assert _sha256(ROOT / f) == shas[f], f"el núcleo cambió: {f}"


# ---------------------------------------------------------------------------
# d) consigna registrada == aplicada (δ de la rejilla)
# ---------------------------------------------------------------------------

def _truth_applied_first_diff(ta: pd.DataFrame, tb: pd.DataFrame,
                              cols: tuple[str, ...]) -> float:
    m = ta[["time", *cols]].merge(tb[["time", *cols]], on="time",
                                  suffixes=("_a", "_b"))
    for c in cols:
        va = m[f"{c}_a"].to_numpy(np.float64)
        vb = m[f"{c}_b"].to_numpy(np.float64)
        d = ~((va == vb) | (np.isnan(va) & np.isnan(vb)))
        if d.any():
            return float(m["time"].to_numpy()[int(np.argmax(d))])
    return float("nan")


@pytest.mark.slow
def test_registro_igual_a_aplicado(tmp_path):
    cfg = SimulatorConfig.from_yaml(CONFIG_V7)
    if not Path(cfg.clinical_path).exists():
        pytest.skip("faltan datos clínicos reales")
    from anessim.simulate import CaseSimulator

    default_cols = {
        # lever: (col_verdad, consigna, δ_verdad, δ_registro, escalón, (lo, hi) obs)
        "set_rr": ("rr_applied", "Primus/SET_RR_IPPV", 3.0, 3.0, 1.0, (0.0, 70.0)),
        "set_tv": ("tv_applied", "Primus/SET_TV_L", 30.0, 0.03, 0.01, (0.0, 1.6)),
        "set_peep": ("peep_applied", "Primus/SET_INTER_PEEP", 3.0, 3.0, 1.0, (0.0, 25.0)),
    }
    forced = {"set_rr": {"rr_delta": 3.0},
              "set_tv": {"tv_delta": 30.0},
              "set_peep": {"fio2_set": 0.35, "peep_delta": 3.0}}
    base_caseid = 171001  # fuera de los rangos de cf_v7
    seed = 12345
    out = tmp_path
    for sub in ("cases", "truth", "metadata", "clinical", "clinical_notes"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cfg.output_dir = out
    probe = CaseSimulator(SimulatorConfig.from_dict(
        {**cfg.to_dict(), "random_seed": seed})).run(base_caseid)
    maint = next(p for p in probe["timeline"].phases if p.name == "maintenance")
    split_t = (maint.start_s + maint.end_s) / 2.0
    presence = [c for c in probe["tracks"] if c != "time"]

    for lever, (truth_col, setpoint, d_truth, d_reg, step, cap) in default_cols.items():
        ov = forced[lever]
        phys = {"fio2_baseline": 0.35, "shunt_fraction": 0.15} if lever == "set_peep" else None
        gcf.generate_pair(cfg, out, base_caseid, lever, split_t, presence,
                          [], [], ov, None, seed, "learning", physiology=phys)
        ca = pq.read_table(out / "cases" / f"{base_caseid:04d}.parquet",
                           columns=["time", setpoint]).to_pandas()
        cb = pq.read_table(out / "cases" / f"{base_caseid + 1:04d}.parquet",
                           columns=["time", setpoint]).to_pandas()
        ta = pq.read_table(out / "truth" / f"{base_caseid:04d}_truth.parquet",
                           columns=["time", truth_col]).to_pandas()
        tb = pq.read_table(out / "truth" / f"{base_caseid + 1:04d}_truth.parquet",
                           columns=["time", truth_col]).to_pandas()
        m = ca.merge(cb, on="time", suffixes=("_a", "_b"))
        mt = ta.merge(tb, on="time", suffixes=("_a", "_b"))
        m = m.merge(mt, on="time")
        post = (m["time"] > split_t).to_numpy()
        # muestras post-split en que la verdad APLICADA difiere exactamente δ
        va = m[f"{truth_col}_a"].to_numpy(np.float64)
        vb = m[f"{truth_col}_b"].to_numpy(np.float64)
        applied_ok = post & (np.abs((va - vb) - d_truth) < 1e-6)
        sa = m[f"{setpoint}_a"].to_numpy(np.float64)
        sb = m[f"{setpoint}_b"].to_numpy(np.float64)
        lo, hi = cap
        # Excluye muestras con el CLIP DE OBSERVACIÓN activo (observe recorta a
        # [lo, hi] ANTES de cuantizar): un valor recortado es exactamente lo/hi,
        # así que estrictamente dentro del rango equivale a "no recortado".
        interior = ((sa > lo) & (sa < hi)) & ((sb > lo) & (sb < hi))
        both = np.isfinite(sa) & np.isfinite(sb) & applied_ok & interior
        assert both.sum() > 0, f"{lever}: sin muestras post-split comparables"
        diff = sa[both] - sb[both]
        bad = np.abs(diff - d_reg) > step / 2.0
        assert not bad.any(), (lever, diff[bad][:5], d_reg, step)
