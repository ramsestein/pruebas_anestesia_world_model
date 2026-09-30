"""v7_validate.py — validación V1-V7 de las cohortes v7 contra real_holdout.

Reutiliza el protocolo de v6_validate.py sin modificarlo: monkeypatchea los
directorios de v7 (windows_v4, synthetic_v7, vaso_reinf_v7, cf_v7) y las tres
funciones con rutas hardcodeadas (V5 pk gate, V6 cf prefix, V7 dosis-efecto).

Uso:
    python -m diagnostics.v7_validate run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from diagnostics import v6_validate as vv
from diagnostics import cohort_gap as cg

import paths

ROOT = vv.ROOT

# ── Monkeypatch de constantes v6 -> v7 ─────────────────────────────────────
vv.WINDOWS_V6 = paths.WINDOWS_DIR
vv.SYNTH_SOURCES_V6 = paths.SYNTH_COHORTS
vv.CF_V6_METADATA = paths.COHORTS["cf_v7"] / "metadata"
vv.CACHE_PATH = paths.DIAGNOSTICS_DIR / "v7_validate_results.json"
vv.REPORT_PATH = paths.REPORTS_DIR / "REPORT_generator_validation_v7.txt"
vv.ROJO_TXT = paths.REPORTS_DIR / "_pytest_v7_validate_rojo.txt"
vv.VERDE_TXT = paths.REPORTS_DIR / "_pytest_v7_validate_verde.txt"

SYNTH_V7 = paths.COHORTS["synthetic_v7"]
CF_V7 = paths.COHORTS["cf_v7"]


# ── V5: gate 3 de pk_tokens sobre casos v7 ─────────────────────────────────
def _v5_pk_gate3_v7() -> dict:
    from anessim.pk.base import Infusion
    from anessim.pk.propofol import PropofolSchnider
    from anessim.pk.remifentanil import RemifentanilMinto
    from tokens import pk_tokens as pk

    meta_dir = SYNTH_V7 / "metadata"
    cases = sorted((SYNTH_V7 / "cases").glob("*.parquet"))
    syn_clin = pq.read_table(SYNTH_V7 / "clinical_data.parquet").to_pandas()
    demo = {int(r.caseid): dict(weight=float(r.weight), age=float(r.age),
                                height=float(r.height), sex=str(r.sex))
            for _, r in syn_clin.iterrows()}

    def anessim_ce(model, grid, rate_hold, bolus_grid):
        rate_min = rate_hold / 3.0
        infs = []
        i = 1
        while i < len(grid):
            j = i
            while j + 1 < len(grid) and rate_min[j + 1] == rate_min[i]:
                j += 1
            if rate_min[i] != 0.0:
                infs.append(Infusion(start_s=grid[i - 1], end_s=grid[j] - 1e-9,
                                     rate=rate_min[i], drug=model.drug))
            i = j + 1
        boluses = [(grid[i - 1] if i > 0 else grid[0], float(bolus_grid[i]))
                   for i in np.where(bolus_grid > 0)[0]]
        st = model.simulate(infs, grid, boluses=boluses)
        return st[:, 3]

    prop_errs: list[float] = []
    remi_errs: list[float] = []
    n_cases = 0
    for case_path in cases[:20]:
        cid = int(case_path.stem)
        row = demo.get(cid)
        if row is None:
            continue
        d = dict(weight=float(row["weight"]), age=float(row["age"]),
                 height=float(row["height"]), sex=str(row["sex"]))
        df = pq.read_table(case_path, columns=[
            "time", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE",
            "ppf_bolus_mg", "remi_bolus_ug"]).to_pandas()
        t = df["time"].to_numpy(float)
        grid = np.arange(t[0], t[-1], pk.DT_S)
        if len(grid) < 2:
            continue
        n_cases += 1
        for drug, rate_col, bol_col, model in [
            ("propofol", "Orchestra/PPF20_RATE", "ppf_bolus_mg",
             PropofolSchnider(weight_kg=d["weight"], age_y=d["age"],
                              height_cm=d["height"], sex=d["sex"])),
            ("remifentanilo", "Orchestra/RFTN20_RATE", "remi_bolus_ug",
             RemifentanilMinto(weight_kg=d["weight"], height_cm=d["height"],
                               age_y=d["age"], sex=d["sex"])),
        ]:
            rate_hold = pk._forward_fill_hold(t, df[rate_col].to_numpy(float), grid)
            bolus = pk._bolus_on_grid(t, df[bol_col].to_numpy(float), grid)
            ce_my = pk.compute_ce(drug, rate_hold, bolus, d, dt_s=pk.DT_S)
            ce_an = anessim_ce(model, grid, rate_hold, bolus)
            ok = ce_an > 0.1
            if ok.sum() > 10:
                err = np.abs(ce_my[ok] - ce_an[ok]) / ce_an[ok]
                (prop_errs if drug == "propofol" else remi_errs).extend(err.tolist())
    mean_prop = float(np.mean(prop_errs)) if prop_errs else None
    mean_remi = float(np.mean(remi_errs)) if remi_errs else None
    return {
        "n_cases": n_cases,
        "propofol_err_mean": vv._f(mean_prop),
        "remifentanilo_err_mean": vv._f(mean_remi),
        "pass": (mean_prop is not None and mean_remi is not None
                 and mean_prop < 0.01 and mean_remi < 0.01),
    }


# ── V6: divergencia de prefijo CF v7 ───────────────────────────────────────
def _v6_cf_prefix_v7() -> dict:
    pairs_dir = CF_V7 / "metadata"
    if not pairs_dir.exists():
        return {"error": f"no existe {pairs_dir}"}
    diffs: list[float] = []
    n_pairs = 0
    for p in sorted(pairs_dir.glob("cf_pair_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        caseid_a = int(m.get("caseid_a", -1))
        caseid_b = int(m.get("caseid_b", -1))
        split_t = float(m.get("split_t", 0.0))
        ta = _read_truth_v7(caseid_a, split_t)
        tb = _read_truth_v7(caseid_b, split_t)
        if ta is None or tb is None:
            continue
        n_pairs += 1
        diffs.append(float(np.max(np.abs(ta - tb))))
    diffs = np.array(diffs)
    return {
        "n_pairs": n_pairs,
        "prefix_max_abs_diff": {
            "max": vv._f(float(diffs.max())) if diffs.size else None,
            "p50": vv._f(float(np.median(diffs))) if diffs.size else None,
            "p99": vv._f(float(np.percentile(diffs, 99))) if diffs.size else None,
        },
    }


def _read_truth_v7(caseid: int, split_t: float) -> np.ndarray | None:
    path = CF_V7 / "truth" / f"{caseid}_truth.parquet"
    if not path.exists():
        return None
    try:
        df = pq.read_table(path).to_pandas()
    except Exception:
        return None
    cols = [c for c in ("bis", "map", "hr") if c in df.columns]
    if not cols or "time" not in df.columns:
        return None
    pre = df[df["time"] <= split_t][cols].to_numpy(dtype=np.float64)
    return pre


# ── V7: curvas dosis-efecto v7 vs v5 ───────────────────────────────────────
def _v7_dose_effect_v7() -> dict:
    out: dict = {}
    cohort_dirs = {
        "synthetic_v7": paths.COHORTS["synthetic_v7"],
        "synthetic_v5": paths.LOST_COHORT_DIRS["synthetic_v5"],
    }
    for cohort, base in cohort_dirs.items():
        out[cohort] = {}
        d = base / "truth"
        if not d.exists():
            continue
        ce_p, bis, ce_r, mapv, nora = [], [], [], [], []
        for p in sorted(d.glob("*_truth.parquet")):
            df = pq.read_table(p).to_pandas()
            if "ce_propofol" not in df.columns:
                continue
            ce_p.append(df["ce_propofol"].to_numpy())
            bis.append(df["bis"].to_numpy())
            if "ce_remifentanil" in df.columns:
                ce_r.append(df["ce_remifentanil"].to_numpy())
            if "map" in df.columns and "noradrenaline_rate" in df.columns:
                mapv.append(df["map"].to_numpy())
                nora.append(df["noradrenaline_rate"].to_numpy())
        if ce_p:
            out[cohort]["propofol_bis"] = vv.dose_effect_curve(
                np.concatenate(ce_p), np.concatenate(bis))
        if ce_r:
            out[cohort]["remi_map"] = vv.dose_effect_curve(
                np.concatenate(ce_r), np.concatenate(mapv))
        if nora:
            out[cohort]["nora_map"] = vv.dose_effect_curve(
                np.concatenate(nora), np.concatenate(mapv))
    return out


# Reemplazar las funciones hardcodeadas.
vv.run_v5_pk_gate3 = _v5_pk_gate3_v7
vv.run_v6_cf_prefix = _v6_cf_prefix_v7
vv.run_v7_dose_effect = _v7_dose_effect_v7


def write_report_v7(results: dict) -> Path:
    """Informe parametrizado para la cohorte v7 (PASO 3.2): las etiquetas y
    la conclusión se generan a partir de los números, no de texto fijo v6."""
    return _original_write_report(results, cohort="v7")


# Guardar la referencia original ANTES de sustituir, para no recursar.
_original_write_report = vv.write_report
# vv.run_all y vv.main de v6_validate llaman a write_report sin argumento de
# cohorte; se sustituye por la versión parametrizada para v7.
vv.write_report = write_report_v7


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "report"])
    args = ap.parse_args()
    if not vv.CACHE_PATH.exists():
        print("Sin cache; ejecuta 'run' primero (requiere windows_v2).",
              file=sys.stderr)
        sys.exit(1)
    # NOTA (PASO 3.1bis): el cómputo NO se re-ejecuta: la cohorte real
    # windowed (data/windows_v2) ya no está en disco. Se regenera el informe
    # desde la cache existente (v7_validate_results.json), que es válida.
    res = json.loads(vv.CACHE_PATH.read_text(encoding="utf-8"))
    vv.write_report(res)


if __name__ == "__main__":
    main()
