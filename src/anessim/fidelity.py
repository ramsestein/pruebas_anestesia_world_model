"""Fidelity analysis: how similar are synthetic cases to real VitalDB cases.

Compares per-channel distributions, missingness, clinical covariates and case
duration between a real cohort and a synthetic cohort, and writes a JSON +
Markdown report. No ground-truth assumption: this is a distribution-level
comparison, not a claim that synthetic cases are interchangeable with real ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

# Channels present in both real and synthetic (VitalDB naming).
COMMON_CHANNELS = [
    "Solar8000/HR",
    "Solar8000/ART_MBP",
    "Solar8000/PLETH_SPO2",
    "Solar8000/ETCO2",
    "Solar8000/BT",
    "Solar8000/RR_CO2",
    "BIS/BIS",
    "BIS/SQI",
    "Primus/MAC",
    "Primus/EXP_SEVO",
    "Primus/FIO2",
    "Primus/ETCO2",
    "Primus/PIP_MBAR",
    "Primus/TV",
    "Primus/MV",
    "Primus/SET_FIO2",
    "Primus/SET_RR_IPPV",
    "Primus/SET_TV_L",
    "Primus/SET_INTER_PEEP",
    "Primus/SET_PIP",
    "Solar8000/NIBP_MBP",
]

CLINICAL_COVARIATES = ["age", "height", "weight", "bmi", "asa"]


def _channel_stats(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {"n": 0, "mean": None, "std": None, "p50": None, "p1": None, "p99": None}
    return {
        "n": int(len(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p50": float(np.percentile(v, 50)),
        "p1": float(np.percentile(v, 1)),
        "p99": float(np.percentile(v, 99)),
    }


def _distribution_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    # Sample to keep the metric tractable.
    if len(a) > 20000:
        a = np.random.default_rng(0).choice(a, 20000, replace=False)
    if len(b) > 20000:
        b = np.random.default_rng(0).choice(b, 20000, replace=False)
    return float(stats.wasserstein_distance(a, b))


def _read_column(path: Path, column: str) -> np.ndarray:
    """Read one column from a parquet, returning an empty array if absent."""
    try:
        cols = pl.read_parquet(path, columns=[column])
    except Exception:  # noqa: BLE001 - column may not exist in this case
        return np.array([], dtype=float)
    return cols[column].to_numpy()


def compare_real_synthetic(
    real_cases_dir: Path,
    synthetic_cases_dir: Path,
    real_clinical: Path,
    synthetic_clinical: Path,
    n_cases: int = 500,
    out_path: Path | None = None,
    seed: int = 20260816,
) -> dict:
    """Compare real vs synthetic cohorts and write a report."""
    rng = np.random.default_rng(seed)

    real_paths = sorted(real_cases_dir.glob("*.parquet"))
    synth_paths = sorted(synthetic_cases_dir.glob("*.parquet"))
    if len(real_paths) > n_cases:
        real_paths = rng.choice(real_paths, n_cases, replace=False).tolist()
    if len(synth_paths) > n_cases:
        synth_paths = rng.choice(synth_paths, n_cases, replace=False).tolist()

    channel_report: dict[str, dict] = {}
    for channel in COMMON_CHANNELS:
        real_vals = []
        synth_vals = []
        real_missing = []
        synth_missing = []
        for p in real_paths:
            a = _read_column(p, channel)
            real_missing.append(float(np.mean(~np.isfinite(a))) if len(a) else 1.0)
            real_vals.append(a[np.isfinite(a)])
        for p in synth_paths:
            a = _read_column(p, channel)
            synth_missing.append(float(np.mean(~np.isfinite(a))) if len(a) else 1.0)
            synth_vals.append(a[np.isfinite(a)])

        real_all = np.concatenate(real_vals) if real_vals else np.array([])
        synth_all = np.concatenate(synth_vals) if synth_vals else np.array([])
        channel_report[channel] = {
            "real": _channel_stats(real_all),
            "synthetic": _channel_stats(synth_all),
            "missingness_real": float(np.mean(real_missing)) if real_missing else None,
            "missingness_synthetic": float(np.mean(synth_missing)) if synth_missing else None,
            "wasserstein_1d": _distribution_distance(real_all, synth_all),
        }

    # Case duration.
    real_dur = []
    synth_dur = []
    for p in real_paths:
        t = pl.read_parquet(p, columns=["time"])["time"].to_numpy()
        real_dur.append(float(t.max() - t.min()) if len(t) else 0.0)
    for p in synth_paths:
        t = pl.read_parquet(p, columns=["time"])["time"].to_numpy()
        synth_dur.append(float(t.max() - t.min()) if len(t) else 0.0)

    # Clinical covariates.
    real_clin = pl.read_parquet(real_clinical)
    synth_clin = pl.read_parquet(synthetic_clinical)
    clinical_report: dict[str, dict] = {}
    for cov in CLINICAL_COVARIATES:
        if cov not in real_clin.columns or cov not in synth_clin.columns:
            continue
        rv = real_clin[cov].to_numpy()
        sv = synth_clin[cov].to_numpy()
        clinical_report[cov] = {
            "real": _channel_stats(rv),
            "synthetic": _channel_stats(sv),
            "wasserstein_1d": _distribution_distance(rv.astype(float), sv.astype(float)),
        }
    if "sex" in real_clin.columns and "sex" in synth_clin.columns:
        clinical_report["sex_real"] = real_clin["sex"].value_counts().to_dicts()
        clinical_report["sex_synthetic"] = synth_clin["sex"].value_counts().to_dicts()

    report = {
        "phase": "F2",
        "n_real_cases": len(real_paths),
        "n_synthetic_cases": len(synth_paths),
        "case_duration_s": {
            "real": _channel_stats(np.array(real_dur)),
            "synthetic": _channel_stats(np.array(synth_dur)),
        },
        "channels": channel_report,
        "clinical": clinical_report,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        md_path = out_path.with_suffix(".md")
        md_path.write_text(_to_markdown(report), encoding="utf-8")
    return report


def _to_markdown(report: dict) -> str:
    lines = [
        "# Fidelidad sintético vs real (F2)",
        "",
        f"- Casos reales muestreados: {report['n_real_cases']}",
        f"- Casos sintéticos muestreados: {report['n_synthetic_cases']}",
        "",
        "## Duración de caso (s)",
        "",
        "| Cohorte | media | std | p50 | p1 | p99 |",
        "|---|---|---|---|---|---|",
    ]
    d = report["case_duration_s"]
    for name in ("real", "synthetic"):
        s = d[name]
        lines.append(f"| {name} | {s['mean']:.0f} | {s['std']:.0f} | {s['p50']:.0f} | {s['p1']:.0f} | {s['p99']:.0f} |")
    lines += [
        "",
        "## Canales (distribución marginal)",
        "",
        "| Canal | media real | media sint | missing real | missing sint | W1 |",
        "|---|---|---|---|---|---|",
    ]
    for ch, r in report["channels"].items():
        rl, sy = r["real"], r["synthetic"]
        rm = r["missingness_real"]
        sm = r["missingness_synthetic"]
        lines.append(
            f"| {ch} | {rl['mean'] if rl['mean'] is not None else '-'} | "
            f"{sy['mean'] if sy['mean'] is not None else '-'} | {rm:.2f} | {sm:.2f} | "
            f"{r['wasserstein_1d']:.3f} |"
        )
    lines += ["", "## Covariables clínicas", "", "| Covariable | media real | media sint | W1 |", "|---|---|---|---|"]
    for cov, r in report["clinical"].items():
        if isinstance(r, dict) and "real" in r and isinstance(r["real"], dict) and "mean" in r["real"]:
            lines.append(
                f"| {cov} | {r['real']['mean'] if r['real']['mean'] is not None else '-'} | "
                f"{r['synthetic']['mean'] if r['synthetic']['mean'] is not None else '-'} | "
                f"{r['wasserstein_1d']:.3f} |"
            )
    lines += [
        "",
        "> W1 = distancia de Wasserstein 1D (menor = más parecido). "
        "Comparación marginal, no implica equivalencia clínica.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compare real vs synthetic fidelity")
    p.add_argument("--real-cases", type=Path, default="data/real/cases")
    p.add_argument("--synth-cases", type=Path, default="D:/data/anestesia_world/synthetic_v5/cases")
    p.add_argument("--real-clinical", type=Path, default="data/real/clinical_data_enriched.parquet")
    p.add_argument("--synth-clinical", type=Path, default="D:/data/anestesia_world/synthetic_v5/clinical_data.parquet")
    p.add_argument("--n-cases", type=int, default=500)
    p.add_argument("--out", type=Path, default="reports/fidelity_v5.json")
    args = p.parse_args()
    compare_real_synthetic(
        args.real_cases, args.synth_cases, args.real_clinical, args.synth_clinical,
        n_cases=args.n_cases, out_path=args.out,
    )
