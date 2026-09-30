"""Compute real-data reference distributions for the v5 generator fixes.

Reads a fixed 500-case sample of real VitalDB cases (seed 2024, as the audit
does) and writes ``data/real_reference_v5.json`` with the values the A1-A4
corrections must match. Read-only over ``data/real/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
REAL_CASES = ROOT / "data" / "real" / "cases"
CLINICAL = ROOT / "data" / "real" / "clinical_data_enriched.parquet"
OUT = ROOT / "data" / "real_reference_v5.json"

# Tracks whose intra-anesthesia presence the gate measures.
PRESENCE_TRACKS = [
    "BIS/BIS", "Solar8000/HR", "Solar8000/PLETH_SPO2",
    "Primus/ETCO2", "Primus/PEEP_MBAR", "Primus/PIP_MBAR",
    "Primus/MV", "Primus/TV", "Primus/RR_CO2",
    "Solar8000/BT", "Solar8000/ETCO2",
    "Solar8000/ART_MBP", "Solar8000/PLETH_HR",
]

# max_age (s) per track, matching window.py TRACK_REGISTRY semantics:
# a grid point is "present" if the last non-null value is no older than this.
MAX_AGE = {
    "BIS/BIS": 10.0, "BIS/EMG": 10.0,
    "Solar8000/HR": 10.0, "Solar8000/PLETH_HR": 10.0,
    "Solar8000/PLETH_SPO2": 10.0,
    "Primus/ETCO2": 10.0, "Solar8000/ETCO2": 10.0,
    "Solar8000/ART_MBP": 10.0, "Solar8000/ART_SBP": 10.0, "Solar8000/ART_DBP": 10.0,
    "Primus/PEEP_MBAR": 30.0, "Primus/PIP_MBAR": 30.0,
    "Primus/MV": 30.0, "Primus/TV": 30.0, "Primus/RR_CO2": 30.0,
    "Solar8000/BT": 600.0,
}
GRID_S = 5.0

IMAGE_TRACKS = [
    "BIS/BIS", "Solar8000/HR", "Solar8000/PLETH_SPO2",
    "Primus/ETCO2", "Primus/PEEP_MBAR", "Primus/PIP_MBAR",
    "Primus/MV", "Primus/TV", "Primus/RR_CO2",
    "Solar8000/BT", "Solar8000/ART_MBP", "Solar8000/ART_SBP",
    "Solar8000/ART_DBP", "BIS/EMG",
]


def pct(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def main() -> None:
    rng = np.random.default_rng(2024)
    case_paths = sorted(REAL_CASES.glob("*.parquet"))
    if len(case_paths) > 500:
        idx = rng.choice(len(case_paths), 500, replace=False)
        case_paths = [case_paths[i] for i in np.sort(idx)]

    clin = pl.read_parquet(CLINICAL).unique(subset=["caseid"], keep="first")
    clin_map = {
        int(r["caseid"]): (
            int(r["casestart"]), int(r["caseend"]),
            int(r["anestart"]),
            None if r["aneend"] is None else float(r["aneend"]),
        )
        for r in clin.iter_rows(named=True)
    }

    presence_per_track: dict[str, list[float]] = {}
    max_gap_per_track: dict[str, list[float]] = {}
    etco2_vals: list[float] = []
    spo2_std: list[float] = []
    bt_vals: list[float] = []
    hr_vals: list[float] = []
    map_vals: list[float] = []
    bis_vals: list[float] = []

    n_used = 0
    for p in case_paths:
        caseid = int(p.stem)
        if caseid not in clin_map:
            continue
        casestart, caseend, anestart, aneend = clin_map[caseid]
        if aneend is None:
            continue
        a0 = max(anestart, casestart)
        a1 = min(aneend, caseend if caseend > 0 else aneend)
        if a1 <= a0:
            continue
        n_used += 1
        try:
            df = pl.read_parquet(p)
        except Exception:
            continue
        t = df["time"].to_numpy()
        m = (t >= a0) & (t <= a1)
        grid = np.arange(a0, a1 + GRID_S, GRID_S)
        for col in PRESENCE_TRACKS + ["Solar8000/ART_SBP", "Solar8000/ART_DBP", "BIS/EMG"]:
            if col not in df.columns:
                continue
            v = df[col].to_numpy()
            nonnull = ~np.isnan(v.astype(float))
            tt = t[nonnull]
            vv = v[nonnull]
            if tt.size == 0:
                presence_per_track.setdefault(col, []).append(0.0)
                max_gap_per_track.setdefault(col, []).append(0.0)
                continue
            # forward-fill with max_age on the 5s grid (window.py semantics)
            idx = np.searchsorted(tt, grid, side="right") - 1
            age = grid - tt[np.clip(idx, 0, None)]
            valid = (idx >= 0) & (age <= MAX_AGE.get(col, 30.0))
            presence_per_track.setdefault(col, []).append(float(valid.sum() / max(1, len(grid))))
            # max gap in seconds between consecutive non-null values
            if tt.size >= 2:
                max_gap_per_track.setdefault(col, []).append(float(np.max(np.diff(tt))))
            else:
                max_gap_per_track.setdefault(col, []).append(0.0)

        # EtCO2 (Primus), maintenance proxy = full anesthesia window
        if "Primus/ETCO2" in df.columns:
            v = df["Primus/ETCO2"].to_numpy()[m]
            v = v[~np.isnan(v.astype(float))]
            etco2_vals.extend(v.tolist())
        if "Solar8000/PLETH_SPO2" in df.columns:
            v = df["Solar8000/PLETH_SPO2"].to_numpy()[m]
            v = v[~np.isnan(v.astype(float))]
            if v.size >= 10:
                spo2_std.append(float(np.std(v)))
        if "Solar8000/BT" in df.columns:
            v = df["Solar8000/BT"].to_numpy()[m]
            v = v[~np.isnan(v.astype(float))]
            v = v[(v >= 30.0) & (v <= 43.0)]  # exclude probe-disconnection artefacts
            bt_vals.extend(v.tolist())
        if "Solar8000/HR" in df.columns:
            v = df["Solar8000/HR"].to_numpy()[m]
            v = v[~np.isnan(v.astype(float))]
            v = v[(v >= 30) & (v <= 200)]
            hr_vals.extend(v.tolist())
        if "Solar8000/ART_MBP" in df.columns:
            v = df["Solar8000/ART_MBP"].to_numpy()[m]
            v = v[~np.isnan(v.astype(float))]
            v = v[(v >= 20) & (v <= 160)]
            map_vals.extend(v.tolist())
        if "BIS/BIS" in df.columns:
            v = df["BIS/BIS"].to_numpy()[m]
            v = v[~np.isnan(v.astype(float))]
            v = v[(v >= 5) & (v <= 98)]
            bis_vals.extend(v.tolist())

    etco2_arr = np.asarray(etco2_vals)
    etco2_arr = etco2_arr[(etco2_arr >= 15) & (etco2_arr <= 70)]
    bt_arr = np.asarray(bt_vals)
    hr_arr = np.asarray(hr_vals)
    map_arr = np.asarray(map_vals)
    bis_arr = np.asarray(bis_vals)
    spo2_std_arr = np.asarray(spo2_std)

    ref = {
        "n_cases_used": n_used,
        "presence_p5": {k: pct(np.asarray(v), 5) for k, v in sorted(presence_per_track.items())},
        "presence_p50": {k: pct(np.asarray(v), 50) for k, v in sorted(presence_per_track.items())},
        "max_gap_p50_s": {k: pct(np.asarray(v), 50) for k, v in sorted(max_gap_per_track.items())},
        "max_gap_p95_s": {k: pct(np.asarray(v), 95) for k, v in sorted(max_gap_per_track.items())},
        "etco2_p1": pct(etco2_arr, 1),
        "etco2_p50": pct(etco2_arr, 50),
        "etco2_p99": pct(etco2_arr, 99),
        "spo2_std_p50": pct(spo2_std_arr, 50),
        "spo2_std_p90": pct(spo2_std_arr, 90),
        "bt_p1": pct(bt_arr, 1),
        "bt_p50": pct(bt_arr, 50),
        "bt_p99": pct(bt_arr, 99),
        "hr_p50": pct(hr_arr, 50),
        "map_p50": pct(map_arr, 50),
        "bis_p50": pct(bis_arr, 50),
    }
    OUT.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print(json.dumps(ref, indent=2))


if __name__ == "__main__":
    main()
