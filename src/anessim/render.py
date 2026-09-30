"""Render a simulated case to VitalDB-compatible parquet + sidecar files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from anessim.clinical import ClinicalTextGenerator
from anessim.config import TIME_COLUMN


def _ensure_float32_series(value: np.ndarray) -> pl.Series:
    """Convert a numpy array to a Polars Float32 series, preserving NaNs."""
    arr = np.asarray(value, dtype=np.float32)
    return pl.Series(arr).cast(pl.Float32)


def render_case(
    caseid: int,
    tracks: dict[str, np.ndarray],
    clinical_row: dict[str, Any],
    truth: dict[str, Any],
    output_dir: Path,
    patient: Any | None = None,
    timeline: Any | None = None,
    actions: Any | None = None,
    duration_min: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """Write case parquet, clinical row, sidecar truth, and clinical note.

    Returns:
        (case_path, truth_path, clinical_parquet_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    truth_dir = output_dir / "truth"
    clinical_dir = output_dir / "clinical"
    notes_dir = output_dir / "clinical_notes"
    metadata_dir = output_dir / "metadata"
    for d in (cases_dir, truth_dir, clinical_dir, notes_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Clinical note
    if patient is not None and timeline is not None and actions is not None:
        text_gen = ClinicalTextGenerator()
        note, alignments = text_gen.generate(patient, timeline, actions, duration_min)
    else:
        note = "Clinical note placeholder.\n\nEvents:\n" + "\n".join(truth.get("actions", []))
        alignments = []

    # Case time-series parquet
    # Sort columns deterministically: time first, then lexicographic tracks.
    ordered_cols = [TIME_COLUMN] + sorted([c for c in tracks.keys() if c != TIME_COLUMN])
    case_series = {}
    for col in ordered_cols:
        if col == TIME_COLUMN:
            case_series[col] = pl.Series(np.asarray(tracks[col], dtype=np.float64)).alias(col)
        else:
            case_series[col] = _ensure_float32_series(tracks[col]).alias(col)
    case_df = pl.DataFrame(case_series)
    # Replace float NaN with Polars null to match the real VitalDB convention.
    case_df = case_df.with_columns(
        [
            pl.when(pl.col(col).is_nan()).then(None).otherwise(pl.col(col)).alias(col)
            for col in case_df.columns
            if case_df[col].dtype in (pl.Float32, pl.Float64)
        ]
    )
    case_path = cases_dir / f"{caseid:04d}.parquet"
    case_df.write_parquet(case_path, compression="zstd")

    # Truth sidecar (parquet) - keep only 1-D numeric arrays of same length as time.
    truth_arrays = {}
    for key, value in truth.items():
        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == len(tracks[TIME_COLUMN]):
            truth_arrays[key] = _ensure_float32_series(value).alias(key)
        elif isinstance(value, list) and key == "phase":
            truth_arrays[key] = pl.Series(value, dtype=pl.String).alias(key)
    truth_df = pl.DataFrame(truth_arrays)
    truth_path = truth_dir / f"{caseid:04d}_truth.parquet"
    truth_df.write_parquet(truth_path, compression="zstd")

    # Clinical row parquet
    clinical_df = pl.DataFrame([clinical_row]).cast(clinical_row_dtypes())
    clinical_path = clinical_dir / f"{caseid:04d}_clinical.parquet"
    clinical_df.write_parquet(clinical_path, compression="zstd")

    # Clinical note
    note_path = notes_dir / f"{caseid:04d}.txt"
    note_path.write_text(note, encoding="utf-8")

    # Text-event alignment JSON
    alignment_path = truth_dir / f"{caseid:04d}_alignment.json"
    alignment_data = [
        {
            "text_span": a.text_span,
            "event_indices": a.event_indices,
            "start_char": a.start_char,
            "end_char": a.end_char,
        }
        for a in alignments
    ]
    alignment_path.write_text(json.dumps(alignment_data, indent=2), encoding="utf-8")

    # Oracle metadata (PK model, IIV, nociception info — NEVER exposed to model)
    if metadata:
        meta_path = metadata_dir / f"{caseid:04d}_meta.json"
        # Convert numpy types to native Python for JSON serialisation
        meta_clean = _jsonify(metadata)
        meta_path.write_text(json.dumps(meta_clean, indent=2), encoding="utf-8")

    return case_path, truth_path, clinical_path


def _jsonify(obj: Any) -> Any:
    """Convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def clinical_row_dtypes() -> dict[str, pl.DataType]:
    """Return the exact dtype mapping for the synthetic clinical row."""
    return {
        "caseid": pl.Int64,
        "subjectid": pl.Int64,
        "casestart": pl.Int64,
        "caseend": pl.Int64,
        "anestart": pl.Int64,
        "aneend": pl.Float64,
        "opstart": pl.Int64,
        "opend": pl.Int64,
        "adm": pl.Int64,
        "dis": pl.Int64,
        "icu_days": pl.Int64,
        "death_inhosp": pl.Int64,
        "age": pl.Float64,
        "sex": pl.String,
        "height": pl.Float64,
        "weight": pl.Float64,
        "bmi": pl.Float64,
        "asa": pl.Int64,
        "emop": pl.Int64,
        "department": pl.String,
        "optype": pl.String,
        "dx": pl.String,
        "opname": pl.String,
        "approach": pl.String,
        "position": pl.String,
        "ane_type": pl.String,
        "preop_htn": pl.Int64,
        "preop_dm": pl.Int64,
        "preop_ecg": pl.String,
        "preop_pft": pl.String,
        "preop_hb": pl.Float64,
        "preop_plt": pl.Int64,
        "preop_pt": pl.Int64,
        "preop_aptt": pl.Float64,
        "preop_na": pl.Int64,
        "preop_k": pl.Float64,
        "preop_gluc": pl.Int64,
        "preop_alb": pl.Float64,
        "preop_ast": pl.Int64,
        "preop_alt": pl.Int64,
        "preop_bun": pl.Int64,
        "preop_cr": pl.Float64,
        "preop_ph": pl.Float64,
        "preop_hco3": pl.Float64,
        "preop_be": pl.Float64,
        "preop_pao2": pl.Float64,
        "preop_paco2": pl.Float64,
        "preop_sao2": pl.Float64,
        "cormack": pl.String,
        "airway": pl.String,
        "tubesize": pl.Float64,
        "dltubesize": pl.String,
        "lmasize": pl.Int64,
        "iv1": pl.String,
        "iv2": pl.String,
        "aline1": pl.String,
        "aline2": pl.String,
        "cline1": pl.String,
        "cline2": pl.String,
        "intraop_ebl": pl.Int64,
        "intraop_uo": pl.Int64,
        "intraop_rbc": pl.Int64,
        "intraop_ffp": pl.Int64,
        "intraop_crystalloid": pl.Int64,
        "intraop_colloid": pl.Int64,
        "intraop_ppf": pl.Int64,
        "intraop_mdz": pl.Float64,
        "intraop_ftn": pl.Int64,
        "intraop_rocu": pl.Int64,
        "intraop_vecu": pl.Int64,
        "intraop_eph": pl.Int64,
        "intraop_phe": pl.Int64,
        "intraop_epi": pl.Int64,
        "intraop_ca": pl.Int64,
    }
