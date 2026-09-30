"""Sampling of virtual patients from the empirical VitalDB distribution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Patient:
    """Static patient characteristics used to initialise a simulation."""

    caseid: int
    age: float  # years
    sex: str  # 'M' or 'F'
    height: float  # cm
    weight: float  # kg
    bmi: float
    asa: int | None
    department: str
    optype: str
    opname: str
    approach: str
    position: str
    ane_type: str
    preop_htn: int
    preop_dm: int
    preop_ecg: str | None
    preop_pft: str | None
    preop_hb: float | None
    preop_plt: int | None
    preop_pt: int | None
    preop_aptt: float | None
    preop_na: int | None
    preop_k: float | None
    preop_gluc: int | None
    preop_alb: float | None
    preop_ast: int | None
    preop_alt: int | None
    preop_bun: int | None
    preop_cr: float | None
    # INSPIRE-derived summaries
    diagnosis_count: int = 0
    top_diagnosis_code: str | None = None
    top_diagnosis_chapter: str | None = None
    lab_count: int = 0
    lab_unique_items: int = 0
    medication_count: int = 0
    medication_unique: int = 0
    common_route: str | None = None
    airway: str | None = None

    @property
    def lean_body_weight(self) -> float:
        """Janmahasatian lean body weight (kg)."""
        if self.sex == "M":
            return 9270 * self.weight / (6680 + 216 * self.bmi)
        return 9270 * self.weight / (8780 + 244 * self.bmi)

    @property
    def ideal_body_weight(self) -> float:
        """Devine formula (kg)."""
        if self.sex == "M":
            return 50 + 0.91 * (self.height - 152.4)
        return 45.5 + 0.91 * (self.height - 152.4)

    def to_clinical_dict(self) -> dict:
        """Return a dictionary with the 74 VitalDB clinical columns."""
        return {
            "caseid": self.caseid,
            "subjectid": self.caseid,  # synthetic, one-to-one mapping
            "casestart": 0,
            "caseend": 0,  # set by renderer after simulation
            "anestart": 0,
            "aneend": None,
            "opstart": 0,
            "opend": None,
            "adm": 0,
            "dis": None,
            "icu_days": 0,
            "death_inhosp": 0,
            "age": self.age,
            "sex": self.sex,
            "height": self.height,
            "weight": self.weight,
            "bmi": self.bmi,
            "asa": self.asa,
            "emop": 0,
            "department": self.department,
            "optype": self.optype,
            "dx": "",  # derived from opname by renderer
            "opname": self.opname,
            "approach": self.approach,
            "position": self.position,
            "ane_type": self.ane_type,
            "preop_htn": self.preop_htn,
            "preop_dm": self.preop_dm,
            "preop_ecg": self.preop_ecg,
            "preop_pft": self.preop_pft,
            "preop_hb": self.preop_hb,
            "preop_plt": self.preop_plt,
            "preop_pt": self.preop_pt,
            "preop_aptt": self.preop_aptt,
            "preop_na": self.preop_na,
            "preop_k": self.preop_k,
            "preop_gluc": self.preop_gluc,
            "preop_alb": self.preop_alb,
            "preop_ast": self.preop_ast,
            "preop_alt": self.preop_alt,
            "preop_bun": self.preop_bun,
            "preop_cr": self.preop_cr,
            "preop_ph": None,
            "preop_hco3": None,
            "preop_be": None,
            "preop_pao2": None,
            "preop_paco2": None,
            "preop_sao2": None,
            "cormack": None,
            "airway": self.airway,
            "tubesize": None,
            "dltubesize": None,
            "lmasize": None,
            "iv1": None,
            "iv2": None,
            "aline1": None,
            "aline2": None,
            "cline1": None,
            "cline2": None,
            "intraop_ebl": 0,
            "intraop_uo": 0,
            "intraop_rbc": 0,
            "intraop_ffp": 0,
            "intraop_crystalloid": 0,
            "intraop_colloid": 0,
            "intraop_ppf": 0,
            "intraop_mdz": 0.0,
            "intraop_ftn": 0,
            "intraop_rocu": 0,
            "intraop_vecu": 0,
            "intraop_eph": 0,
            "intraop_phe": 0,
            "intraop_epi": 0,
            "intraop_ca": 0,
            "diagnosis_count": self.diagnosis_count,
            "top_diagnosis_code": self.top_diagnosis_code,
            "top_diagnosis_chapter": self.top_diagnosis_chapter,
            "lab_count": self.lab_count,
            "lab_unique_items": self.lab_unique_items,
            "medication_count": self.medication_count,
            "medication_unique": self.medication_unique,
            "common_route": self.common_route,
        }


class PatientSampler:
    """Sample virtual patients from the empirical VitalDB clinical table."""

    def __init__(self, clinical_path: Path | str) -> None:
        self.df = pl.read_parquet(clinical_path)
        self._numeric_cols = [
            c for c, dtype in zip(self.df.columns, self.df.dtypes)
            if dtype in (pl.Float32, pl.Float64, pl.Int64, pl.Int32)
        ]
        self._categorical_cols = [c for c in self.df.columns if c not in self._numeric_cols]
        # Precompute categorical distributions
        self._cat_dist = {
            c: list(self.df[c].value_counts().to_dicts()) for c in self._categorical_cols if c != "caseid"
        }

    def sample(self, caseid: int, rng: np.random.Generator | None = None) -> Patient:
        """Sample one virtual patient, preserving marginal correlations naively."""
        if rng is None:
            rng = np.random.default_rng()

        # Sample a donor row and then perturb continuous variables
        idx = int(rng.integers(0, self.df.shape[0]))
        row = self.df.row(idx, named=True)

        age = self._perturb(row["age"], 5.0, 18.0, 90.0, rng)
        height = self._perturb(row["height"], 3.0, 130.0, 210.0, rng)
        weight = self._perturb(row["weight"], 5.0, 35.0, 180.0, rng)
        bmi = weight / ((height / 100.0) ** 2)

        asa = row["asa"]
        if asa is not None and rng.random() < 0.05:
            asa = int(rng.choice([1, 2, 3, 4]))

        # Clinical variables kept close to donor to preserve correlations
        preop_hb = self._perturb(row.get("preop_hb"), 0.5, 5.0, 18.0, rng)
        preop_plt = int(self._perturb(row.get("preop_plt"), 20.0, 50.0, 600.0, rng)) if row.get("preop_plt") else None
        preop_pt = int(self._perturb(row.get("preop_pt"), 2.0, 8.0, 18.0, rng)) if row.get("preop_pt") else None
        preop_aptt = self._perturb(row.get("preop_aptt"), 2.0, 20.0, 60.0, rng)
        preop_na = int(self._perturb(row.get("preop_na"), 2.0, 120.0, 155.0, rng)) if row.get("preop_na") else None
        preop_k = self._perturb(row.get("preop_k"), 0.2, 2.5, 6.5, rng)
        preop_gluc = int(self._perturb(row.get("preop_gluc"), 10.0, 60.0, 350.0, rng)) if row.get("preop_gluc") else None
        preop_alb = self._perturb(row.get("preop_alb"), 0.2, 1.5, 5.5, rng)
        preop_ast = int(self._perturb(row.get("preop_ast"), 10.0, 5.0, 400.0, rng)) if row.get("preop_ast") else None
        preop_alt = int(self._perturb(row.get("preop_alt"), 10.0, 5.0, 400.0, rng)) if row.get("preop_alt") else None
        preop_bun = int(self._perturb(row.get("preop_bun"), 2.0, 3.0, 80.0, rng)) if row.get("preop_bun") else None
        preop_cr = self._perturb(row.get("preop_cr"), 0.1, 0.3, 5.0, rng)

        return Patient(
            caseid=caseid,
            age=age,
            sex=row["sex"],
            height=height,
            weight=weight,
            bmi=bmi,
            asa=asa,
            department=row["department"] or "General surgery",
            optype=row["optype"] or "General",
            opname=row["opname"] or "General surgery",
            approach=row["approach"] or "Open",
            position=row["position"] or "Supine",
            ane_type=row["ane_type"] or "General",
            airway=row.get("airway") or "Oral",
            preop_htn=int(row.get("preop_htn") or 0),
            preop_dm=int(row.get("preop_dm") or 0),
            preop_ecg=row.get("preop_ecg") or "Normal Sinus Rhythm",
            preop_pft=row.get("preop_pft") or "Normal",
            preop_hb=preop_hb,
            preop_plt=preop_plt,
            preop_pt=preop_pt,
            preop_aptt=preop_aptt,
            preop_na=preop_na,
            preop_k=preop_k,
            preop_gluc=preop_gluc,
            preop_alb=preop_alb,
            preop_ast=preop_ast,
            preop_alt=preop_alt,
            preop_bun=preop_bun,
            preop_cr=preop_cr,
            diagnosis_count=int(row.get("diagnosis_count") or 0),
            top_diagnosis_code=row.get("top_diagnosis_code") or None,
            top_diagnosis_chapter=row.get("top_diagnosis_chapter") or None,
            lab_count=int(row.get("lab_count") or 0),
            lab_unique_items=int(row.get("lab_unique_items") or 0),
            medication_count=int(row.get("medication_count") or 0),
            medication_unique=int(row.get("medication_unique") or 0),
            common_route=row.get("common_route") or None,
        )

    def _perturb(
        self,
        value: float | None,
        sigma: float,
        low: float,
        high: float,
        rng: np.random.Generator,
    ) -> float | None:
        if value is None or np.isnan(value):
            return None
        new = rng.normal(value, sigma)
        return float(np.clip(new, low, high))
