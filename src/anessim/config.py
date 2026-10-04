"""Global configuration and constants for the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default paths
DEFAULT_CASES_DIR = PROJECT_ROOT / "data" / "real" / "cases"
DEFAULT_CLINICAL_PATH = PROJECT_ROOT / "data" / "real" / "clinical_data_enriched.parquet"
DEFAULT_LABS_PATH = PROJECT_ROOT / "data" / "real" / "lab_data.parquet"
DEFAULT_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"

# Temporal defaults
SECONDS_PER_MINUTE = 60.0
DEFAULT_DT_SECONDS = 0.5

# Track names aligned with VitalDB real files
TIME_COLUMN = "time"

# Core tracks the simulator will generate.  Missing or extra tracks are allowed
# to vary by case, but the set below is the canonical target.
CANONICAL_TRACKS = [
    # Monitoring
    "Solar8000/HR",
    "Solar8000/ART_SBP",
    "Solar8000/ART_MBP",
    "Solar8000/ART_DBP",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_MBP",
    "Solar8000/NIBP_DBP",
    "Solar8000/PLETH_HR",
    "Solar8000/PLETH_SPO2",
    "Solar8000/ETCO2",
    "Solar8000/RR_CO2",
    "Solar8000/BT",
    "BIS/BIS",
    "BIS/EMG",
    "BIS/SQI",
    "BIS/SEF",
    "BIS/SR",
    "BIS/TOTPOW",
    # Ventilator / gas
    "Primus/FIO2",
    "Primus/ETCO2",
    "Primus/MAC",
    "Primus/EXP_SEVO",
    "Primus/INSP_SEVO",
    "Primus/PEEP_MBAR",
    "Primus/PIP_MBAR",
    "Primus/PPLAT_MBAR",
    "Primus/MV",
    "Primus/TV",
    "Primus/RR_CO2",
    "Solar8000/VENT_RR",
    "Solar8000/VENT_PIP",
    "Solar8000/VENT_PPLAT",
    "Solar8000/VENT_TV",
    "Solar8000/VENT_MV",
    # Infusion pumps (rate and target Ce)
    "Orchestra/PPF20_RATE",
    "Orchestra/PPF20_CE",
    "Orchestra/PPF20_CP",
    "Orchestra/PPF20_CT",
    "Orchestra/PPF20_VOL",
    "Orchestra/RFTN20_RATE",
    "Orchestra/RFTN20_CE",
    "Orchestra/RFTN20_CP",
    "Orchestra/RFTN20_CT",
    "Orchestra/RFTN20_VOL",
    "Orchestra/ROC_RATE",
    "Orchestra/ROC_VOL",
]


@dataclass
class SimulatorConfig:
    """Configuration container for a generation run."""

    n_cases: int = 100
    random_seed: int | None = 42
    dt_seconds: float = DEFAULT_DT_SECONDS
    policy_fraction: float = 0.75
    include_llm_text: bool = False
    output_dir: Path = field(default_factory=lambda: DEFAULT_SYNTHETIC_DIR)
    cases_dir: Path = field(default_factory=lambda: DEFAULT_CASES_DIR)
    clinical_path: Path = field(default_factory=lambda: DEFAULT_CLINICAL_PATH)
    labs_path: Path = field(default_factory=lambda: DEFAULT_LABS_PATH)
    n_workers: int = 1
    resume: bool = True

    # Modelo del sorteo del BT inicial (ver ``simulate.sample_bt_start``).
    # Por defecto "v7_normal" porque la cohorte vigente en disco es v7 y el
    # código debe reproducirla. "uniform_c3_revert" es la corrección C3
    # (mejor W1 de BT) PENDIENTE de una regeneración GLOBAL de todas las
    # cohortes sintéticas + reentrenamiento del AE (LIMITACIONES §6).
    bt_start_model: str = "v7_normal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "random_seed": self.random_seed,
            "dt_seconds": self.dt_seconds,
            "policy_fraction": self.policy_fraction,
            "include_llm_text": self.include_llm_text,
            "output_dir": str(self.output_dir),
            "cases_dir": str(self.cases_dir),
            "clinical_path": str(self.clinical_path),
            "labs_path": str(self.labs_path),
            "n_workers": self.n_workers,
            "resume": self.resume,
            "bt_start_model": self.bt_start_model,
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "SimulatorConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulatorConfig":
        kwargs = dict(data)
        for key in ("output_dir", "cases_dir", "clinical_path", "labs_path"):
            if key in kwargs and isinstance(kwargs[key], str):
                path = Path(kwargs[key])
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                kwargs[key] = path
        return cls(**kwargs)

    def save_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)
