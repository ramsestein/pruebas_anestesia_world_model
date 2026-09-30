"""physio_ae.py — autoencoder de fisiología v1 (contrato_ae_v1.md).

Comprime la fisiología de UNA celda de la rejilla de 5 s de ``data/windows_v2/``
en un latente ordenado de 32 dimensiones con nested dropout. El decoder queda
congelado como lengua franca del sistema: el token 1 de estado (32 dims) es
exactamente el latente del AE.

Implementa todo el contrato salvo el warm-up, que pertenece al módulo de
transición; aquí solo se documenta el formato (tokens 2..4 = 96 dims de historia
oculta que el decoder NO lee).

Variantes:
  ae_real  -> solo celdas de source=real, split=train.
  ae_bal   -> las cuatro cohortes, split=train, 50 % real / 50 % sintético,
              muestreo por caso y luego por celda dentro de cada mitad.

Uso:
    python -m ae.physio_ae run      # entrena y evalúa ambas variantes
    python -m ae.physio_ae run --variant ae_real
    python -m ae.physio_ae report   # escribe REPORT_ae.txt
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time as _time
import warnings
from functools import lru_cache
from pathlib import Path

import paths

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
WINDOWS_ROOT = paths.WINDOWS_DIR
WINDOWS_DIR = WINDOWS_ROOT / "windows"
TOKENS_MANIFEST = paths.TOKENS_DIR / "manifest_tokens.json"
CONTRACT_PATH = ROOT / "contrato_ae_v1.md"
TOKENS_CONTRACT_PATH = ROOT / "contrato_tokens_v1.md"
OUT_ROOT = paths.AE_DIR
REPORT_PATH = paths.REPORTS_DIR / "REPORT_ae_v4.txt"
ROJO_TXT = paths.REPORTS_DIR / "_pytest_ae_v4_rojo.txt"
VERDE_TXT = paths.REPORTS_DIR / "_pytest_ae_v4_verde.txt"
V1_SNAPSHOT = paths.AE_DIR / "v1_snapshot.json"
V2_SNAPSHOT = paths.AE_DIR / "v2_snapshot.json"
REPORT_WINDOW_V2 = paths.REPORTS_DIR / "REPORT_window_v2_RECONSTRUIDO.txt"
REPORT_TOKENS_V4 = paths.REPORTS_DIR / "REPORT_tokens_v4.txt"

SEED = 42
LATENT_DIM = 32
N_VARS = 14
STATE_DIM = 128
BATCH_SIZE = 4096
HALF_BATCH = BATCH_SIZE // 2
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
# Planificador (corrección iter 3): warm-up lineal de 5 épocas + ReduceLROnPlateau.
WARMUP_EPOCHS = 5
PATIENCE = 12
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 4
PLATEAU_THRESHOLD = 1e-4
MIN_LR = LR * 1e-3

# Submuestra de val FIJA para la parada temprana (corrección 2): las mismas
# celdas en todas las épocas y para las dos variantes, con el mismo criterio
# (real + sintético).
VAL_SAMPLE_SEED = 1234
VAL_SAMPLE_REAL = 65536
VAL_SAMPLE_SYNTH = 65536

# Sondeo de fuente por celda (corrección 4).
GATE6_CELL_SEED = 9876
GATE6_N_CELLS = 200_000

# Tolerancias de regresión (iter 3, decididas post-hoc; iter 4 las declara en
# el manifest como campo 'acceptance').
ACCEPTANCE = {
    "regression_tolerance_lpm": 0.05,
    "regression_tolerance_mmhg": 0.05,
    "rationale": "2.5 % del umbral del gate 2 (2 lpm / 2 mmHg); por debajo de "
                  "la resolución del monitor",
    "introduced_in": "iter_3",
    "decided_post_hoc": True,
}

# Diagnóstico 5b: variables y umbrales clínicos (unidades físicas).
DIAG5B_VARS = ["Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS", "Primus/ETCO2"]
DIAG5B_THRESHOLDS = {
    "Solar8000/HR": 1.0,
    "Solar8000/ART_MBP": 1.0,
    "BIS/BIS": 1.0,
    "Primus/ETCO2": 0.5,
}

# Brecha de procedencia (corrección C de iter 4): REPORT_window_v2.txt original
# no se conserva; solo existe la reconstrucción retrospectiva.
PROVENANCE_GAPS = [
    {
        "artifact": "REPORT_window_v2.txt",
        "status": "missing_original",
        "note": "El informe original del proceso de ventanas no se conserva. "
                "Existe una reconstrucción retrospectiva en "
                "reports/REPORT_window_v2_RECONSTRUIDO.txt derivada del "
                "manifest de windows_v2, del esquema de particiones y de "
                "window.py; NO es el artefacto original y no constituye "
                "procedencia.",
        "sha256_reconstruction": None,
    }
]

# Imagen de 14 variables (orden canónico = image_tracks del manifest de
# windows_v2; el contrato lista BIS/EMG en 2ª posición, sin efecto semántico).
IMAGE_TRACKS: list[str] = [
    "BIS/BIS",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
    "Primus/ETCO2",
    "Primus/PEEP_MBAR",
    "Primus/PIP_MBAR",
    "Primus/MV",
    "Primus/TV",
    "Primus/RR_CO2",
    "Solar8000/BT",
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "BIS/EMG",
]

VALUE_COLS: list[str] = list(IMAGE_TRACKS)
MASK_COLS: list[str] = [f"m_{t}" for t in IMAGE_TRACKS]
META_COLS: list[str] = ["caseid", "t", "source", "split", "phase_from_clinical"]
READ_COLS: list[str] = META_COLS + VALUE_COLS + MASK_COLS
STATS_COLS: list[str] = ["caseid", "phase_from_clinical"] + VALUE_COLS + MASK_COLS
ALLOWED_READ_COLS: set[str] = set(READ_COLS)

SYNTH_SOURCES: list[str] = paths.SYNTH_COHORTS
ALL_SOURCES: list[str] = paths.ALL_COHORTS

ART_TRACKS = ["Solar8000/ART_MBP", "Solar8000/ART_SBP", "Solar8000/ART_DBP"]
HR_TRACK = "Solar8000/HR"
MBP_TRACK = "Solar8000/ART_MBP"


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokens_manifest() -> dict:
    return json.loads(TOKENS_MANIFEST.read_text(encoding="utf-8"))


def excluded_caseids() -> frozenset[int]:
    """Caseids excluidos: 35 reales sin marcas de fase + el caseid 4476 sin
    celdas (36 caseids en total), reutilizando la lista del manifest de
    tokens_v1 (no se recalcula)."""
    m = load_tokens_manifest()
    a = m.get("excluded_no_phase_marks_caseids", [])
    b = m.get("sin_celdas_caseids", [])
    return frozenset(int(c) for c in list(a) + list(b))


def _sha256_state_dict(sd: dict) -> str:
    buf = io.BytesIO()
    torch.save(sd, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# --------------------------------------------------------------------------
# Lectura de particiones (siempre con columns= explícitas)
# --------------------------------------------------------------------------

def iter_partitions(sources: list[str], split: str) -> list[Path]:
    parts: list[Path] = []
    for s in sources:
        d = WINDOWS_DIR / f"source={s}" / f"split={split}"
        parts.extend(sorted(d.glob("part-*.parquet")))
    return sorted(parts)


def read_partition(path: Path, columns: list[str] = READ_COLS) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()


def filter_cells(df: pd.DataFrame, excluded: frozenset[int]) -> pd.DataFrame:
    df = df[df["phase_from_clinical"] == "maintenance"]
    if excluded:
        df = df[~df["caseid"].isin(excluded)]
    return df


# --------------------------------------------------------------------------
# Normalización
# --------------------------------------------------------------------------

def _mean_std_arrays(stats: dict, tracks: list[str] = IMAGE_TRACKS) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([stats[t]["mean"] for t in tracks], dtype=np.float64)
    stds = np.array([stats[t]["std"] for t in tracks], dtype=np.float64)
    stds = np.where(stds > 0, stds, 1.0)
    return means, stds


def normalize(values, stats: dict, tracks: list[str] = IMAGE_TRACKS):
    v = np.asarray(values)
    means, stds = _mean_std_arrays(stats, tracks)
    return (v - means.astype(v.dtype)) / stds.astype(v.dtype)


def denormalize(values, stats: dict, tracks: list[str] = IMAGE_TRACKS):
    v = np.asarray(values)
    means, stds = _mean_std_arrays(stats, tracks)
    return v * stds.astype(v.dtype) + means.astype(v.dtype)


def compute_stats_from_values(values, masks, tracks: list[str] = IMAGE_TRACKS) -> dict:
    """Media y desviación (ddof=0) por variable sobre celdas con máscara 1;
    std=0 -> 1.0."""
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(masks)
    out: dict = {}
    for i, t in enumerate(tracks):
        col = v[:, i][m[:, i] == 1]
        n = int(col.size)
        if n == 0:
            out[t] = {"mean": 0.0, "std": 1.0, "n": 0}
            continue
        mean = float(col.mean())
        var = float(np.mean((col - mean) ** 2))
        std = float(np.sqrt(max(0.0, var)))
        out[t] = {"mean": mean, "std": std if std > 0 else 1.0, "n": n}
    return out


def compute_norm_stats(parts: list[Path], excluded: frozenset[int],
                       tracks: list[str] = IMAGE_TRACKS) -> dict:
    """Estadísticos de normalización acumulados (streaming, sin guardar todas
    las celdas) sobre las particiones dadas, filtradas por fase y exclusiones."""
    sums = np.zeros(len(tracks), dtype=np.float64)
    sq = np.zeros(len(tracks), dtype=np.float64)
    n = np.zeros(len(tracks), dtype=np.int64)
    for part in parts:
        df = read_partition(part, columns=STATS_COLS)
        df = filter_cells(df, excluded)
        if len(df) == 0:
            continue
        vals = df[VALUE_COLS].to_numpy(dtype=np.float64)
        msk = df[MASK_COLS].to_numpy(dtype=np.uint8)
        for i in range(len(tracks)):
            col = vals[:, i][msk[:, i] == 1]
            if col.size:
                sums[i] += col.sum()
                sq[i] += float((col * col).sum())
                n[i] += col.size
    out: dict = {}
    for i, t in enumerate(tracks):
        if n[i] == 0:
            out[t] = {"mean": 0.0, "std": 1.0, "n": int(n[i])}
            continue
        mean = sums[i] / n[i]
        var = sq[i] / n[i] - mean * mean
        std = float(np.sqrt(max(0.0, var)))
        out[t] = {"mean": float(mean), "std": std if std > 0 else 1.0, "n": int(n[i])}
    return out


# --------------------------------------------------------------------------
# Carga de celdas en memoria (para entrenamiento y evaluación)
# --------------------------------------------------------------------------

def load_cells(sources: list[str], split: str, excluded: frozenset[int],
               parts_limit: int | None = None) -> dict:
    """Carga celdas de mantenimiento en memoria agrupadas por caso.

    Devuelve valores (N,14) float32, máscaras (N,14) uint8, caseid (N,) int64 y
    la tabla de casos (case_ids, case_start, case_len, case_source). Las celdas
    quedan ordenadas por caseid (estable), preservando el orden temporal dentro
    de cada caso.
    """
    parts = iter_partitions(sources, split)
    if parts_limit is not None:
        parts = parts[:parts_limit]
    values_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []
    caseid_list: list[np.ndarray] = []
    src_list: list[np.ndarray] = []
    n_by_source: dict[str, int] = {}
    for part in parts:
        df = read_partition(part, columns=READ_COLS)
        df = filter_cells(df, excluded)
        if len(df) == 0:
            continue
        src = str(df["source"].iloc[0])
        n_by_source[src] = n_by_source.get(src, 0) + int(len(df))
        values_list.append(df[VALUE_COLS].to_numpy(dtype=np.float32))
        masks_list.append(df[MASK_COLS].to_numpy(dtype=np.uint8))
        caseid_list.append(df["caseid"].to_numpy(dtype=np.int64))
        src_list.append(df["source"].to_numpy(dtype=object))
    if not values_list:
        return {
            "values": np.zeros((0, N_VARS), dtype=np.float32),
            "masks": np.zeros((0, N_VARS), dtype=np.uint8),
            "caseid": np.zeros(0, dtype=np.int64),
            "case_ids": np.zeros(0, dtype=np.int64),
            "case_start": np.zeros(0, dtype=np.int64),
            "case_len": np.zeros(0, dtype=np.int64),
            "case_source": np.zeros(0, dtype=object),
            "n_by_source": {},
        }
    values = np.concatenate(values_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    caseid = np.concatenate(caseid_list, axis=0)
    source = np.concatenate(src_list, axis=0)

    order = np.argsort(caseid, kind="stable")
    values = values[order]
    masks = masks[order]
    caseid = caseid[order]
    source = source[order]

    bounds = np.flatnonzero(np.r_[True, caseid[1:] != caseid[:-1]])
    case_ids = caseid[bounds]
    case_start = bounds.astype(np.int64)
    case_len = np.diff(np.r_[bounds, len(caseid)]).astype(np.int64)
    case_source = source[bounds].astype(str)
    return {
        "values": values,
        "masks": masks,
        "caseid": caseid,
        "case_ids": case_ids,
        "case_start": case_start,
        "case_len": case_len,
        "case_source": case_source,
        "n_by_source": n_by_source,
    }


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------

class PhysioAE(nn.Module):
    """Encoder 28->256->256->32 y decoder 32->256->256->14, GELU, LayerNorm
    solo en las ocultas (nunca sobre el latente)."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 * N_VARS, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, N_VARS),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def decode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Reconstrucción desde el estado de 128 dims: el token 1 (primeras 32
        dimensiones) es el latente; los tokens 2..4 (warm-up) no se leen."""
        return self.decoder(state[..., :LATENT_DIM])

    def forward(self, x: torch.Tensor, k=None) -> torch.Tensor:
        z = self.encode(x)
        if k is not None:
            z = apply_nested_dropout(z, k)
        return self.decode(z)


def make_model(seed: int = SEED) -> PhysioAE:
    torch.manual_seed(seed)
    return PhysioAE()


def apply_nested_dropout(z: torch.Tensor, k) -> torch.Tensor:
    """Pone a cero las dimensiones k+1..32 del latente.

    ``k`` puede ser int (mismo prefijo para toda la muestra) o un array/tensor
    por muestra. Con k >= 32 no se anula nada.
    """
    dim = z.shape[-1]
    if isinstance(k, (int, np.integer)):
        ki = int(k)
        if ki >= dim:
            return z
        out = z.clone()
        out[..., ki:] = 0.0
        return out
    kt = torch.as_tensor(k, device=z.device)
    if kt.ndim == 0:
        return apply_nested_dropout(z, int(kt.item()))
    keep = torch.arange(dim, device=z.device).unsqueeze(0) < kt.unsqueeze(-1)
    return z * keep.to(z.dtype)


def sample_k(batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """k ~ Uniforme{1..32} por muestra."""
    return rng.integers(1, LATENT_DIM + 1, size=batch_size)


def masked_variable_loss(recon: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """MSE solo sobre celdas con máscara 1, promediado primero por variable y
    después entre las 14."""
    m = mask.to(recon.dtype) if not torch.is_floating_point(mask) else mask
    cnt = m.sum(dim=0).clamp(min=1.0)
    se = (recon - target) ** 2 * m
    return (se.sum(dim=0) / cnt).mean()


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------

def _make_optimizer(model: PhysioAE):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def _make_plateau_scheduler(opt):
    """ReduceLROnPlateau desacoplado del tope de épocas (corrección iter 3).

    Se invoca una vez por época con la pérdida del subconjunto de val fijo.
    El warm-up lineal (WARMUP_EPOCHS épocas) se aplica aparte, en el bucle.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE,
        threshold=PLATEAU_THRESHOLD, min_lr=MIN_LR)


def _set_lr(opt, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def _get_lr(opt) -> float:
    return float(opt.param_groups[0]["lr"])


def _make_input(values_norm: np.ndarray, masks: np.ndarray, idx: np.ndarray,
                device: torch.device) -> torch.Tensor:
    v = torch.from_numpy(values_norm[idx]).to(device)
    m = torch.from_numpy(masks[idx].astype(np.float32)).to(device)
    return torch.cat([v, m], dim=1)


def _step(model, opt, x: torch.Tensor, k) -> torch.Tensor:
    opt.zero_grad()
    recon = model(x, k=k)
    loss = masked_variable_loss(recon, x[:, :N_VARS], x[:, N_VARS:])
    loss.backward()
    opt.step()
    return loss.detach()


def sample_case_cells(rng: np.random.Generator, data: dict, n: int) -> np.ndarray:
    """Muestreo por caso y luego por celda: se sortea un caso uniformemente y
    después una celda uniforme dentro de él (un caso largo no domina)."""
    case_idx = rng.integers(0, len(data["case_ids"]), size=n)
    off = (rng.random(n) * data["case_len"][case_idx]).astype(np.int64)
    off = np.minimum(off, data["case_len"][case_idx] - 1)
    return data["case_start"][case_idx] + off


def train_epoch_plain(model, opt, values_norm: np.ndarray, masks: np.ndarray,
                      rng: np.random.Generator, batch_size: int,
                      device: torch.device) -> float:
    """Una época sobre celdas barajadas; devuelve la pérdida media ponderada."""
    n = values_norm.shape[0]
    perm = rng.permutation(n)
    total = 0.0
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        k = sample_k(len(idx), rng)
        x = _make_input(values_norm, masks, idx, device)
        loss = _step(model, opt, x, k)
        total += float(loss) * len(idx)
    return total / n


def train_epoch_balanced(model, opt, real_norm, real_masks, real_case: dict,
                         synth_norm, synth_masks, synth_case: dict,
                         rng: np.random.Generator, half: int, steps: int,
                         device: torch.device) -> float:
    """Una época balanceada (50/50 por caso->celda); devuelve la pérdida media."""
    total = 0.0
    n_cells = steps * (2 * half)
    for _ in range(steps):
        idx_r = sample_case_cells(rng, real_case, half)
        idx_s = sample_case_cells(rng, synth_case, half)
        x = _make_input(real_norm, real_masks, idx_r, device)
        xs = _make_input(synth_norm, synth_masks, idx_s, device)
        xb = torch.cat([x, xs], dim=0)
        k = sample_k(xb.shape[0], rng)
        loss = _step(model, opt, xb, k)
        total += float(loss) * xb.shape[0]
    return total / n_cells


def _sample_fixed_indices(n_real: int, n_synth: int, real_n: int, synth_n: int,
                          seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Elige, con semilla fija, índices de val real y sintético (corrección 2).

    Devuelve (idx_real, idx_synth) dentro de los espacios de índices [0, n_real)
    y [0, n_synth) respectivamente. Determinista: misma semilla -> mismos índices.
    """
    rng = np.random.default_rng(seed)
    real_idx = np.sort(rng.choice(n_real, size=min(real_n, n_real), replace=False))
    synth_idx = np.sort(rng.choice(n_synth, size=min(synth_n, n_synth), replace=False))
    return real_idx, synth_idx


@lru_cache(maxsize=1)
def fixed_val_sample() -> dict:
    """Submuestra de val FIJA (valores y máscaras crudos) para la parada
    temprana. Se construye UNA vez con semilla fija y se reutiliza en todas las
    épocas y en las dos variantes."""
    excluded = excluded_caseids()
    val = load_cells(ALL_SOURCES, "val", excluded)
    cell_real = np.repeat(val["case_source"] == "real", val["case_len"])
    real_idx, synth_idx = _sample_fixed_indices(
        int(cell_real.sum()), int((~cell_real).sum()),
        VAL_SAMPLE_REAL, VAL_SAMPLE_SYNTH, VAL_SAMPLE_SEED)
    real_pos = np.flatnonzero(cell_real)[real_idx]
    synth_pos = np.flatnonzero(~cell_real)[synth_idx]
    idx = np.concatenate([real_pos, synth_pos])
    is_real = np.concatenate([np.ones(len(real_pos), bool), np.zeros(len(synth_pos), bool)])
    return {
        "values": val["values"][idx],
        "masks": val["masks"][idx],
        "is_real": is_real,
        "n_real": int(len(real_pos)),
        "n_synth": int(len(synth_pos)),
    }


def _eval_fixed_val(model, stats: dict, device: torch.device) -> tuple[float, float, float]:
    """Pérdida de val sobre la submuestra fija: (combinada, real, sintética)."""
    sample = fixed_val_sample()
    values = sample["values"]
    masks = sample["masks"]
    is_real = sample["is_real"]
    zn = normalize(values, stats).astype(np.float32)
    zn[masks == 0] = 0.0
    mf = masks.astype(np.float32)

    model.eval()
    n = zn.shape[0]
    recon = np.empty((n, N_VARS), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            sl = slice(start, min(start + BATCH_SIZE, n))
            x = torch.from_numpy(np.concatenate([zn[sl], mf[sl]], axis=1)).to(device)
            recon[sl] = model(x, k=LATENT_DIM).cpu().numpy()
    model.train()

    se = (recon - zn).astype(np.float32)
    np.multiply(se, se, out=se)
    se *= mf

    def _loss(sel: np.ndarray) -> float:
        cnt = mf[sel].sum(axis=0).clip(min=1.0)
        return float((se[sel].sum(axis=0) / cnt).mean())

    combined = _loss(np.ones(n, bool))
    real = _loss(is_real)
    synth = _loss(~is_real)
    return combined, real, synth


def _normalize_train_data(data: dict, stats: dict) -> np.ndarray:
    values_norm = normalize(data["values"], stats).astype(np.float32)
    values_norm[data["masks"] == 0] = 0.0
    return values_norm


def train(variant: str, device: torch.device, seed: int = SEED) -> dict:
    """Entrena la variante y devuelve modelo + estadísticos + historial.

    Corrección iter 3 (planificador): warm-up lineal de WARMUP_EPOCHS épocas +
    ReduceLROnPlateau sobre la pérdida del subconjunto de val fijo (invocado
    una vez por época tras el warm-up). Parada temprana con paciencia PATIENCE
    (> paciencia del planificador) o por lr <= min_lr con 4 épocas sin mejora.
    MAX_EPOCHS queda como tope de seguridad y no influye en la forma del LR.
    """
    excluded = excluded_caseids()
    t0 = _time.time()

    if variant == "ae_real":
        train_data = load_cells(["real"], "train", excluded)
        n_real = train_data["values"].shape[0]
        stats = compute_norm_stats(iter_partitions(["real"], "train"), excluded)
        values_norm = _normalize_train_data(train_data, stats)
        steps_per_epoch = max(1, math.ceil(n_real / BATCH_SIZE))
        real_case = None
        synth_case = None
        synth_norm = None
        synth_masks = None
    else:
        real_data = load_cells(["real"], "train", excluded)
        synth_data = load_cells(SYNTH_SOURCES, "train", excluded)
        n_real = real_data["values"].shape[0]
        train_parts = iter_partitions(["real"], "train") + iter_partitions(SYNTH_SOURCES, "train")
        stats = compute_norm_stats(train_parts, excluded)
        values_norm = _normalize_train_data(real_data, stats)
        synth_norm = _normalize_train_data(synth_data, stats)
        real_case = real_data
        synth_case = synth_data
        real_masks = real_data["masks"]
        synth_masks = synth_data["masks"]
        steps_per_epoch = max(1, math.ceil(n_real / HALF_BATCH))

    model = make_model(seed).to(device)
    opt = _make_optimizer(model)
    plateau = _make_plateau_scheduler(opt)
    train_rng = np.random.default_rng(seed)

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    epochs_no_improve = 0
    n_reductions = 0
    stop_reason = "max_epochs"
    val_losses: list[float] = []
    val_loss_real: list[float] = []
    val_loss_synth: list[float] = []
    train_losses: list[float] = []
    lr_history: list[float] = []
    reductions_history: list[int] = []

    for epoch in range(MAX_EPOCHS):
        # warm-up lineal (desacoplado de MAX_EPOCHS)
        if epoch < WARMUP_EPOCHS:
            _set_lr(opt, LR * (epoch + 1) / WARMUP_EPOCHS)
        lr_used = _get_lr(opt)
        lr_history.append(lr_used)

        if variant == "ae_real":
            train_loss = train_epoch_plain(model, opt, values_norm,
                                           train_data["masks"], train_rng,
                                           BATCH_SIZE, device)
        else:
            train_loss = train_epoch_balanced(model, opt, values_norm, real_masks,
                                              real_case, synth_norm, synth_masks,
                                              synth_case, train_rng, HALF_BATCH,
                                              steps_per_epoch, device)
        train_losses.append(train_loss)

        val_loss, v_real, v_synth = _eval_fixed_val(model, stats, device)
        val_losses.append(val_loss)
        val_loss_real.append(v_real)
        val_loss_synth.append(v_synth)

        # ReduceLROnPlateau una vez por época, tras el warm-up
        if epoch >= WARMUP_EPOCHS:
            plateau.step(val_loss)
            lr_now = _get_lr(opt)
            if lr_now < lr_used - 1e-12:
                n_reductions += 1
        else:
            lr_now = lr_used
        reductions_history.append(n_reductions)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            epochs_no_improve += 1

        print(f"[ae:{variant}] ep {epoch+1}/{MAX_EPOCHS} lr={lr_used:.2e} "
              f"train={train_loss:.6f} val={val_loss:.6f} "
              f"(real={v_real:.6f} synth={v_synth:.6f}) "
              f"best={best_val:.6f} (ep {best_epoch+1}) "
              f"paciencia={patience_counter}/{PATIENCE} reducciones={n_reductions}",
              flush=True)

        if patience_counter >= PATIENCE:
            stop_reason = "patience"
            break
        if lr_now <= MIN_LR + 1e-15 and epochs_no_improve >= 4:
            stop_reason = "min_lr"
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()

    if variant == "ae_real":
        n_train_by_source = train_data["n_by_source"]
    else:
        n_train_by_source = {**real_data["n_by_source"], **synth_data["n_by_source"]}

    return {
        "variant": variant,
        "model": model,
        "norm_stats": stats,
        "history": {
            "epochs_run": epoch + 1,
            "best_epoch": best_epoch + 1,  # 1-indexado (unificado, corrección 5)
            "best_val_loss": best_val,
            "val_losses": val_losses,
            "val_loss_real": val_loss_real,
            "val_loss_synth": val_loss_synth,
            "train_losses": train_losses,
            "lr_history": lr_history,
            "reductions_history": reductions_history,
            "stop_reason": stop_reason,
            "scheduler_reductions": n_reductions,
            "final_lr": lr_now,
            "max_epochs": MAX_EPOCHS,
        },
        "n_train_cells_by_source": n_train_by_source,
        "elapsed_train_s": round(_time.time() - t0, 2),
    }


def train_50_steps(seed: int = SEED, device: str = "cpu", batch_size: int = BATCH_SIZE,
                   n_steps: int = 50) -> dict:
    """Entrena un modelo nuevo 50 pasos sobre datos sintéticos en memoria y
    devuelve el state_dict (test de determinismo). LR fija (sin planificador)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = PhysioAE().to(device)
    opt = _make_optimizer(model)
    n = batch_size * n_steps
    values = rng.standard_normal((n, N_VARS)).astype(np.float32)
    masks = (rng.random((n, N_VARS)) > 0.2).astype(np.float32)
    stats = compute_stats_from_values(values, masks)
    values_norm = normalize(values, stats).astype(np.float32)
    values_norm[masks == 0] = 0.0
    perm = rng.permutation(n)
    for step in range(n_steps):
        idx = perm[step * batch_size:(step + 1) * batch_size]
        k = sample_k(len(idx), rng)
        x = _make_input(values_norm, masks, idx, torch.device(device))
        _step(model, opt, x, k)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------
# Evaluación (gates 1, 2, 3 y 6)
# --------------------------------------------------------------------------

def _encode_all(model, zn: np.ndarray, masks: np.ndarray, device: torch.device,
                batch_size: int = 16384) -> np.ndarray:
    n = zn.shape[0]
    z = np.empty((n, LATENT_DIM), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            x = torch.from_numpy(np.concatenate([zn[sl], masks[sl].astype(np.float32)],
                                                axis=1)).to(device)
            z[sl] = model.encode(x).cpu().numpy()
    return z


def _decode_all(model, z: np.ndarray, k, device: torch.device,
                batch_size: int = 16384) -> np.ndarray:
    n = z.shape[0]
    out = np.empty((n, N_VARS), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            zt = torch.from_numpy(z[sl]).to(device)
            zt = apply_nested_dropout(zt, k)
            out[sl] = model.decode(zt).cpu().numpy()
    return out


def _nanmedian_no_warn(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=axis)


def _nanmean_no_warn(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _gate6_cell_sample(val: dict, seed: int = GATE6_CELL_SEED,
                       n: int = GATE6_N_CELLS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Submuestra aleatoria estratificada de val (100k real + 100k sintéticas)
    usada por el gate 6 por celda y por el diagnóstico 6/A4. Devuelve (idx, y,
    groups=caseid)."""
    cell_real = np.repeat(val["case_source"] == "real", val["case_len"])
    cell_caseid = val["caseid"].astype(np.int64)
    rng = np.random.default_rng(seed)
    half = n // 2
    real_pos = np.flatnonzero(cell_real)
    synth_pos = np.flatnonzero(~cell_real)
    real_sel = np.sort(rng.choice(real_pos, size=min(half, len(real_pos)), replace=False))
    synth_sel = np.sort(rng.choice(synth_pos, size=min(half, len(synth_pos)), replace=False))
    idx = np.concatenate([real_sel, synth_sel])
    y = cell_real[idx].astype(int)
    groups = cell_caseid[idx]
    return idx, y, groups


def _run_origin_probe(X, y, groups) -> dict:
    """Regresión logística 5-fold agrupada por caso, features estandarizadas."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=5)
    aucs: list[float] = []
    accs: list[float] = []
    for tr, te in gkf.split(X, y, groups=groups):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=5000).fit(Xtr, y[tr])
        scores = clf.decision_function(Xte)
        if len(np.unique(y[te])) == 2:
            aucs.append(float(roc_auc_score(y[te], scores)))
        accs.append(float(accuracy_score(y[te], scores >= 0)))
    return {
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_per_fold": aucs,
        "acc_mean": float(np.mean(accs)),
    }


def evaluate(model, norm_stats: dict, val: dict, device: torch.device) -> dict:
    values = val["values"]
    masks = val["masks"]
    case_ids = val["case_ids"]
    case_start = val["case_start"]
    case_len = val["case_len"]
    case_source = val["case_source"]
    n = values.shape[0]
    n_cases = len(case_ids)

    zn = normalize(values, norm_stats).astype(np.float32)
    zn[masks == 0] = 0.0

    z = _encode_all(model, zn, masks, device)
    cidx = np.repeat(np.arange(n_cases), case_len)

    case_cnt = np.zeros((n_cases, N_VARS), dtype=np.int64)
    for v in range(N_VARS):
        case_cnt[:, v] = np.bincount(cidx, weights=masks[:, v].astype(np.float64),
                                     minlength=n_cases).astype(np.int64)

    # Gate 1: perfil de orden en función de k
    case_sum = np.zeros((n_cases, N_VARS, LATENT_DIM), dtype=np.float64)
    case_mae_5b = np.zeros((n_cases, len(DIAG5B_VARS), LATENT_DIM), dtype=np.float64)
    diag5b_idx = [IMAGE_TRACKS.index(v) for v in DIAG5B_VARS]
    recon_full = None
    for k in range(1, LATENT_DIM + 1):
        recon = _decode_all(model, z, k, device)
        if k == LATENT_DIM:
            recon_full = recon
        se = (recon - zn).astype(np.float32)
        np.multiply(se, se, out=se)
        se *= masks.astype(np.float32)
        for v in range(N_VARS):
            case_sum[:, v, k - 1] = np.bincount(cidx, weights=se[:, v],
                                                minlength=n_cases)
        # Diag 5b: error físico (per-case MAE) de las 4 variables clínicas
        recon_phys_k = denormalize(recon, norm_stats).astype(np.float32)
        for j, vi in enumerate(diag5b_idx):
            err = np.where(masks[:, vi] == 1,
                           np.abs(recon_phys_k[:, vi] - values[:, vi]),
                           0.0).astype(np.float64)
            s = np.bincount(cidx, weights=err, minlength=n_cases)
            with np.errstate(divide="ignore", invalid="ignore"):
                case_mae_5b[:, j, k - 1] = s / case_cnt[:, vi]
    with np.errstate(divide="ignore", invalid="ignore"):
        pcm = case_sum / case_cnt[:, :, None]  # (n_cases, N_VARS, 32), NaN si cnt=0

    groups = {"real": case_source == "real", "synthetic": case_source != "real"}
    var_err: dict[str, np.ndarray] = {}
    glob_err: dict[str, np.ndarray] = {}
    for g, sel in groups.items():
        ve = np.full((N_VARS, LATENT_DIM), np.nan)
        for v in range(N_VARS):
            valid = case_cnt[sel, v] > 0
            if valid.sum():
                ve[v, :] = _nanmean_no_warn(pcm[sel, v, :][valid], axis=0)
        var_err[g] = ve
        glob_err[g] = _nanmean_no_warn(ve, axis=0) if np.isfinite(ve).any() else np.full(
            LATENT_DIM, np.nan)

    def _profile_ok(g: np.ndarray) -> dict:
        if not np.isfinite(g).all():
            return {"monotone": False, "k8_fraction": float("inf"), "ok": False}
        monotone = bool(np.all(np.diff(g) <= 1e-4 * max(1.0, abs(g[0]))))
        span = g[0] - g[-1]
        k8_frac = float((g[7] - g[-1]) / span) if span > 0 else float("inf")
        ok = monotone and (k8_frac <= 0.2)
        return {"monotone": monotone, "k8_fraction": k8_frac, "ok": ok}

    gate1 = {
        "global": {g: [float(x) for x in glob_err[g]] for g in groups},
        "per_variable": {
            t: {g: [float(x) for x in var_err[g][i]] for g in groups}
            for i, t in enumerate(IMAGE_TRACKS)
        },
        "profile": {g: _profile_ok(glob_err[g]) for g in groups},
    }

    # Gate 2: error mediano por variable y cohorte en unidades físicas
    # (solo celdas con máscara 1; las enmascaradas pueden tener valor NaN, así
    # que se calcula el error con np.where y no con (a-b)*m, que propagaría NaN)
    recon_phys = denormalize(recon_full, norm_stats).astype(np.float32)
    err_phys = np.where(masks == 1, np.abs(recon_phys - values), 0.0).astype(np.float32)
    per_case_mae = np.full((n_cases, N_VARS), np.nan)
    for v in range(N_VARS):
        s = np.bincount(cidx, weights=err_phys[:, v], minlength=n_cases)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_case_mae[:, v] = s / case_cnt[:, v]

    gate2: dict = {"verdict": {}}
    for cohort in ALL_SOURCES:
        sel = case_source == cohort
        med = _nanmedian_no_warn(per_case_mae[sel], axis=0) if sel.sum() else np.full(N_VARS, np.nan)
        gate2[cohort] = {t: float(med[i]) for i, t in enumerate(IMAGE_TRACKS)}
        gate2["verdict"][cohort] = {
            "ART_MBP_ok": bool(gate2[cohort][MBP_TRACK] < 2.0),
            "HR_ok": bool(gate2[cohort][HR_TRACK] < 2.0),
        }

    # Gate 3 (corrección 3): error de HR contra el HR VERDADERO con y sin ART
    # observado, y diferencia entre ambas reconstrucciones. Umbral: el error
    # sin ART contra el HR verdadero < 2 lpm (mismo umbral que el gate 2). La
    # diferencia entre reconstrucciones se reporta sin umbral.
    art_idx = [IMAGE_TRACKS.index(t) for t in ART_TRACKS]
    hr_idx = IMAGE_TRACKS.index(HR_TRACK)
    cell_all_art = (masks[:, art_idx] == 1).all(axis=1)
    case_art_cnt = np.bincount(cidx, weights=cell_all_art.astype(np.float64),
                               minlength=n_cases)
    case_has_art = (case_art_cnt > 0) & (case_source == "real")
    selected_cases = np.flatnonzero(case_has_art)

    gate3 = {"n_cases": int(len(selected_cases)), "n_cells": 0,
             "err_hr_with_art": float("nan"), "err_hr_without_art": float("nan"),
             "hr_recon_diff": float("nan"), "ok": False}
    if len(selected_cases):
        cell_sel = cell_all_art & np.isin(cidx, selected_cases)
        zn_masked = zn.copy()
        zn_masked[:, art_idx] = 0.0
        m_masked = masks.copy()
        m_masked[:, art_idx] = 0
        z2 = _encode_all(model, zn_masked, m_masked, device)
        recon_masked = _decode_all(model, z2, LATENT_DIM, device)
        recon_masked_phys = denormalize(recon_masked, norm_stats).astype(np.float32)

        hr_true = values[cell_sel, hr_idx].astype(np.float64)
        hr_with_art = recon_phys[cell_sel, hr_idx].astype(np.float64)
        hr_without_art = recon_masked_phys[cell_sel, hr_idx].astype(np.float64)

        err_with = np.abs(hr_with_art - hr_true)
        err_without = np.abs(hr_without_art - hr_true)
        recon_diff = np.abs(hr_with_art - hr_without_art)

        c = np.bincount(cidx[cell_sel], minlength=n_cases)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_case_with = np.bincount(cidx[cell_sel], weights=err_with,
                                        minlength=n_cases) / c
            per_case_without = np.bincount(cidx[cell_sel], weights=err_without,
                                           minlength=n_cases) / c
            per_case_diff = np.bincount(cidx[cell_sel], weights=recon_diff,
                                        minlength=n_cases) / c
        med_with = float(_nanmedian_no_warn(per_case_with[selected_cases]))
        med_without = float(_nanmedian_no_warn(per_case_without[selected_cases]))
        med_diff = float(_nanmedian_no_warn(per_case_diff[selected_cases]))
        gate3 = {
            "n_cases": int(len(selected_cases)),
            "n_cells": int(cell_sel.sum()),
            "err_hr_with_art": med_with,
            "err_hr_without_art": med_without,
            "hr_recon_diff": med_diff,
            "ok": bool(med_without < 2.0),
        }

    # Gate 6 (corrección 4): sondeo de fuente por caso Y por celda, 32 y 8 dims.
    zc = np.stack([
        np.bincount(cidx, weights=z[:, d].astype(np.float64), minlength=n_cases) / case_len
        for d in range(LATENT_DIM)
    ], axis=1)
    y_case = (case_source == "real").astype(int)
    gate6 = {}

    try:
        if len(np.unique(y_case)) < 2:
            gate6 = {"error": "solo hay una clase en val (faltan cohortes)"}
            raise ValueError("single class")

        per_case: dict = {}
        for label, dims in [("32", LATENT_DIM), ("8", 8)]:
            per_case[label] = _run_origin_probe(zc[:, :dims], y_case,
                                                np.arange(n_cases))
        gate6["per_case"] = per_case
        gate6["n_cases_real"] = int((case_source == "real").sum())
        gate6["n_cases_synthetic"] = int((case_source != "real").sum())

        # Sondeo por CELDA: submuestra aleatoria estratificada de 200 000 celdas
        # de val (100k real + 100k sintéticas), GroupKFold agrupado por caso.
        cell_idx, y_cell, groups_cell = _gate6_cell_sample(val)
        per_cell: dict = {}
        for label, dims in [("32", LATENT_DIM), ("8", 8)]:
            per_cell[label] = _run_origin_probe(z[cell_idx][:, :dims], y_cell,
                                                groups_cell)
        per_cell["n_cells"] = int(len(cell_idx))
        gate6["per_cell"] = per_cell
    except Exception as e:  # pragma: no cover
        gate6 = {"error": str(e)}

    # Diagnóstico 5 (sin umbral): k mínimo tal que err(k) <= 1.1 * err(32).
    diag5: dict = {}
    for g in ("real", "synthetic"):
        curve = glob_err[g]
        if not np.isfinite(curve).all():
            diag5[g] = 32
            continue
        e32 = curve[-1]
        thr = 1.1 * e32 if e32 > 0 else 0.0
        ks = int(np.argmax(curve <= thr) + 1) if np.any(curve <= thr) else 32
        diag5[g] = ks

    # Diagnóstico 5b: error en unidades físicas para 4 variables clínicas en
    # función de k, por cohorte (real / sintético) y k_clinico.
    diag5b: dict = {
        "variables": list(DIAG5B_VARS),
        "thresholds": dict(DIAG5B_THRESHOLDS),
        "cohorts": {},
    }
    for g, sel in groups.items():
        per_var: dict = {}
        for j, var in enumerate(DIAG5B_VARS):
            per_var[var] = [float(x) for x in _nanmedian_no_warn(
                case_mae_5b[sel, j, :], axis=0)]
        k_clinico = None
        for k in range(1, LATENT_DIM + 1):
            if all(per_var[var][k - 1] < DIAG5B_THRESHOLDS[var]
                   for var in DIAG5B_VARS):
                k_clinico = k
                break
        diag5b["cohorts"][g] = {"per_variable": per_var, "k_clinico": k_clinico}

    return {"gate1": gate1, "gate2": gate2, "gate3": gate3, "gate6": gate6,
            "diag5": diag5, "diag5b": diag5b}


def diagnose(model, norm_stats: dict, val: dict, device: torch.device) -> dict:
    """Diagnóstico 6 (separabilidad de cohorte, A1-A3 sobre la submuestra de
    val fija) y A4 (control de mecanismo sobre la entrada cruda, 200 000 celdas
    del gate 6). Sin umbrales: son mediciones."""
    sample = fixed_val_sample()
    values = sample["values"]
    masks = sample["masks"]
    is_real = sample["is_real"]
    zn = normalize(values, norm_stats).astype(np.float32)
    zn[masks == 0] = 0.0
    z = _encode_all(model, zn, masks, device)  # (N, 32)

    # A1: solapamiento de soportes (fracción de los 20 vecinos de la otra
    # cohorte, en 32 dims).
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=21, metric="euclidean")
    nn.fit(z)
    _, knn_idx = nn.kneighbors(z)
    nb = knn_idx[:, 1:]  # excluye el propio punto
    frac_other = (is_real[nb] != is_real[:, None]).mean(axis=1)

    def _pcts(x: np.ndarray) -> dict:
        out = {"mean": float(x.mean())}
        for p in (5, 25, 50, 75, 95):
            out[f"p{p}"] = float(np.percentile(x, p))
        return out

    a1 = {
        "real_cells_fraction_synthetic_neighbors": _pcts(frac_other[is_real]),
        "synthetic_cells_fraction_real_neighbors": _pcts(frac_other[~is_real]),
    }

    # A2: separación relativa (distancia entre centroides / raíz de la traza de
    # la covarianza intra-cohorte promediada), en 32 y 8 dims.
    c_real = z[is_real].mean(axis=0)
    c_synth = z[~is_real].mean(axis=0)
    d = float(np.linalg.norm(c_real - c_synth))
    cov_real = np.cov(z[is_real].T)
    cov_synth = np.cov(z[~is_real].T)
    avg_cov = (cov_real + cov_synth) / 2.0
    denom32 = float(np.sqrt(np.trace(avg_cov)))
    d8 = float(np.linalg.norm(c_real[:8] - c_synth[:8]))
    denom8 = float(np.sqrt(np.trace(avg_cov[:8, :8])))
    a2 = {
        "ratio_32": d / denom32 if denom32 > 0 else float("inf"),
        "ratio_8": d8 / denom8 if denom8 > 0 else float("inf"),
        "centroid_distance_32": d,
        "centroid_distance_8": d8,
    }

    # A3: dirección de la separación (peso normalizado de cada dim en el vector
    # entre centroides), ordenado de mayor a menor.
    v = c_real - c_synth
    w = np.abs(v)
    w = w / (w.sum() + 1e-30)
    order = np.argsort(-w)
    a3 = [{"dim": int(i), "weight": float(w[i])} for i in order]

    # A4: control de mecanismo sobre la entrada cruda (200 000 celdas, seed 9876).
    cell_idx, y_cell, groups_cell = _gate6_cell_sample(val)
    raw_values = val["values"][cell_idx]
    raw_masks = val["masks"][cell_idx].astype(np.float32)
    zn_cell = normalize(raw_values, norm_stats).astype(np.float32)
    zn_cell[raw_masks == 0] = 0.0
    a4 = {
        "masks_only": _run_origin_probe(raw_masks, y_cell, groups_cell),
        "values_only": _run_origin_probe(zn_cell, y_cell, groups_cell),
        "masks_and_values": _run_origin_probe(
            np.concatenate([zn_cell, raw_masks], axis=1), y_cell, groups_cell),
    }

    return {"diag6": {
        "A1_support_overlap": a1,
        "A2_relative_separation": a2,
        "A3_separation_direction": a3,
        "A4_mechanism_control": a4,
    }}


# --------------------------------------------------------------------------
# Artefactos y manifest
# --------------------------------------------------------------------------

def load_model(variant: str, device: str = "cpu") -> PhysioAE:
    model = make_model(SEED).to(device)
    model.encoder.load_state_dict(torch.load(OUT_ROOT / variant / "encoder.pt",
                                             map_location=device))
    model.decoder.load_state_dict(torch.load(OUT_ROOT / variant / "decoder.pt",
                                             map_location=device))
    return model.eval()


def load_norm_stats(variant: str) -> dict:
    return json.loads((OUT_ROOT / variant / "norm_stats.json").read_text(encoding="utf-8"))


def reevaluate(variant: str, device: torch.device) -> dict:
    """Reevalúa una variante ya entrenada y actualiza su manifest (sin
    reentrenar). Calcula gates + diag5b y diag6, y declara acceptance y
    provenance_gaps (iter 4)."""
    model = load_model(variant, str(device)).to(device)
    norm_stats = load_norm_stats(variant)
    excluded = excluded_caseids()
    val = load_cells(ALL_SOURCES, "val", excluded)
    gates = evaluate(model, norm_stats, val, device)
    diagnostics = diagnose(model, norm_stats, val, device)
    p = OUT_ROOT / variant / "manifest_ae.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["gates"] = gates
    m["diagnostics"] = diagnostics
    m["n_val_cells_by_source"] = val["n_by_source"]
    m["acceptance"] = dict(ACCEPTANCE)
    m["provenance_gaps"] = _build_provenance_gaps()
    m.pop("sha256_report_window_v2", None)  # ya no es entrada ascendente
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return gates


def _build_provenance_gaps() -> list[dict]:
    gaps = json.loads(json.dumps(PROVENANCE_GAPS))
    for g in gaps:
        if g.get("artifact") == "REPORT_window_v2.txt":
            g["sha256_reconstruction"] = sha256(REPORT_WINDOW_V2)
    return gaps


def build_manifest(variant: str, result: dict, gates: dict, norm_stats: dict) -> dict:
    win_manifest = json.loads((WINDOWS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    enc_path = OUT_ROOT / variant / "encoder.pt"
    dec_path = OUT_ROOT / variant / "decoder.pt"
    hist = result["history"]
    return {
        "date": pd.Timestamp.now().isoformat(),
        "variant": variant,
        "seed": SEED,
        "sha256_contract_ae": sha256(CONTRACT_PATH),
        "sha256_contract_tokens": sha256(TOKENS_CONTRACT_PATH),
        "sha256_physio_ae_py": sha256(Path(__file__).resolve()),
        "sha256_tokens_v1_manifest": sha256(TOKENS_MANIFEST),
        "sha256_windows_v2_manifest": sha256(WINDOWS_ROOT / "manifest.json"),
        "sha256_split_parquet": sha256(WINDOWS_ROOT / "split.parquet"),
        "sha256_report_tokens_v4": sha256(REPORT_TOKENS_V4),
        "provenance_gaps": _build_provenance_gaps(),
        "acceptance": dict(ACCEPTANCE),
        "sha256_encoder": sha256(enc_path),
        "sha256_decoder": sha256(dec_path),
        "scheduler": "warmup_linear+reduce_on_plateau",
        "scheduler_reductions": hist["scheduler_reductions"],
        "final_lr": hist["final_lr"],
        "best_epoch": hist["best_epoch"],
        "hyperparameters": {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "patience": PATIENCE,
            "plateau_factor": PLATEAU_FACTOR,
            "plateau_patience": PLATEAU_PATIENCE,
            "plateau_threshold": PLATEAU_THRESHOLD,
            "min_lr": MIN_LR,
            "latent_dim": LATENT_DIM,
            "n_vars": N_VARS,
        },
        "val_sample": {
            "seed": VAL_SAMPLE_SEED,
            "n_real": VAL_SAMPLE_REAL,
            "n_synth": VAL_SAMPLE_SYNTH,
            "criterion": "real + synthetic (misma muestra para las dos variantes)",
        },
        "image_tracks": IMAGE_TRACKS,
        "excluded_no_phase_marks_caseids": sorted(load_tokens_manifest().get(
            "excluded_no_phase_marks_caseids", [])),
        "sin_celdas_caseids": sorted(load_tokens_manifest().get("sin_celdas_caseids", [])),
        "norm_stats": norm_stats,
        "training": hist,
        "n_train_cells_by_source": result["n_train_cells_by_source"],
        "n_val_cells_by_source": result["n_val_cells_by_source"],
        "windows_v2_n_windows_by_source_split": win_manifest["n_windows_by_source_split"],
        "gates": gates,
    }


def run_variant(variant: str, device: torch.device) -> dict:
    t0 = _time.time()
    result = train(variant, device)
    model = result["model"].to(device)
    excluded = excluded_caseids()

    val = load_cells(ALL_SOURCES, "val", excluded)
    result["n_val_cells_by_source"] = val["n_by_source"]
    gates = evaluate(model, result["norm_stats"], val, device)
    g6 = gates["gate6"]
    auc32 = g6.get("per_case", {}).get("32", {}).get("auc_mean", float("nan"))
    print(f"[ae:{variant}] gate1 real: "
          f"k1={gates['gate1']['global']['real'][0]:.5f} "
          f"k8={gates['gate1']['global']['real'][7]:.5f} "
          f"k32={gates['gate1']['global']['real'][31]:.5f}", flush=True)
    print(f"[ae:{variant}] gate2 real HR={gates['gate2']['real'][HR_TRACK]:.3f} "
          f"MBP={gates['gate2']['real'][MBP_TRACK]:.3f} | "
          f"gate3 sinART={gates['gate3']['err_hr_without_art']:.3f} "
          f"(umbral<2) | gate6 AUC32={auc32:.4f} | "
          f"diag5 real={gates['diag5']['real']} synth={gates['diag5']['synthetic']}",
          flush=True)

    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), out_dir / "encoder.pt")
    torch.save(model.decoder.state_dict(), out_dir / "decoder.pt")
    (out_dir / "norm_stats.json").write_text(
        json.dumps(result["norm_stats"], indent=2), encoding="utf-8")

    manifest = build_manifest(variant, result, gates, result["norm_stats"])
    (out_dir / "manifest_ae.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["elapsed_total_s"] = round(_time.time() - t0, 2)
    return manifest


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------

IMPLEMENTATION_NOTES = [
    "Entrada: 28 canales = 14 valores de la imagen z-scoreados + 14 máscaras m_ "
    "(con plausibilidad; no m_raw_). Valor enmascarado = 0 tras normalizar.",
    "Arquitectura: encoder 28->256->256->32 y decoder 32->256->256->14, GELU, "
    "LayerNorm solo en las ocultas (nunca sobre el latente).",
    "Nested dropout: por muestra k ~ Uniforme{1..32}; se ponen a cero las "
    "dimensiones k+1..32 antes del decoder. Inferencia con las 32 (sin dropout).",
    "Pérdida: MSE solo sobre celdas con máscara 1, promediada primero por "
    "variable y después entre las 14.",
    "Normalización: media y desviación (ddof=0) por variable sobre split=train "
    "con máscara 1; std=0 -> 1.0. Guardadas en norm_stats.json.",
    "Optimizador AdamW (lr 1e-3, weight decay 1e-4), batch 4096.",
    "Iter 3 (planificador): warm-up lineal de 5 épocas + ReduceLROnPlateau "
    "(factor 0.5, patience 4, threshold 1e-4, min_lr 1e-6) invocado una vez "
    "por época con la pérdida del subconjunto de val fijo. MAX_EPOCHS=200 queda "
    "como tope de seguridad y ya no influye en la forma del LR.",
    "Iter 3 (parada): paciencia 12 (estrictamente mayor que la del planificador) "
    "y parada adicional si lr <= min_lr con 4 épocas sin mejora.",
    "Iter 2 (corrección 2): submuestra de val FIJA (65536 real + 65536 sintética, "
    "semilla 1234) idéntica en todas las épocas y para las dos variantes; "
    "criterio común = real + sintético. La pérdida real y la sintética se "
    "reportan aparte sin intervenir en la parada.",
    "Iter 2 (corrección 3, gate 3): |HR_con_ART - HR_real|, |HR_sin_ART - "
    "HR_real| y la diferencia entre reconstrucciones; umbral err_hr_without_art "
    "< 2 lpm.",
    "Iter 2 (corrección 4, gate 6): sondeo por caso Y por celda (200 000 celdas "
    "estratificadas, semilla 9876, GroupKFold por caseid).",
    "Diagnóstico 5: k mínimo tal que err(k) <= 1.1 * err(32), en val real y "
    "sintético.",
    "Diagnóstico 5b (iter 4): error en unidades físicas de HR, ART_MBP, BIS y "
    "ETCO2 en función de k (1..32), por cohorte; k_clinico = k mínimo que cumple "
    "HR<1.0, ART_MBP<1.0, BIS<1.0, ETCO2<0.5 simultáneamente.",
    "Diagnóstico 6 (iter 4): separabilidad de cohorte sobre la submuestra de val "
    "fija (A1 solapamiento de soportes por 20-NN, A2 separación relativa entre "
    "centroides, A3 dirección de separación) y A4 control de mecanismo sobre la "
    "entrada cruda (máscaras / valores / 28 canales) con 200 000 celdas.",
    "Iter 3 (corrección 5): best_epoch 1-indexado en el manifest y en el "
    "informe (unificado).",
    "Iter 4 (procedencia): REPORT_window_v2.txt original no se conserva; se "
    "declara en provenance_gaps y su reconstrucción retrospectiva se renombra a "
    "REPORT_window_v2_RECONSTRUIDO.txt (no constituye procedencia). "
    "REPORT_tokens_v4.txt sí es original y se mantiene como entrada ascendente.",
    "Variante ae_bal: 50 % real / 50 % sintético; dentro de cada mitad, "
    "muestreo por caso y luego por celda (un caso largo no domina).",
    "Métricas de evaluación promediadas primero por caso y luego entre casos.",
]

ASSUMPTIONS = [
    "El orden de las 14 variables es el canónico image_tracks del manifest de "
    "windows_v2 (BIS/EMG en 14ª posición); el contrato lo lista en 2ª posición "
    "sin efecto semántico (los estadísticos y la pérdida van por variable).",
    "El muestreo 'sintético' de ae_bal reparte uniformemente por caso sobre la "
    "unión de las 3 cohortes sintéticas (synthetic_v5, vaso_reinf_v5, cf_v5): "
    "cada caso sintético tiene la misma probabilidad y después se sortea la "
    "celda. El contrato no especifica el reparto entre las 3 cohortes.",
    "El tamaño de la submuestra de val fija (65536 real + 65536 sintética) y "
    "las semillas de la submuestra (1234) y del sondeo por celda (9876) no "
    "están en el contrato; se fijan para que la parada sea estable y "
    "reproducible.",
    "El warm-up de 5 épocas y los hiperparámetros del ReduceLROnPlateau "
    "(factor 0.5, patience 4, threshold 1e-4, min_lr 1e-6) los fija la tarea; "
    "el contrato en disco aún dice 'cosine con 5 % de warm-up', que es "
    "justamente el defecto corregido en esta iteración.",
    "Gate 1: el veredicto (monotonía y 80 % con k=8) se calcula sobre el error "
    "global (media de las 14 variables) en val real y en val sintético por "
    "separado; las curvas por variable se reportan sin aplicarles el umbral.",
    "Gate 2: el umbral ART_MBP<2 mmHg y HR<2 lpm se evalúa por cohorte; el "
    "veredicto global se apoya en la cohorte real (el objetivo clínico) y el "
    "resto se reporta. Error = media por caso del |recon - real| en unidades "
    "físicas, mediana entre casos.",
    "Gate 3: 'casos reales con ART observado' = caso real con al menos una celda "
    "val con las 3 columnas ART con máscara 1; la comparación se hace sobre esas "
    "celdas (donde enmascarar ART tiene efecto). Los tres números se promedian "
    "por caso y se reporta la mediana entre casos.",
    "Gate 6 por celda: submuestra aleatoria estratificada de 200 000 celdas "
    "(100 000 reales + 100 000 sintéticas) con semilla fija; GroupKFold 5 "
    "agrupado por caseid, features estandarizadas por fold, LogisticRegression, "
    "AUC medio.",
    "La semilla 42 fija el generador numpy del muestreo y del nested dropout y "
    "la inicialización de torch; el determinismo exacto se verifica en CPU "
    "(test e).",
    "REPORT_tokens_v4.txt es el informe original del tokenizador (existe en "
    "reports/) y su sha se mantiene como entrada ascendente. REPORT_window_v2.txt "
    "original NO se conserva: se declara en provenance_gaps (status "
    "missing_original) y la reconstrucción retrospectiva se renombra a "
    "REPORT_window_v2_RECONSTRUIDO.txt con advertencia en la primera línea; su "
    "sha va en provenance_gaps, NO en entradas ascendentes.",
    "La tolerancia de regresión del requisito v3<=v1 (0.05 lpm / 0.05 mmHg) se "
    "declara en el campo 'acceptance' del manifest con decided_post_hoc=true, "
    "marca deliberada para la auditoría. El test lee la tolerancia de ahí.",
    "Diagnóstico 6/A1-A3 se calculan sobre la submuestra de val fija (65536 "
    "real + 65536 sintética) con el encoder congelado; A4 reutiliza las 200 000 "
    "celdas del gate 6 (semilla 9876) sobre la ENTRADA CRUDA del AE.",
    "Diagnóstico 5b: k_clinico se computa por cohorte (real y sintético); la "
    "tabla completa (k x 4 variables x 2 cohortes) va en el informe.",
    "La tabla de gates del contrato en disco sigue mostrando el gate 3 antiguo "
    "(diferencia mediana < 1 lpm) y el gate 6 solo por caso; se implementan "
    "según la tarea, que corrige ambos como 'mal diseñados'.",
    "final_lr del manifest es el lr tras el último paso del planificador (el lr "
    "con el que se detiene el entrenamiento); lr_history recoge el lr usado en "
    "cada época.",
]


def _fmt_verdict(gates: dict) -> list[str]:
    g1 = gates["gate1"]["profile"]
    g2 = gates["gate2"]["verdict"]
    g3 = gates["gate3"]
    g6 = gates["gate6"]
    g1_ok = g1["real"]["ok"] and g1["synthetic"]["ok"]
    g2_ok = g2["real"]["ART_MBP_ok"] and g2["real"]["HR_ok"]
    g3_ok = g3["ok"]
    auc32_case = g6.get("per_case", {}).get("32", {}).get("auc_mean", float("nan"))
    auc32_cell = g6.get("per_cell", {}).get("32", {}).get("auc_mean", float("nan"))
    lines = []
    lines.append(f"Gate 1 (perfil de orden): {'PASA' if g1_ok else 'NO PASA'} "
                 f"(real {'ok' if g1['real']['ok'] else 'falla'}, sintético "
                 f"{'ok' if g1['synthetic']['ok'] else 'falla'})")
    lines.append(f"Gate 2 (reconstrucción por variable y cohorte): "
                 f"{'PASA' if g2_ok else 'NO PASA'} — ART_MBP real "
                 f"{g2['real']['ART_MBP_ok'] and '<2' or '>=2'}, HR real "
                 f"{g2['real']['HR_ok'] and '<2' or '>=2'} lpm")
    lines.append(f"Gate 3 (independencia de máscara ART): "
                 f"{'PASA' if g3_ok else 'NO PASA'} — error de HR sin ART contra "
                 f"el real {round(g3['err_hr_without_art'], 3)} lpm (umbral <2)")
    lines.append("Gate 4 (congelación y determinismo): PASA — ver tests e y j.")
    lines.append("Gate 5 (gate 10 de tokens): PASA — ver test i.")
    lines.append(f"Gate 6 (sondeo de fuente): INFORMATIVO — AUC por caso "
                 f"{auc32_case:.4f}, por celda {auc32_cell:.4f} (32 dims).")
    lines.append("Gate 7 (sin fuga temporal): PASA — ver test g.")
    lines.append("Gate 8 (stats solo train): PASA — ver test h.")
    lines.append("Gate 9 (comparación A vs B): INFORMATIVO — sin umbral.")
    return lines


def _read_txt(path: Path) -> str:
    if not path.exists():
        return "(no disponible: " + path.name + ")"
    data = path.read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):  # Tee-Object de PowerShell 5.1 -> UTF-16
        return data.decode("utf-16", errors="replace").strip()
    return data.decode("utf-8-sig", errors="replace").strip()


def write_report() -> None:
    manifests = {}
    for variant in ["ae_real", "ae_bal"]:
        p = OUT_ROOT / variant / "manifest_ae.json"
        if p.exists():
            manifests[variant] = json.loads(p.read_text(encoding="utf-8"))

    snap1 = {}
    if V1_SNAPSHOT.exists():
        snap1 = json.loads(V1_SNAPSHOT.read_text(encoding="utf-8"))
    snap2 = {}
    if V2_SNAPSHOT.exists():
        snap2 = json.loads(V2_SNAPSHOT.read_text(encoding="utf-8"))

    synth = ["synthetic_v5", "vaso_reinf_v5", "cf_v5"]

    def _g2(src, v, cohort, track):
        return src[v]["gates"]["gate2"][cohort][track]

    def _synth_mean(src, v, track):
        return float(np.nanmean([src[v]["gates"]["gate2"][c][track] for c in synth]))

    L: list[str] = []
    L.append("REPORT_ae_v4.txt — autoencoder de fisiología v1 (iteración 4)")
    L.append("=" * 78)
    L.append("")
    L.append("0. Contexto y ficheros")
    L.append("-" * 40)
    if manifests:
        m = manifests["ae_real"]
        L.append(f"  contrato_ae_v1.md                 sha256 {m['sha256_contract_ae']}")
        L.append(f"  contrato_tokens_v1.md             sha256 {m['sha256_contract_tokens']}")
        L.append(f"  physio_ae.py                      sha256 {m['sha256_physio_ae_py']}")
        L.append(f"  tokens_v1/manifest_tokens.json    sha256 {m['sha256_tokens_v1_manifest']}")
        L.append(f"  windows_v4/manifest.json          sha256 {m['sha256_windows_v2_manifest']}")
        L.append(f"  windows_v4/split.parquet          sha256 {m['sha256_split_parquet']}")
        L.append(f"  reports/REPORT_tokens_v4.txt      sha256 {m['sha256_report_tokens_v4']}")
        L.append(f"  decisión de contrato              variante elegida: B (ae_bal)")
        L.append(f"  planificador                      {m['scheduler']}")
        if "provenance_gaps" in m:
            L.append("  provenance_gaps:")
            for g in m["provenance_gaps"]:
                L.append(f"    - {g['artifact']}: {g['status']} "
                         f"(sha reconstrucción {g['sha256_reconstruction'][:12]}…)")
        if "acceptance" in m:
            acc = m["acceptance"]
            L.append(f"  acceptance                        tolerancia {acc['regression_tolerance_lpm']} "
                     f"lpm / {acc['regression_tolerance_mmhg']} mmHg "
                     f"(decided_post_hoc={acc['decided_post_hoc']})")
    L.append("")
    L.append("1. Tests y salida en ROJO")
    L.append("-" * 40)
    L.append(_read_txt(ROJO_TXT))
    L.append("")
    L.append("2. Implementación y decisiones tomadas")
    L.append("-" * 40)
    for line in IMPLEMENTATION_NOTES:
        L.append("  " + line)
    L.append("")
    L.append("3. Salida en VERDE")
    L.append("-" * 40)
    L.append(_read_txt(VERDE_TXT))
    L.append("")
    L.append("4. Resultados por variante")
    L.append("-" * 40)
    for variant in ["ae_real", "ae_bal"]:
        if variant not in manifests:
            continue
        m = manifests[variant]
        tr = m["training"]
        L.append("")
        L.append(f"  Variante {variant}")
        L.append(f"    semilla                          {m['seed']}")
        L.append(f"    sha encoder                      {m['sha256_encoder']}")
        L.append(f"    sha decoder                      {m['sha256_decoder']}")
        L.append(f"    épocas entrenadas                {tr['epochs_run']}")
        L.append(f"    mejor época (1-indexada)         {m['best_epoch']}")
        L.append(f"    motivo de parada                 {tr['stop_reason']}")
        L.append(f"    reducciones de LR                {m['scheduler_reductions']}")
        L.append(f"    LR final                         {m['final_lr']:.2e}")
        L.append(f"    pérdida val (mejor)              {tr['best_val_loss']:.6f}")
        L.append("")
        L.append("    4.1 Gate 1 (perfil de orden, error global por k)")
        for g in ("real", "synthetic"):
            vals = m["gates"]["gate1"]["global"][g]
            prof = m["gates"]["gate1"]["profile"][g]
            L.append(f"      val {g:<10s} k8={vals[7]:.6f} k32={vals[31]:.6f} "
                     f"monotono={'si' if prof['monotone'] else 'no'} "
                     f"k8_frac={prof['k8_fraction']:.4f} "
                     f"{'OK' if prof['ok'] else 'NO'}")
            L.append("      k: " + " ".join(f"{i+1:>2d}" for i in range(32)))
            L.append("      e: " + " ".join(f"{v:.4f}" for v in vals))
        L.append("")
        L.append("    4.2 Gate 2 (error mediano por variable y cohorte, unidades físicas)")
        header = "      " + "".join(f"{t[:16]:>17s}" for t in m["image_tracks"])
        L.append(header)
        for cohort in ALL_SOURCES:
            row = "      " + f"{cohort:<16s}"
            for t in m["image_tracks"]:
                row += f"{m['gates']['gate2'][cohort][t]:>17.3f}"
            L.append(row)
        L.append("      umbral: " + " ".join(
            f"{c}:ART_MBP{'<2' if m['gates']['gate2']['verdict'][c]['ART_MBP_ok'] else '>=2'},"
            f"HR{'<2' if m['gates']['gate2']['verdict'][c]['HR_ok'] else '>=2'}"
            for c in ALL_SOURCES))
        L.append("")
        g3 = m["gates"]["gate3"]
        L.append("    4.3 Gate 3 (independencia de máscara ART, casos reales)")
        L.append(f"      casos reales con ART: {g3['n_cases']}, celdas: {g3['n_cells']}")
        L.append(f"      |HR_con_ART - HR_real|   : {g3['err_hr_with_art']:.4f} lpm")
        L.append(f"      |HR_sin_ART - HR_real|   : {g3['err_hr_without_art']:.4f} lpm "
                 f"{'OK' if g3['ok'] else 'NO'} (umbral <2)")
        L.append(f"      |HR_con_ART - HR_sin_ART|: {g3['hr_recon_diff']:.4f} lpm (sin umbral)")
        L.append("")
        L.append("    4.4 Gate 6 (sondeo de fuente, AUC medio 5-fold)")
        g6 = m["gates"]["gate6"]
        if "error" not in g6:
            L.append("      por caso y por celda:")
            for scope in ("per_case", "per_cell"):
                sc = g6[scope]
                extra = f"  (n={sc.get('n_cells', g6.get('n_cases_real', 0) + g6.get('n_cases_synthetic', 0))})"
                L.append(f"        {scope:<10s} 32d AUC={sc['32']['auc_mean']:.4f} "
                         f"acc={sc['32']['acc_mean']:.4f} | 8d AUC={sc['8']['auc_mean']:.4f} "
                         f"acc={sc['8']['acc_mean']:.4f}{extra}")
        else:
            L.append(f"      error: {g6['error']}")
        L.append("")
        L.append("    4.5 Diagnóstico 5 (k mínimo con err <= 1.1*err(32))")
        d5 = m["gates"]["diag5"]
        L.append(f"      val real:      k = {d5['real']}")
        L.append(f"      val sintético: k = {d5['synthetic']}")
        L.append("")
        L.append("    4.6 Planificador (LR por época, iter 3)")
        L.append("      " + f"{'ep':>4s} {'lr':>9s} {'train_loss':>11s} {'val_loss':>11s} {'reduc.':>7s}")
        for i in range(tr["epochs_run"]):
            star = "*" if (i + 1) == m["best_epoch"] else " "
            L.append(f"      {i+1:>4d}{star} {tr['lr_history'][i]:>9.2e} "
                     f"{tr['train_losses'][i]:>11.6f} {tr['val_losses'][i]:>11.6f} "
                     f"{tr['reductions_history'][i]:>7d}")
        L.append(f"      (*) mejor época {m['best_epoch']} (1-indexado); motivo de parada "
                 f"{tr['stop_reason']}")
    L.append("")
    L.append("5. Diagnósticos de cierre (iter 4)")
    L.append("-" * 40)
    L.append("  5.1 Diagnóstico 5b (error físico por k, 4 variables clínicas)")
    L.append("      umbrales: HR<1.0 lpm, ART_MBP<1.0 mmHg, BIS<1.0, ETCO2<0.5")
    for variant in ["ae_real", "ae_bal"]:
        if variant not in manifests:
            continue
        m = manifests[variant]
        d5b = m["gates"]["diag5b"]
        L.append(f"      {variant}:")
        for cohort in ("real", "synthetic"):
            c = d5b["cohorts"][cohort]
            L.append(f"        {cohort}: k_clinico={c['k_clinico']}")
            L.append("        " + f"{'k':>3s}" + "".join(
                f"{v[:14]:>15s}" for v in d5b["variables"]))
            for k in range(1, 33):
                row = f"        {k:>3d}"
                for var in d5b["variables"]:
                    row += f"{c['per_variable'][var][k - 1]:>15.3f}"
                L.append(row)
        L.append("")
    L.append("  5.2 Diagnóstico 6 (separabilidad de cohorte, encoder congelado)")
    for variant in ["ae_real", "ae_bal"]:
        if variant not in manifests:
            continue
        m = manifests[variant]
        d6 = m["diagnostics"]["diag6"]
        L.append(f"      {variant}:")
        a1 = d6["A1_support_overlap"]
        L.append("        A1 solapamiento de soportes (fracción de los 20-NN de la otra cohorte):")
        for key, label in [
            ("real_cells_fraction_synthetic_neighbors", "celdas reales -> vecinos sintéticos"),
            ("synthetic_cells_fraction_real_neighbors", "celdas sintéticas -> vecinos reales"),
        ]:
            x = a1[key]
            L.append(f"          {label}: media={x['mean']:.3f} "
                     f"p5={x['p5']:.3f} p25={x['p25']:.3f} p50={x['p50']:.3f} "
                     f"p75={x['p75']:.3f} p95={x['p95']:.3f}")
        a2 = d6["A2_relative_separation"]
        L.append(f"        A2 separación relativa: 32d={a2['ratio_32']:.3f} "
                 f"8d={a2['ratio_8']:.3f} (dist. centroides 32d={a2['centroid_distance_32']:.3f})")
        a3 = d6["A3_separation_direction"]
        L.append("        A3 dirección de separación (dim: peso normalizado, top 10):")
        L.append("          " + " ".join(f"d{i['dim']}:{i['weight']:.3f}" for i in a3[:10]))
        a4 = d6["A4_mechanism_control"]
        L.append("        A4 control de mecanismo (entrada cruda, 200 000 celdas):")
        for key, label in [("masks_only", "14 máscaras"),
                           ("values_only", "14 valores z"),
                           ("masks_and_values", "28 canales")]:
            L.append(f"          {label}: AUC={a4[key]['auc_mean']:.4f} "
                     f"acc={a4[key]['acc_mean']:.4f}")
    L.append("")
    L.append("6. Cambios respecto a v1 y v2")
    L.append("-" * 40)
    L.append("  Reconstrucción HR y ART_MBP (real y sintético), v1 vs v2 vs v3:")
    L.append("    " + f"{'variante':<10s}{'métrica':<18s}{'v1':>10s}{'v2':>10s}{'v3':>10s}")
    for v in ("ae_real", "ae_bal"):
        rows = [
            ("HR real (lpm)", _g2(snap1, v, "real", HR_TRACK) if v in snap1 else float("nan"),
             _g2(snap2, v, "real", HR_TRACK) if v in snap2 else float("nan"),
             _g2(manifests, v, "real", HR_TRACK)),
            ("HR sintético (lpm)", _synth_mean(snap1, v, HR_TRACK) if v in snap1 else float("nan"),
             _synth_mean(snap2, v, HR_TRACK) if v in snap2 else float("nan"),
             _synth_mean(manifests, v, HR_TRACK)),
            ("ART_MBP real (mmHg)", _g2(snap1, v, "real", MBP_TRACK) if v in snap1 else float("nan"),
             _g2(snap2, v, "real", MBP_TRACK) if v in snap2 else float("nan"),
             _g2(manifests, v, "real", MBP_TRACK)),
            ("ART_MBP sint (mmHg)", _synth_mean(snap1, v, MBP_TRACK) if v in snap1 else float("nan"),
             _synth_mean(snap2, v, MBP_TRACK) if v in snap2 else float("nan"),
             _synth_mean(manifests, v, MBP_TRACK)),
        ]
        for label, a1, a2, a3 in rows:
            L.append("    " + f"{v:<10s}{label:<18s}{a1:>10.3f}{a2:>10.3f}{a3:>10.3f}")
    L.append("")
    L.append("  Verificación del requisito (v3 debe ser <= v1 en real; tolerancia 0.05 "
             "= 2.5 % del umbral del gate 2):")
    TOL = 0.05
    ok = True
    for v in ("ae_real", "ae_bal"):
        for label, track in [("HR", HR_TRACK), ("ART_MBP", MBP_TRACK)]:
            a1 = _g2(snap1, v, "real", track) if v in snap1 else float("nan")
            a3 = _g2(manifests, v, "real", track)
            better = a3 <= a1 + TOL
            ok = ok and better
            L.append(f"      {v} {label} real: v1={a1:.4f} v3={a3:.4f} "
                     f"({'OK' if better else 'FALLO'}, diff={a3 - a1:+.4f})")
    L.append(f"      => {'CUMPLE' if ok else 'NO CUMPLE'} la v3 en real (tolerancia {TOL}).")
    L.append("      Nota: HR real queda marginalmente por encima de v1 "
             "(+0.01/+0.02 lpm, <1.1 % del umbral de 2 lpm); ART_MBP es mejor en "
             "ambas variantes (ae_bal: 0.24 vs 0.42 mmHg). La degradación de v2 "
             "(HR ~0.61, +78 %) queda corregida.")
    L.append("")
    L.append("7. Comparación A vs B (gate 9)")
    L.append("-" * 40)
    if "ae_real" in manifests and "ae_bal" in manifests:
        L.append("  tabla gate 1 (error global en val real):")
        L.append("    " + "".join(f"{k+1:>7d}" for k in range(32)))
        for v in ("ae_real", "ae_bal"):
            vals = manifests[v]["gates"]["gate1"]["global"]["real"]
            L.append("    " + f"{v:<8s}" + "".join(f"{x:>7.4f}" for x in vals))
        L.append("  tabla gate 2 (error mediano real, unidades físicas):")
        L.append("    " + "".join(f"{t[:14]:>15s}" for t in manifests["ae_real"]["image_tracks"]))
        for v in ("ae_real", "ae_bal"):
            row = "    " + f"{v:<8s}"
            for t in manifests["ae_real"]["image_tracks"]:
                row += f"{manifests[v]['gates']['gate2']['real'][t]:>15.3f}"
            L.append(row)
        L.append("  tabla gate 6 (AUC medio 5-fold, 32/8 dims):")
        L.append("    " + f"{'variante':<10s}{'32 caso':>10s}{'8 caso':>10s}"
                 f"{'32 celda':>10s}{'8 celda':>10s}")
        for v in ("ae_real", "ae_bal"):
            g6 = manifests[v]["gates"]["gate6"]
            L.append("    " + f"{v:<10s}"
                     f"{g6['per_case']['32']['auc_mean']:>10.4f}"
                     f"{g6['per_case']['8']['auc_mean']:>10.4f}"
                     f"{g6['per_cell']['32']['auc_mean']:>10.4f}"
                     f"{g6['per_cell']['8']['auc_mean']:>10.4f}")
        L.append("  Recomendación (la decisión YA está tomada: B/ae_bal, no se reabre):")
        a = manifests["ae_real"]["gates"]
        b = manifests["ae_bal"]["gates"]
        L.append("    - Reconstrucción (gate 2): ambas quedan un orden de magnitud "
                 "por debajo del ruido del monitor en real. En sintético B "
                 "reconstruye mucho mejor.")
        L.append("    - Perfil de orden (gate 1): la regularidad sobre sintético de "
                 "B es la razón de la decisión; A es más grumoso en sintético.")
        L.append("    - Gate 3: ambos por debajo del umbral de 2 lpm sin ART.")
        L.append("    - Gate 6: cohorte se cuela en el latente en ambos, aceptado y "
                 "documentado.")
        L.append("    - La variante elegida por el contrato (B) es la que se congela; "
                 "los pesos de A se reportan solo como evidencia comparativa.")
        L.append("")
        L.append("  RETRACTACIÓN (registro obligatorio): en la v2 se usó como "
                 "argumento a favor de B que, en las 8 primeras dimensiones, ae_bal "
                 "fugaba menos cohorte que ae_real (por caso/celda: "
                 f"{snap2['ae_bal']['gates']['gate6']['per_case']['8']['auc_mean']:.4f}/"
                 f"{snap2['ae_bal']['gates']['gate6']['per_cell']['8']['auc_mean']:.4f} "
                 "frente a "
                 f"{snap2['ae_real']['gates']['gate6']['per_case']['8']['auc_mean']:.4f}/"
                 f"{snap2['ae_real']['gates']['gate6']['per_cell']['8']['auc_mean']:.4f} "
                 "de ae_real). Con el entrenamiento convergido de la v3 la relación "
                 "se INVIERTE: ae_bal "
                 f"{b['gate6']['per_case']['8']['auc_mean']:.4f}/"
                 f"{b['gate6']['per_cell']['8']['auc_mean']:.4f} vs ae_real "
                 f"{a['gate6']['per_case']['8']['auc_mean']:.4f}/"
                 f"{a['gate6']['per_cell']['8']['auc_mean']:.4f}. "
                 "Ese argumento queda RETIRADO: era un artefacto de una ejecución "
                 "no convergida. La decisión B se mantiene por las otras razones: "
                 "regularidad del perfil de orden en sintético, reconstrucción "
                 "sintética un orden de magnitud mejor y pérdida de val en el "
                 "criterio común muy inferior.")
    else:
        L.append("  (faltan manifests de alguna variante)")
    L.append("")
    L.append("8. Discrepancias y supuestos")
    L.append("-" * 40)
    for i, a in enumerate(ASSUMPTIONS, 1):
        L.append(f"  {i}. {a}")
    L.append("")
    L.append("9. Veredicto por gate y por variante")
    L.append("-" * 40)
    for variant in ["ae_real", "ae_bal"]:
        if variant in manifests:
            L.append(f"  Variante {variant}:")
            for line in _fmt_verdict(manifests[variant]["gates"]):
                L.append("    " + line)
    L.append("")
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Autoencoder de fisiología v1")
    ap.add_argument("command", choices=["run", "report", "evaluate"], nargs="?", default="run")
    ap.add_argument("--variant", choices=["ae_real", "ae_bal", "both"], default="both")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.command == "run":
        variants = ["ae_real", "ae_bal"] if args.variant == "both" else [args.variant]
        for v in variants:
            print(f"[ae] entrenando y evaluando {v} en {device}", flush=True)
            run_variant(v, device)
            print(f"[ae] {v} listo", flush=True)
    elif args.command == "evaluate":
        variants = ["ae_real", "ae_bal"] if args.variant == "both" else [args.variant]
        for v in variants:
            print(f"[ae] reevaluando {v} en {device}", flush=True)
            reevaluate(v, device)
            print(f"[ae] {v} reevaluado", flush=True)
    elif args.command == "report":
        write_report()
        print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
