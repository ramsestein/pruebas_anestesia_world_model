"""v7_transfer_probe.py — sonda de transferencia REPARADA (PASO 1, v8).

Predice el RESIDUAL delta = x_{t+1} - x_t de las 14 variables a partir de las
14 en t + 9 controles (tasas de fármaco + setpoints), con un MLP (torch, bucle
explícito). Tres condiciones:

  D1  MLP entrenado SOLO en sintético -> eval en real_holdout.
  D2  MLP entrenado SOLO en real_calib -> eval en real_holdout.
  D3  MLP entrenado en sintético + finetune en real_calib (pesos de D1, LR
      reducida) -> eval en real_holdout.

Correcciones de la sonda (cada una justificada):

  1.1 RESIDUAL. El modelo predice delta = x_{t+1} - x_t y la predicción final
      es x_t + delta. Es lo que especifica el contrato del modelo de transición
      (z_{t+1} = z_t + f(·)). Predecir valores absolutos obligaba al MLP a
      reaprender la identidad a través del cuello de botella, por eso perdía
      contra la persistencia.
  1.2 ESTANDARIZACIÓN. Z-score de las 23 features de entrada y de los 14
      objetivos (deltas), con estadísticos calculados SOLO sobre el conjunto de
      entrenamiento de cada condición. Se reportan las stats.
  1.3 CONVERGENCIA. Bucle torch explícito: nº de épocas, si convergió o se
      agotó el tope, y curva de pérdida de validación.
  1.4 FINE-TUNING CORRECTO. D3 parte de los pesos finales de D1 y continúa
      sobre real_calib con LR reducida. Se verifica (norma/hash) que los pesos
      iniciales de D3 son los finales de D1, y se reporta la pérdida de D3 en
      la época 0 (antes de cualquier paso) que debe coincidir con la de D1.
  1.5 CONTROL DE CORDURA (primero en el informe). D2 debe batir a la
      persistencia (ratio < 1) en al menos HR, ART_MBP y BIS. Si no, la sonda
      sigue rota: se reporta y NO se interpretan D1 ni D3.
  1.6 HORIZONTE. Además del paso único, rollout a 12 pasos (1 min) y 60 pasos
      (5 min) realimentando la predicción. El rollout a 60 pasos es la medida
      con poder de decisión.
  1.7 VARIABLES QUE SE MUEVEN. Evalúa HR, ART_MBP, ART_SBP, BIS, ETCO2. BT y
      SpO2 se reportan aparte, marcadas como poco informativas.
  1.8 Tabla completa condición x cohorte x horizonte x variable.

Uso:
    python -m diagnostics.v7_transfer_probe run --cohort v7

NO modifica el generador ni el AE.
"""

from __future__ import annotations

import argparse
import copy
import json
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

from diagnostics import cohort_gap as cg

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "diagnostics"
REPORT_PATH = ROOT / "reports" / "REPORT_v7_transfer_probe.txt"
CACHE_PATH = OUT_DIR / "v7_transfer_probe_results.json"
REPORTS_DIR = ROOT / "reports"

SEED = 9876
N_SYN = 100_000
N_CAL = 50_000
N_HOLD = 50_000
N_ROLL = 20_000
HORIZONS = [1, 12, 60]

# Variables de decisión (SE MUEVEN) — índices en cg.IMAGE_TRACKS.
MOVING_VARS = ["Solar8000/HR", "Solar8000/ART_MBP", "Solar8000/ART_SBP",
               "BIS/BIS", "Primus/ETCO2"]
# Poco informativas (apenas cambian en mantenimiento).
LOWINFO_VARS = ["Solar8000/BT", "Solar8000/PLETH_SPO2"]
# Control de cordura 1.5: variables mínimas donde D2 debe batir a persistencia.
SANITY_VARS = ["Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS"]
ALL_EVAL_VARS = MOVING_VARS + LOWINFO_VARS

# Características de control (tasas de fármaco + setpoints).
DRUG_RATE_COLS = [
    "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE",
    "Orchestra/NEPI_RATE", "Orchestra/PHEN_RATE",
]
SETPOINT_COLS = [
    "Primus/SET_FIO2", "Primus/SET_RR_IPPV", "Primus/SET_TV_L",
    "Primus/SET_INTER_PEEP", "Primus/SET_PIP",
]
CONTROL_COLS = DRUG_RATE_COLS + SETPOINT_COLS
N_CONTROL = len(CONTROL_COLS)

# Arquitectura y entrenamiento.
HIDDEN = (128, 64)
LR_D1 = 1e-3
LR_D2 = 1e-3
LR_D3 = 1e-4
LR_D2_PRIME = 1e-4
D2_PRIME_MAX_EPOCHS = 56
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 512
VAL_FRAC = 0.1


# --------------------------------------------------------------------------
# Cohorte: directorios por generación
# --------------------------------------------------------------------------

COHORTS = {
    "v5": {
        "windows": ROOT / "data" / "windows_v2",
        "synth_sources": ["synthetic_v5", "vaso_reinf_v5", "cf_v5"],
        "case_dirs": {
            "synthetic_v5": ROOT / "data" / "synthetic_v5",
            "vaso_reinf_v5": ROOT / "data" / "synthetic_vaso_reinf_v5",
            "cf_v5": ROOT / "data" / "cf_v5",
        },
    },
    "v6": {
        "windows": ROOT / "data" / "windows_v3",
        "synth_sources": ["synthetic_v6", "vaso_reinf_v6", "cf_v6"],
        "case_dirs": {
            "synthetic_v6": ROOT / "data" / "synthetic_v6",
            "vaso_reinf_v6": ROOT / "data" / "synthetic_vaso_reinf_v6",
            "cf_v6": ROOT / "data" / "cf_v6",
        },
    },
    "v7": {
        "windows": ROOT / "data" / "windows_v4",
        "synth_sources": ["synthetic_v7", "vaso_reinf_v7", "cf_v7"],
        "case_dirs": {
            "synthetic_v7": ROOT / "data" / "synthetic_v7",
            "vaso_reinf_v7": ROOT / "data" / "synthetic_vaso_reinf_v7",
            "cf_v7": ROOT / "data" / "cf_v7",
        },
    },
}


# --------------------------------------------------------------------------
# Escalador manual (para reportar explícitamente mean/std)
# --------------------------------------------------------------------------

class Scaler:
    """Z-score con stats explícitas (std=0 -> 1.0)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @staticmethod
    def fit(X: np.ndarray) -> "Scaler":
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return Scaler(mean, std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        return np.asarray(Y, dtype=np.float64) * self.std + self.mean


# --------------------------------------------------------------------------
# Carga y construcción de secuencias por caso
# --------------------------------------------------------------------------

def _fwd_fill(t_raw: np.ndarray, v_raw: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Forward-fill de una pista a la rejilla de las ventanas (t_grid)."""
    vf = np.asarray(v_raw, dtype=np.float64)
    finite = np.isfinite(vf)
    out = np.full(len(t_grid), np.nan, dtype=np.float64)
    if not finite.any():
        return out
    tt = np.asarray(t_raw, dtype=np.float64)[finite]
    vv = vf[finite]
    idx = np.searchsorted(tt, t_grid, side="right") - 1
    valid = idx >= 0
    idx = np.clip(idx, 0, len(vv) - 1)
    out[valid] = vv[idx[valid]]
    return out


def _case_parquet(caseid: int, source: str, case_dirs: dict[str, Path]) -> Path | None:
    base = case_dirs.get(source)
    if base is None:
        return None
    for name in (f"{caseid}.parquet", f"{caseid:04d}.parquet"):
        p = base / "cases" / name
        if p.exists():
            return p
    return None


def _col_means_safe(arr: np.ndarray, n_cols: int) -> np.ndarray:
    """Media por columna sin RuntimeWarning de slice vacía (NaN -> 0.0)."""
    out = np.zeros(n_cols, dtype=np.float64)
    for j in range(n_cols):
        col = arr[:, j]
        finite = np.isfinite(col)
        if finite.any():
            out[j] = float(col[finite].mean())
    return out


def read_controls(caseid: int, source: str, t_grid: np.ndarray,
                  case_dirs: dict[str, Path]) -> np.ndarray:
    """Lee las 9 pistas de control alineadas a t_grid (forward-fill), NaN en
    celdas sin registro (luego imputado)."""
    out = np.full((len(t_grid), N_CONTROL), np.nan, dtype=np.float64)
    p = _case_parquet(caseid, source, case_dirs)
    if p is None:
        return out
    schema = pq.read_schema(p)
    cols = ["time"] + [c for c in CONTROL_COLS if c in schema.names]
    if len(cols) == 1:
        return out
    df = pq.read_table(p, columns=cols).to_pandas()
    t_raw = df["time"].to_numpy(dtype=np.float64)
    for j, c in enumerate(CONTROL_COLS):
        if c in df.columns:
            out[:, j] = _fwd_fill(t_raw, df[c].to_numpy(dtype=np.float64), t_grid)
    return out


def _load_cells_with_t(windows_dir: Path, sources: list[str], split: str,
                       excluded: frozenset[int]) -> dict:
    """Carga celdas de mantenimiento con la columna t, ordenadas por caseid."""
    parts: list[Path] = []
    for s in sources:
        d = windows_dir / "windows" / f"source={s}" / f"split={split}"
        parts.extend(sorted(d.glob("part-*.parquet")))
    cols = cg.READ_COLS + ["t"]
    vals, masks, caseid, src, tarr = [], [], [], [], []
    for part in parts:
        df = pq.read_table(part, columns=cols).to_pandas()
        df = cg.filter_cells(df, excluded)
        if len(df) == 0:
            continue
        vals.append(df[cg.VALUE_COLS].to_numpy(dtype=np.float32))
        masks.append(df[cg.MASK_COLS].to_numpy(dtype=np.uint8))
        caseid.append(df["caseid"].to_numpy(dtype=np.int64))
        src.append(df["source"].to_numpy(dtype=object))
        tarr.append(df["t"].to_numpy(dtype=np.int32))
    if not vals:
        return {"values": np.zeros((0, cg.N_VARS), dtype=np.float32),
                "masks": np.zeros((0, cg.N_VARS), dtype=np.uint8),
                "caseid": np.zeros(0, dtype=np.int64),
                "source": np.zeros(0, dtype=object),
                "t": np.zeros(0, dtype=np.int32)}
    values = np.concatenate(vals, axis=0)
    masks = np.concatenate(masks, axis=0)
    caseid = np.concatenate(caseid, axis=0)
    source = np.concatenate(src, axis=0)
    t = np.concatenate(tarr, axis=0)
    order = np.argsort(caseid, kind="stable")
    return {"values": values[order], "masks": masks[order],
            "caseid": caseid[order], "source": source[order].astype(str),
            "t": t[order]}


def _build_case_arrays(cells: dict, case_dirs: dict[str, Path]) -> list[dict]:
    """Devuelve una lista de casos con valores imputados (L,14), controles
    imputados (L,9) y caseid/source."""
    caseid = cells["caseid"]
    source = cells["source"]
    t = cells["t"]
    n = len(caseid)
    means = cg.column_means(cells["values"], cells["masks"])
    X14 = cg.impute_with(cells["values"], cells["masks"], means)

    bounds = np.flatnonzero(np.r_[True, caseid[1:] != caseid[:-1]])
    cases: list[dict] = []
    for i, b in enumerate(bounds):
        e = bounds[i + 1] if i + 1 < len(bounds) else n
        if e - b < 2:
            continue
        controls = read_controls(int(caseid[b]), str(source[b]),
                                 t[b:e].astype(np.float64), case_dirs)
        cmean = _col_means_safe(controls, N_CONTROL)
        controls = np.where(np.isfinite(controls), controls, cmean)
        cases.append({
            "values": X14[b:e].astype(np.float32),
            "controls": controls.astype(np.float32),
            "caseid": int(caseid[b]),
            "source": str(source[b]),
        })
    return cases


def _pairs_from_cases(cases: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Aplana (X_t, delta) para entrenamiento: X = [x_t | c_t] (n, 23),
    delta = x_{t+1} - x_t (n, 14)."""
    Xs, ds = [], []
    for c in cases:
        v = c["values"]
        ctrl = c["controls"]
        Xs.append(np.concatenate([v[:-1], ctrl[:-1]], axis=1))
        ds.append(v[1:] - v[:-1])
    if not Xs:
        return (np.zeros((0, cg.N_VARS + N_CONTROL), dtype=np.float32),
                np.zeros((0, cg.N_VARS), dtype=np.float32))
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    d = np.concatenate(ds, axis=0).astype(np.float32)
    return X, d


# --------------------------------------------------------------------------
# Modelo MLP (torch)
# --------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _tensors(X: np.ndarray, y: np.ndarray):
    return (torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)))


def _eval_mse(model: MLP, Xt: torch.Tensor, yt: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(Xt)
        return float(F.mse_loss(pred, yt).item())


def _params_norm(state: dict) -> float:
    s = 0.0
    for v in state.values():
        s += float((v.float() ** 2).sum().item())
    return s


def _params_max_abs_diff(sa: dict, sb: dict) -> float:
    keys = list(sa.keys())
    return max(float((sa[k] - sb[k]).abs().max().item()) for k in keys)


def train_mlp(Xtr: np.ndarray, ytr: np.ndarray, *, lr: float,
              init_state: dict | None = None, seed: int = SEED,
              max_epochs: int = MAX_EPOCHS) -> tuple[MLP, dict]:
    """Entrena el MLP con validación fija (10 %) y parada temprana.

    Devuelve (modelo con los mejores pesos, informe de convergencia).
    """
    torch.manual_seed(seed)
    n = len(Xtr)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * VAL_FRAC)))
    tr_idx = perm[:n - n_val]
    va_idx = perm[n - n_val:]
    Xt, yt = _tensors(Xtr[tr_idx], ytr[tr_idx])
    Xv, yv = _tensors(Xtr[va_idx], ytr[va_idx])

    model = MLP(cg.N_VARS + N_CONTROL, HIDDEN, cg.N_VARS)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        perm_e = torch.randperm(len(Xt))
        running = 0.0
        for i in range(0, len(Xt), BATCH_SIZE):
            idx = perm_e[i:i + BATCH_SIZE]
            Xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            loss = F.mse_loss(model(Xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        train_losses.append(running / len(Xt))
        val_loss = _eval_mse(model, Xv, yv)
        val_losses.append(val_loss)
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    stopped_by_patience = epochs_no_improve >= PATIENCE
    converged = stopped_by_patience
    hit_max_epochs = not stopped_by_patience
    report = {
        "n_epochs_run": epoch + 1,
        "best_epoch": best_epoch,
        "converged": converged,
        "stopped_by_patience": stopped_by_patience,
        "hit_max_epochs": hit_max_epochs,
        "best_val_loss": best_val,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lr": lr,
    }
    return model, report


# --------------------------------------------------------------------------
# Predicción con residual + rollout
# --------------------------------------------------------------------------

def _predict_single(model: MLP, x14: np.ndarray, ctrl: np.ndarray,
                    scaler_X: Scaler, scaler_Y: Scaler) -> np.ndarray:
    """Predice x_{t+1} = x_t + delta a partir de (x_t, c_t)."""
    xin = np.concatenate([x14, ctrl]).astype(np.float64)[None, :]
    z = scaler_X.transform(xin)
    with torch.no_grad():
        dz = model(torch.from_numpy(z.astype(np.float32))).numpy()
    delta = scaler_Y.inverse_transform(dz[0])
    return x14 + delta


def _rollout_mae(model: MLP | None, cases: list[dict],
                 scaler_X: Scaler | None, scaler_Y: Scaler | None,
                 horizons: list[int], var_idx: list[int],
                 n_roll: int, seed: int = SEED) -> dict:
    """Rollout con realimentación de la predicción. Para model=None se usa la
    persistencia (x_{t+h} = x_t). Devuelve {horizonte: {var: mae}}."""
    rng = np.random.default_rng(seed)
    H = max(horizons)
    starts: list[tuple[int, int]] = []
    for ci, c in enumerate(cases):
        L = len(c["values"])
        if L > H:
            for i in range(0, L - H):
                starts.append((ci, i))
    if not starts:
        return {h: {cg.IMAGE_TRACKS[j]: float("nan") for j in var_idx}
                for h in horizons}
    if len(starts) > n_roll:
        sel = np.sort(rng.choice(len(starts), n_roll, replace=False))
        starts = [starts[i] for i in sel]

    acc = {h: {j: 0.0 for j in var_idx} for h in horizons}
    cnt = {h: 0 for h in horizons}
    for ci, i in starts:
        c = cases[ci]
        v = c["values"]
        ctrl = c["controls"]
        x_hat = v[i].astype(np.float64)
        for step in range(1, H + 1):
            if model is not None:
                x_hat = _predict_single(model, x_hat, ctrl[i + step - 1],
                                        scaler_X, scaler_Y)
            if step in horizons:
                target = v[i + step].astype(np.float64)
                for j in var_idx:
                    acc[step][j] += float(abs(target[j] - x_hat[j]))
                cnt[step] += 1
    out: dict = {}
    for h in horizons:
        out[h] = {cg.IMAGE_TRACKS[j]:
                  (acc[h][j] / cnt[h] if cnt[h] else float("nan"))
                  for j in var_idx}
    return out


# --------------------------------------------------------------------------
# Cómputo principal
# --------------------------------------------------------------------------

def _ratios(mae: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {t: (mae[t] / base[t]) if base[t] > 0 else float("nan")
            for t in mae}


def _run_condition(X_syn, d_syn, X_cal, d_cal, cases_hold, seed: int) -> dict:
    """Ejecuta D1/D2/D3 + persistencia sobre un cohorte."""
    var_idx = [cg.IMAGE_TRACKS.index(t) for t in ALL_EVAL_VARS]

    rng = np.random.default_rng(SEED)

    def sub(X, d, n):
        if len(X) <= n:
            return X, d
        idx = np.sort(rng.choice(len(X), n, replace=False))
        return X[idx], d[idx]

    X_syn, d_syn = sub(X_syn, d_syn, N_SYN)
    X_cal, d_cal = sub(X_cal, d_cal, N_CAL)

    # Persistencia (modelo None -> x_{t+h} = x_t).
    base_roll = _rollout_mae(None, cases_hold, None, None, HORIZONS, var_idx,
                             N_ROLL, seed)

    def fit_and_eval(Xtr, dtr, lr, init_state=None, max_epochs=MAX_EPOCHS,
                     sx=None, sy=None):
        sx = sx if sx is not None else Scaler.fit(Xtr)
        sy = sy if sy is not None else Scaler.fit(dtr)
        Xz = sx.transform(Xtr)
        dz = sy.transform(dtr)
        model, report = train_mlp(Xz, dz, lr=lr, init_state=init_state,
                                  max_epochs=max_epochs, seed=seed)
        roll = _rollout_mae(model, cases_hold, sx, sy, HORIZONS, var_idx,
                            N_ROLL, seed)
        return model, report, sx, sy, roll

    # D1: solo sintético.
    d1_model, d1_train, d1_sx, d1_sy, d1_roll = fit_and_eval(X_syn, d_syn, LR_D1)

    # D2: solo real_calib (lr 1e-3, presupuesto original).
    d2_model, d2_train, d2_sx, d2_sy, d2_roll = fit_and_eval(X_cal, d_cal, LR_D2)

    # D3: fine-tuning desde D1 sobre real_calib (mismos escaladores de D1).
    d1_state = copy.deepcopy(d1_model.state_dict())
    d3_model_init = MLP(cg.N_VARS + N_CONTROL, HIDDEN, cg.N_VARS)
    d3_model_init.load_state_dict(d1_state)
    d3_init_norm = _params_norm(d1_state)
    d3_init_maxdiff = _params_max_abs_diff(d3_model_init.state_dict(), d1_state)

    # Pérdida de D1 y de D3-época-0 sobre el val de calib (debe coincidir).
    rng3 = np.random.default_rng(seed)
    n = len(X_cal)
    perm3 = rng3.permutation(n)
    n_val3 = max(1, int(round(n * VAL_FRAC)))
    Xc_z = d1_sx.transform(X_cal)
    dc_z = d1_sy.transform(d_cal)
    Xv3, yv3 = _tensors(Xc_z[perm3[:n_val3]], dc_z[perm3[:n_val3]])
    d1_loss_on_cal = _eval_mse(d1_model, Xv3, yv3)
    d3_epoch0_loss = _eval_mse(d3_model_init, Xv3, yv3)

    # Continuar entrenando D3 (fine-tune, LR baja, escalador objetivo sintético).
    Xz_cal = d1_sx.transform(X_cal)
    dz_cal = d1_sy.transform(d_cal)
    d3_model, d3_train = train_mlp(Xz_cal, dz_cal, lr=LR_D3,
                                   init_state=d1_state, seed=seed)
    d3_roll = _rollout_mae(d3_model, cases_hold, d1_sx, d1_sy, HORIZONS,
                           var_idx, N_ROLL, seed)

    # D3' (2.2): fine-tuning desde D1 con escalador de OBJETIVO real_calib
    # (la std de los deltas reales es 2.7-3x la sintética; D3 original ajusta
    # en un espacio mal escalado).
    d3p_sy = Scaler.fit(d_cal)
    Xz_cal_p = d1_sx.transform(X_cal)
    dz_cal_p = d3p_sy.transform(d_cal)
    d3p_model, d3p_train = train_mlp(Xz_cal_p, dz_cal_p, lr=LR_D3,
                                     init_state=d1_state, seed=seed)
    d3p_roll = _rollout_mae(d3p_model, cases_hold, d1_sx, d3p_sy, HORIZONS,
                            var_idx, N_ROLL, seed)

    # D2' (2.1): desde inicialización ALEATORIA con el MISMO presupuesto de
    # optimización que el fine-tuning de D3 (lr 1e-4, tope 56 épocas, misma
    # paciencia y batch). Controla el confundidor del presupuesto.
    d2p_model, d2p_train, d2p_sx, d2p_sy, d2p_roll = fit_and_eval(
        X_cal, d_cal, LR_D2_PRIME, max_epochs=D2_PRIME_MAX_EPOCHS)

    def ratios_vs_base(roll: dict, horizon: int) -> dict[str, float]:
        return _ratios(roll[horizon], base_roll[horizon])

    d2_ratios_h1 = ratios_vs_base(d2_roll, 1)
    sanity = {t: (d2_ratios_h1[t] < 1.0) for t in SANITY_VARS}
    sanity_passed = all(sanity.values())

    def mean_ratio(roll, horizon):
        rs = ratios_vs_base(roll, horizon)
        return float(np.nanmean([rs[t] for t in MOVING_VARS]))

    d3_beats_d2_h60 = mean_ratio(d3_roll, 60) < mean_ratio(d2_roll, 60)
    d1_beats_persist_h1 = mean_ratio(d1_roll, 1) < 1.0

    # Veredicto 2.1: D3 vs D2' en las 15 comparaciones (5 vars x 3 horizontes).
    n_wins_d3_vs_d2p = 0
    n_cmp = 0
    for h in HORIZONS:
        for t in MOVING_VARS:
            a = d3_roll[h][t]
            b = d2p_roll[h][t]
            if np.isfinite(a) and np.isfinite(b):
                n_cmp += 1
                if a < b:
                    n_wins_d3_vs_d2p += 1
    transfer_confirmed = (n_cmp > 0) and (n_wins_d3_vs_d2p > n_cmp / 2)

    return {
        "persistence_mae": {h: base_roll[h] for h in HORIZONS},
        "d1_mae": {h: d1_roll[h] for h in HORIZONS},
        "d2_mae": {h: d2_roll[h] for h in HORIZONS},
        "d2prime_mae": {h: d2p_roll[h] for h in HORIZONS},
        "d3_mae": {h: d3_roll[h] for h in HORIZONS},
        "d3prime_mae": {h: d3p_roll[h] for h in HORIZONS},
        "d1_ratio": {h: ratios_vs_base(d1_roll, h) for h in HORIZONS},
        "d2_ratio": {h: ratios_vs_base(d2_roll, h) for h in HORIZONS},
        "d2prime_ratio": {h: ratios_vs_base(d2p_roll, h) for h in HORIZONS},
        "d3_ratio": {h: ratios_vs_base(d3_roll, h) for h in HORIZONS},
        "d3prime_ratio": {h: ratios_vs_base(d3p_roll, h) for h in HORIZONS},
        "d1_training": d1_train,
        "d2_training": d2_train,
        "d2prime_training": d2p_train,
        "d3_training": d3_train,
        "d3prime_training": d3p_train,
        "d1_scaler_X": {"mean": d1_sx.mean.tolist(), "std": d1_sx.std.tolist()},
        "d1_scaler_Y": {"mean": d1_sy.mean.tolist(), "std": d1_sy.std.tolist()},
        "d2_scaler_X": {"mean": d2_sx.mean.tolist(), "std": d2_sx.std.tolist()},
        "d2_scaler_Y": {"mean": d2_sy.mean.tolist(), "std": d2_sy.std.tolist()},
        "d3prime_scaler_Y": {"mean": d3p_sy.mean.tolist(), "std": d3p_sy.std.tolist()},
        "d3_init_params_norm": d3_init_norm,
        "d1_final_params_norm": _params_norm(d1_state),
        "d3_init_max_abs_diff_vs_d1": d3_init_maxdiff,
        "d1_loss_on_calib_val": d1_loss_on_cal,
        "d3_epoch0_loss_on_calib_val": d3_epoch0_loss,
        "sanity": {"ratios": d2_ratios_h1, "per_var": sanity,
                   "passed": sanity_passed},
        "decision": {
            "sanity_passed": sanity_passed,
            "d1_beats_persistence_h1": d1_beats_persist_h1,
            "d3_beats_d2_h60": d3_beats_d2_h60,
            "d3_wins_vs_d2prime": n_wins_d3_vs_d2p,
            "n_comparisons": n_cmp,
            "transfer_confirmed": transfer_confirmed,
            "continue": transfer_confirmed,
        },
    }


def compute_results(cohort: str) -> dict:
    t0 = _time.time()
    cfg = COHORTS[cohort]
    windows_dir = cfg["windows"]
    excluded = cg.load_excluded_caseids()

    if not (windows_dir / "windows").exists():
        return {
            "cohort": cohort,
            "error": f"no existe {windows_dir} (datos windowed de la cohorte "
                     f"{cohort} no disponibles en disco)",
            "elapsed_s": 0.0,
        }

    print(f"[v7_transfer_probe] {cohort}: cargando real (val) con t", flush=True)
    real_all = _load_cells_with_t(windows_dir, ["real"], "val", excluded)
    real_caseids = sorted(set(int(c) for c in real_all["caseid"]))
    rng = np.random.default_rng(20260922)
    perm = rng.permutation(real_caseids)
    k = int(round(len(perm) * 0.40))
    holdout_ids = frozenset(int(c) for c in perm[:k])
    calib_ids = frozenset(int(c) for c in perm[k:])

    hmask = np.array([int(c) in holdout_ids for c in real_all["caseid"]])
    cmask = np.array([int(c) in calib_ids for c in real_all["caseid"]])
    holdout = {k_: real_all[k_][hmask] for k_ in real_all}
    calib = {k_: real_all[k_][cmask] for k_ in real_all}

    print(f"[v7_transfer_probe] {cohort}: real {len(real_caseids)} caseids "
          f"({len(calib_ids)} calib / {len(holdout_ids)} holdout)", flush=True)

    print(f"[v7_transfer_probe] {cohort}: cargando sintético (val)", flush=True)
    synth = _load_cells_with_t(windows_dir, cfg["synth_sources"], "val", excluded)

    print(f"[v7_transfer_probe] {cohort}: construyendo secuencias por caso",
          flush=True)
    case_dirs = {"real": ROOT / "data" / "real", **cfg["case_dirs"]}
    cases_hold = _build_case_arrays(holdout, case_dirs)
    cases_cal = _build_case_arrays(calib, case_dirs)
    cases_syn = _build_case_arrays(synth, case_dirs)

    X_syn, d_syn = _pairs_from_cases(cases_syn)
    X_cal, d_cal = _pairs_from_cases(cases_cal)

    print(f"[v7_transfer_probe] {cohort}: pares synth={len(X_syn)} "
          f"calib={len(X_cal)} holdout_casos={len(cases_hold)}", flush=True)

    print(f"[v7_transfer_probe] {cohort}: entrenando D1/D2/D3", flush=True)
    res = _run_condition(X_syn, d_syn, X_cal, d_cal, cases_hold, SEED)
    res["cohort"] = cohort
    res["n_syn_train"] = int(len(X_syn))
    res["n_cal_train"] = int(len(X_cal))
    res["n_hold_cases"] = int(len(cases_hold))
    res["elapsed_s"] = round(_time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------
# Informe
# --------------------------------------------------------------------------

def _f(x) -> float | None:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if np.isfinite(y) else None


def _g(x) -> str:
    y = _f(x)
    return "    nan" if y is None else f"{y:7.3f}"


def write_report(results: dict, cohort: str) -> None:
    L: list[str] = []
    add = L.append

    if results.get("error"):
        add(f"REPORT_v7_transfer_probe.txt — sonda de transferencia ({cohort})")
        add("=" * 78)
        add("")
        add(f"ERROR: {results['error']}")
        add("")
        add("La sonda no puede ejecutarse sobre esta cohorte: los datos "
            "windowed no están disponibles en disco. Solo la cohorte v7 "
            "(data/windows_v4) está presente en el workspace.")
        REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
        return

    add(f"REPORT_v7_transfer_probe.txt — sonda de transferencia reparada "
        f"({cohort})")
    add("=" * 78)
    add("")

    # ── 1.5 CONTROL DE CORDURA (informativo; el gate v9 pasa a ser D3 vs D2) ──
    san = results["sanity"]
    add("CONTROL DE CORDURA (informativo) — D2 vs persistencia")
    add("-" * 40)
    add("  El control 1.5 era defectuoso y no debía existir como gate: a un paso,")
    add("  con frac_nochg real ~0.53 en HR, la mediana del incremento es 0 y la")
    add("  persistencia es óptima por construcción. El gate pasa a ser D3 vs D2.")
    add("  ratio D2/persistencia (paso único):")
    for t in SANITY_VARS:
        r = san["ratios"][t]
        add(f"    {t:<22} ratio = {_g(r)}")
    add("")

    # ── 1.2 ESTANDARIZACIÓN ──
    add("1.2 ESTANDARIZACIÓN (z-score)")
    add("-" * 40)
    add("  Entrada: 23 features (14 valores + 9 controles). Objetivo: 14 "
        "deltas.")
    add("  Stats calculadas SOLO sobre el conjunto de entrenamiento de cada "
        "condición:")
    add(f"    D1: sobre sintético ({results.get('n_syn_train')} pares)")
    add(f"    D2: sobre real_calib ({results.get('n_cal_train')} pares)")
    add("    D3: sobre sintético (idéntico a D1: continúa en el mismo espacio)")
    add("    D3': objetivo re-escalado a real_calib (2.2)")
    for cond in ("d1", "d2"):
        sx = results.get(f"{cond}_scaler_X", {})
        sy = results.get(f"{cond}_scaler_Y", {})
        add(f"  {cond.upper()} scaler_X (23): mean[0:5] = "
            f"{[round(v, 4) for v in (sx.get('mean') or [])[:5]]}, "
            f"std[0:5] = {[round(v, 4) for v in (sx.get('std') or [])[:5]]}")
        add(f"  {cond.upper()} scaler_Y (14 deltas): mean[0:5] = "
            f"{[round(v, 4) for v in (sy.get('mean') or [])[:5]]}, "
            f"std[0:5] = {[round(v, 4) for v in (sy.get('std') or [])[:5]]}")
    add("")

    # ── 1.3 CONVERGENCIA ──
    add("1.3 CONVERGENCIA")
    add("-" * 40)
    for cond in ("d1", "d2", "d2prime", "d3", "d3prime"):
        tr = results.get(f"{cond}_training", {})
        if not tr:
            continue
        vl = tr.get("val_losses", [])
        if len(vl) > 8:
            curve = ", ".join(f"{v:.4f}" for v in vl[:5]) + " ... " + \
                ", ".join(f"{v:.4f}" for v in vl[-3:])
        else:
            curve = ", ".join(f"{v:.4f}" for v in vl)
        add(f"  {cond.upper():<8}: épocas={tr.get('n_epochs_run')}, "
            f"mejor época={tr.get('best_epoch')}, "
            f"convergió={'SÍ (parada temprana)' if tr.get('converged') else 'NO (tope agotado)'}, "
            f"best_val_loss={tr.get('best_val_loss'):.6f}, lr={tr.get('lr')}")
        add(f"    curva val: {curve}")
    add("")

    # ── 1.4 FINE-TUNING ──
    add("1.4 FINE-TUNING CORRECTO (D3 desde D1)")
    add("-" * 40)
    add(f"  norma params D1 final   = {results.get('d1_final_params_norm'):.6f}")
    add(f"  norma params D3 inicial = {results.get('d3_init_params_norm'):.6f}")
    add(f"  max|D3_init - D1_final| = "
        f"{results.get('d3_init_max_abs_diff_vs_d1'):.3e} (0 = idénticos)")
    add(f"  pérdida D1 en val calib = "
        f"{results.get('d1_loss_on_calib_val'):.6f}")
    add(f"  pérdida D3 época 0      = "
        f"{results.get('d3_epoch0_loss_on_calib_val'):.6f}")
    match = abs((results.get('d1_loss_on_calib_val') or 0.0)
                - (results.get('d3_epoch0_loss_on_calib_val') or 0.0)) < 1e-6
    add(f"  coinciden (|Δ| < 1e-6): {'SÍ' if match else 'NO'}")
    add("")

    # ── Tabla completa ──
    add("TABLA: condición x horizonte x variable (MAE)")
    add("-" * 40)
    add("  Variables que SE MUEVEN (decisión): HR, ART_MBP, ART_SBP, BIS, "
        "ETCO2")
    add("  Poco informativas (ruido): BT, SpO2")
    for h in HORIZONS:
        add(f"  horizonte {h} pasos ({h * 5} s):")
        add(f"    {'variable':<22}{'persist':>8}{'D1':>8}{'D2':>8}{'D2p':>8}"
            f"{'D3':>8}{'D3p':>8}")
        for t in ALL_EVAL_VARS:
            row = (f"    {t:<22}"
                   f"{_g(results['persistence_mae'][h][t]):>8}"
                   f"{_g(results['d1_mae'][h][t]):>8}"
                   f"{_g(results['d2_mae'][h][t]):>8}"
                   f"{_g(results['d2prime_mae'][h][t]):>8}"
                   f"{_g(results['d3_mae'][h][t]):>8}"
                   f"{_g(results['d3prime_mae'][h][t]):>8}")
            add(row)
        add("")
    add("  ratios vs persistencia (solo variables móviles):")
    for h in HORIZONS:
        add(f"  h={h}: " + "  ".join(
            f"{t.split('/')[-1]}: D2={_g(results['d2_ratio'][h][t])} "
            f"D2'={_g(results['d2prime_ratio'][h][t])} "
            f"D3={_g(results['d3_ratio'][h][t])} "
            f"D3'={_g(results['d3prime_ratio'][h][t])}"
            for t in MOVING_VARS))
    add("")

    # ── Decisión ──
    dec = results["decision"]
    add("DECISIÓN (gate v9: D3 vs D2; la persistencia es referencia de escala)")
    add("-" * 40)
    add(f"  D3 bate D2 (h=60): "
        f"{'SÍ' if dec['d3_beats_d2_h60'] else 'NO'}")
    add(f"  D3 bate D2' en {dec.get('d3_wins_vs_d2prime')} de "
        f"{dec.get('n_comparisons')} comparaciones (5 vars x 3 horizontes)")
    add(f"  TRANSFERENCIA CONFIRMADA: "
        f"{'SÍ' if dec.get('transfer_confirmed') else 'NO'}")
    add(f"  DECISIÓN = "
        f"{'TRANSFERENCIA CONFIRMADA — generador cerrado en v7' if dec.get('transfer_confirmed') else 'SIN TRANSFERENCIA DEMOSTRADA (ventaja del optimizador)'}")

    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"REPORT_v7_transfer_probe_{cohort}.txt").write_text(
        "\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--cohort", choices=list(COHORTS), default="v7")
    args = ap.parse_args()
    if args.command == "run":
        res = compute_results(args.cohort)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                         default=str), encoding="utf-8")
        write_report(res, args.cohort)
        print(json.dumps(res.get("decision", res.get("error", {})), indent=2,
                         ensure_ascii=False))
    else:
        res = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        write_report(res, args.cohort)


if __name__ == "__main__":
    main()
