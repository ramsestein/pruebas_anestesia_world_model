"""cf_pairs.py — anotación de pares contrafactuales (paso 3b).

Este módulo concentra las definiciones y funciones reutilizables para la
anotación de los pares CF de ``data/tokens_v2/pairs.parquet``:

  - el mapa ``lever_group -> tracks crudos`` que alimentan las features de cada
    grupo (tasas de bomba, columnas de bolo, SET_* del ventilador y MAC);
  - la derivación de ``t_action`` (el momento real en que actúa la palanca) a
    partir de los metadatos ``cf_pair_*.json`` o, para los overrides de
    ventilación que no registran un ``intervention_a``, del código del
    generador (``src/anessim/simulate.py``);
  - ``post_action(t1, t_action)``: pertenencia de una ventana a la fase
    posterior a la acción, calculada al vuelo (t1 > t_action), que es el
    criterio CORREGIDO del paso 3b (el paso 3 medía en la ventana que contiene
    split_t, ignorando que el generador aplica las intervenciones
    farmacológicas en split_t + 10 s);
  - el conjunto CF de entrenamiento: los pares con ``lever_effective = True``
    de ``data/tokens_v2/pairs_annotated.parquet``.

No se regenera nada: este módulo SOLO lee ``pairs.parquet``,
``pairs_annotated.parquet`` y los metadatos de cf_v7.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import paths
from tokens import tokenize as tk

# ---------------------------------------------------------------------------
# Rutas de artefactos
# ---------------------------------------------------------------------------
PAIRS_PARQUET = paths.TOKENS_V2_DIR / "pairs.parquet"
PAIRS_ANNOTATED_PARQUET = paths.TOKENS_V2_DIR / "pairs_annotated.parquet"
CF_META_DIR = paths.COHORTS["cf_v7"] / "metadata"
CF_CASES_DIR = paths.COHORTS["cf_v7"] / "cases"

# ---------------------------------------------------------------------------
# Mapa lever_group -> tracks crudos que alimentan las features del grupo.
#
# Farmacológicos: la feature de fármaco (ce_*, bolo, dose_cum) la calcula
# pk_tokens.py a partir de los tracks OBSERVADOS de tasa de bomba y de las
# columnas de bolo (no del truth). Ventilación: las features vent_* vienen de
# los setpoints del ventilador (SET_*) y sus proxies medidos. Sevo: MAC.
# ---------------------------------------------------------------------------
LEVER_TRACKS: dict[str, tuple[str, ...]] = {
    "propofol": ("Orchestra/PPF20_RATE", "ppf_bolus_mg"),
    "remifentanilo": ("Orchestra/RFTN20_RATE", "remi_bolus_ug"),
    "efedrina": ("Orchestra/EPH_RATE", "eph_bolus_mg"),
    "fenilefrina": ("Orchestra/PHEN_RATE", "phen_bolus_mcg"),
    "noradrenalina": ("Orchestra/NEPI_RATE",),
    "sevoflurano": ("Primus/SET_MAC", "Primus/MAC"),
    "ventilacion": ("Primus/SET_FIO2", "Primus/SET_INTER_PEEP",
                    "Primus/SET_RR_IPPV", "Primus/SET_TV_L", "Primus/SET_PIP"),
}

# Palancas cuyo t_action NO está registrado en intervention_a (son overrides de
# ventilación) y se deriva del código: simulate.py aplica el override a
# ``post = t > counterfactual_split_t`` (es decir, t_action = split_t).
VENT_OVERRIDE_LEVERS: frozenset[str] = frozenset({
    "peep_up", "peep_down", "fio2_down",
    "set_fio2", "set_rr", "set_tv", "set_peep",
})

# Palancas de CONSIGNA PERSISTENTE: el simulador aplica un delta acumulativo a
# la señal (``peep_arr[post] += delta`` en simulate.py) en lugar de un evento
# puntual. La divergencia OBSERVADA aparece cuando el plan base cambia esa
# consigna, no en t_action, de modo que C1/C2 (ventana de t_action o siguiente)
# no son aplicables a esta clase y se reportan aparte.
PERSISTENT_LEVERS: frozenset[str] = VENT_OVERRIDE_LEVERS

# Origen de t_action para los overrides de ventilación.
VENT_ACTION_SOURCE = "codigo:src/anessim/simulate.py:333"


def is_persistent(lever: str) -> bool:
    """Verdadero si la palanca es una consigna persistente (no un evento)."""
    return lever in PERSISTENT_LEVERS


# features del lever_group de cada palanca (misma definición que el paso 3).
LEVER_FEATURES: dict[str, tuple[str, ...]] = {
    d: tuple(f"drug_{d}_{f}" for f in tk.DRUG_FEATURES) for d in tk.DRUGS
}
LEVER_FEATURES["ventilacion"] = tuple(
    f"vent_{v}_{t}" for v in tk.VENT_ITEMS for t in ("t0", "t1"))


# ---------------------------------------------------------------------------
# t_action
# ---------------------------------------------------------------------------

_ACTION_T_RE = re.compile(r"^t=([0-9.]+)s")


def parse_action_time(summary: str) -> float:
    """Extrae el instante ``t=XXXs`` de un resumen de acción del manifiesto."""
    m = _ACTION_T_RE.match(summary.strip())
    if not m:
        raise ValueError(f"no se pudo parsear t de la acción: {summary!r}")
    return float(m.group(1))


def action_time(meta: dict, lever: str, split_t: float) -> tuple[float, str]:
    """Devuelve (t_action, action_source) para un par CF.

    ``meta`` es el dict de ``cf_pair_<caseid_a>.json``. Si la colección
    registra la intervención (``intervention_a`` con ``t=XXXs``), t_action se
    lee de los metadatos; si no (overrides de ventilación), se deriva del
    código del generador y se cita ``codigo:...``.
    """
    inter = meta.get("intervention_a") or []
    if inter and not lever in VENT_OVERRIDE_LEVERS:
        return parse_action_time(str(inter[0])), "metadata"
    # Override de ventilación: simulate.py aplica ``post = t > split_t``.
    return float(split_t), VENT_ACTION_SOURCE


# ---------------------------------------------------------------------------
# post_action (criterio corregido del paso 3b)
# ---------------------------------------------------------------------------

# Rejilla de la capa de tokens (tokenize.GRID_S). pk_v2 reporta en el inicio de
# la celda [5k, 5k+5) el efecto de las acciones que caen dentro de ella, así que
# la identidad de la intervención puede aparecer en las features hasta 5 s ANTES
# de t_action (verificado: p. ej. en el par 171501, la acción está en 6781.7 s y
# ce_efedrina ya salta en el punto de rejilla 6780).
GRID_S = 5.0


def action_grid_time(t_action: float, grid_s: float = GRID_S) -> float:
    """Instante de ``t_action`` en la rejilla de 5 s de la capa de tokens."""
    return math.floor(float(t_action) / grid_s) * grid_s


def post_action(t1: float, t_action: float) -> bool:
    """Verdadero si la ventana de cierre ``t1`` es posterior a ``t_action``.

    Refina ``post_action = (t1 > t_action)`` (B1) a la rejilla de la capa de
    tokens: la PRIMERA ventana post-acción es la que cierra en el punto de
    rejilla que contiene ``t_action`` (``t1 >= action_grid_time(t_action)``).
    Sin este refinamiento, la ventana que cierra justo antes de ``t_action``
    quedaría en el prefijo aunque sus features ya incluyan la intervención, por
    la cuantización de 5 s de pk_v2 (violando C4).
    """
    return float(t1) >= action_grid_time(t_action)


# Margen (en celdas de la rejilla de tokens) con el que se calcula el PREFIJO de
# C4. Los tracks de evento discreto (``Orchestra/PHEN_RATE``, ``EPH_RATE``,
# spike-hold de bolos) pueden atribuir el bolo a la celda ANTERIOR a la que lo
# contiene: medido, la Ce de fenilefrina/efedrina puede divergir en el punto de
# rejilla ``action_grid_time(t_action) - 5 s`` (pares 194307, 194363, 194667,
# 195027, 195125, 195571). Por eso el prefijo exigible se queda 2 celdas por
# detrás: ``t1 <= action_grid_time(t_action) - 2 * GRID_S``.
PREFIX_MARGIN_CELLS = 2


def pre_action(t1: float, t_action: float,
               margin_cells: int = PREFIX_MARGIN_CELLS) -> bool:
    """Verdadero si la ventana es PREFIJO: no puede ver la intervención.

    Es el complementario conservador de ``post_action``: exige dos celdas de
    rejilla de margen para absorber la atribución temprana de la capa de
    observación. Las ventanas de prefijo deben ser idénticas entre ramas en
    todas las features y máscaras (C4).
    """
    return float(t1) <= action_grid_time(t_action) - margin_cells * GRID_S


# ---------------------------------------------------------------------------
# Conjunto CF de entrenamiento (pares con lever_effective = True)
# ---------------------------------------------------------------------------

def load_pairs_annotated(path: Path | None = None) -> pd.DataFrame:
    """Lee ``data/tokens_v2/pairs_annotated.parquet`` (columnas de pairs +
    anotación). Lanza FileNotFoundError si aún no existe."""
    p = Path(path) if path is not None else PAIRS_ANNOTATED_PARQUET
    if not p.exists():
        raise FileNotFoundError(
            f"{p} no existe (ejecuta scripts/paso3b_annotate_cf_pairs.py)")
    return pq.read_table(p).to_pandas()


def effective_pair_ids(annotated: pd.DataFrame | None = None) -> frozenset[int]:
    """Los pair_id del conjunto CF de entrenamiento: pares con efecto
    (``lever_effective == True``) de ``pairs_annotated.parquet``."""
    if annotated is None:
        annotated = load_pairs_annotated()
    mask = annotated["lever_effective"].astype(bool)
    return frozenset(int(x) for x in annotated.loc[mask, "pair_id"])


def effective_pair_ids_count(annotated: pd.DataFrame | None = None) -> int:
    """Número de pares CF efectivos (para recuentos del manifiesto)."""
    return len(effective_pair_ids(annotated))


# ---------------------------------------------------------------------------
# Utilidades de metadatos CF
# ---------------------------------------------------------------------------

def load_cf_pair_meta(pair_id: int) -> dict:
    """Lee ``cf_pair_<pair_id>.json``."""
    p = CF_META_DIR / f"cf_pair_{pair_id}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def lever_of(meta: dict) -> str:
    return str(meta["lever"])


def lever_group_of(lever: str) -> str:
    return str(tk.LEVER_GROUP[lever])


# ---------------------------------------------------------------------------
# tracks crudos de un par
# ---------------------------------------------------------------------------

def load_raw_tracks(caseid: int, tracks: tuple[str, ...]) -> pd.DataFrame:
    """Lee ``time`` + ``tracks`` del parquet de caso crudo (eje irregular)."""
    cols = ["time", *[t for t in tracks]]
    df = pq.read_table(CF_CASES_DIR / f"{caseid}.parquet",
                       columns=cols).to_pandas()
    return df


def first_divergence_raw(
    base: pd.DataFrame,
    inter: pd.DataFrame,
    tracks: tuple[str, ...],
) -> float | None:
    """Primer instante (``time``) en que difiere algún track crudo entre las
    dos ramas; ``None`` (NaN) si nunca difieren. Comparación NaN-aware."""
    t = base["time"].to_numpy()
    first: float | None = None
    for tr in tracks:
        if tr not in base.columns or tr not in inter.columns:
            continue
        b = base[tr].to_numpy(dtype=np.float64)
        i = inter[tr].to_numpy(dtype=np.float64)
        # NaN == NaN (no difieren); NaN vs valor difiere.
        diff = ~((b == i) | (np.isnan(b) & np.isnan(i)))
        idx = int(np.argmax(diff)) if diff.any() else -1
        if idx >= 0:
            tt = float(t[idx])
            first = tt if first is None else min(first, tt)
    return first
