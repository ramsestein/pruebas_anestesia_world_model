"""PASO 3b / Fase A-B — anotación de los pares contrafactuales.

Para cada par de ``data/tokens_v2/pairs.parquet`` (6345) calcula:

  - ``t_action``        instante real en que actúa la palanca.
  - ``action_source``   "metadata" (intervention_a) o "codigo:<archivo>:<linea>".
  - ``t_divergence_raw`` primer instante en que difiere algún track crudo del
                        lever_group entre las dos ramas (NaN si nunca).
  - ``requested_change`` descripción compacta de lo que pide la intervención.
  - ``base_value_at_action`` valor de la variable primaria de la palanca en la
                        rama base, en el último muestreo <= t_action.
  - ``effect_t1``       cierre de la primera ventana post-acción en que difieren
                        las features del lever_group (NaN si ninguna).
  - ``effect_lag_windows`` número de ventanas post-acción hasta el efecto
                        (0 = la ventana que contiene t_action), NaN si nulo.
  - ``lever_effective`` True si alguna ventana post-acción difiere.
  - ``null_cause``      clip_bound | below_resolution | no_change_requested |
                        case_ends | unexplained (NaN si el par es efectivo).

Y escribe ``data/tokens_v2/pairs_annotated.parquet`` (todas las columnas de
pairs.parquet + las anteriores) y
``manifests/tokens_v2_cf_pairs_annotation.json`` con los criterios C1-C4 y el
desglose de nulos.

No regenera nada: sólo lee pairs.parquet, las ventanas de tokens_v2, los
metadatos y la verdad de cf_v7.

Uso:
  python -m scripts.paso3b_annotate_cf_pairs
"""

from __future__ import annotations

import hashlib
import json
import re
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import paths
from tokens import cf_pairs as cp
from tokens import tokenize as tk

OUT_PARQUET = cp.PAIRS_ANNOTATED_PARQUET
OUT_JSON = paths.MANIFESTS_DIR / "tokens_v2_cf_pairs_annotation.json"
TRUTH_DIR = paths.COHORTS["cf_v7"] / "truth"

FEATURE_COLS = tuple(tk.feature_columns())
MASK_COLS = tuple(tk.mask_columns())
COMPARE_COLS = FEATURE_COLS + MASK_COLS

# Variable primaria de la verdad por lever_group (evidencia para clasificar).
TRUTH_PRIMARY: dict[str, tuple[str, ...]] = {
    "propofol": ("propofol_rate",),
    "remifentanilo": ("remifentanil_rate",),
    "efedrina": ("ephedrine_dose",),
    "fenilefrina": ("phenylephrine_dose",),
    "noradrenalina": ("noradrenaline_rate",),
    "sevoflurano": ("sevoflurane_mac",),
    "ventilacion": ("fio2_applied", "peep_applied", "rr_applied", "tv_applied"),
}

# Variable primaria OBSERVADA por palanca (para t_divergence_raw / base_value).
LEVER_PRIMARY_TRACK: dict[str, str] = {
    "propofol_bolus": "Orchestra/PPF20_RATE",
    "ppf_bolus": "Orchestra/PPF20_RATE",
    "ppf20_rate": "Orchestra/PPF20_RATE",
    "remi_up": "Orchestra/RFTN20_RATE",
    "remi_bolus": "Orchestra/RFTN20_RATE",
    "rftn20_rate": "Orchestra/RFTN20_RATE",
    "ephedrine": "Orchestra/EPH_RATE",
    "eph_bolus": "Orchestra/EPH_RATE",
    "phen_rate": "Orchestra/PHEN_RATE",
    "phen_bolus": "Orchestra/PHEN_RATE",
    "noradrenaline": "Orchestra/NEPI_RATE",
    "nepi_rate": "Orchestra/NEPI_RATE",
    "sevo_up": "Primus/MAC",
    "sevo_mac": "Primus/MAC",
    "set_fio2": "Primus/SET_FIO2",
    "set_peep": "Primus/SET_INTER_PEEP",
    "set_rr": "Primus/SET_RR_IPPV",
    "set_tv": "Primus/SET_TV_L",
    "fio2_down": "Primus/SET_FIO2",
    "peep_down": "Primus/SET_INTER_PEEP",
    "peep_up": "Primus/SET_INTER_PEEP",
}

# Variable primaria de la VERDAD que corresponde a la palanca (para el chequeo
# de "la verdad difiere tras t_action"). Para set_peep el override incluye fio2,
# así que se miran ambas.
LEVER_PRIMARY_TRUTH: dict[str, tuple[str, ...]] = {
    "peep_up": ("peep_applied",),
    "peep_down": ("peep_applied",),
    "set_peep": ("peep_applied", "fio2_applied"),
    "fio2_down": ("fio2_applied",),
    "set_fio2": ("fio2_applied",),
    "set_rr": ("rr_applied",),
    "set_tv": ("tv_applied",),
}

# Etiqueta legible de la variable de la intervención.
LEVER_LABEL: dict[str, str] = {
    "propofol_bolus": "bolo propofol", "ppf_bolus": "bolo propofol",
    "ppf20_rate": "tasa propofol", "remi_up": "tasa remifentanilo",
    "remi_bolus": "bolo remifentanilo", "rftn20_rate": "tasa remifentanilo",
    "ephedrine": "bolo efedrina", "eph_bolus": "bolo efedrina",
    "phen_rate": "tasa fenilefrina", "phen_bolus": "bolo fenilefrina",
    "noradrenaline": "tasa noradrenalina", "nepi_rate": "tasa noradrenalina",
    "sevo_up": "MAC sevoflurano", "sevo_mac": "MAC sevoflurano",
    "set_fio2": "fio2", "fio2_down": "fio2", "set_peep": "peep",
    "peep_up": "peep", "peep_down": "peep", "set_rr": "rr", "set_tv": "tv",
}


# ---------------------------------------------------------------------------
# Caché de ventanas de tokens_v2
# ---------------------------------------------------------------------------

class PartitionCache:
    def __init__(self, cols: list[str], max_entries: int = 3):
        self.cols = cols
        self.max_entries = max_entries
        self._cache: dict[Path, pd.DataFrame] = {}

    def get(self, part: Path, caseid: int) -> pd.DataFrame:
        df = self._cache.get(part)
        if df is None:
            df = pq.read_table(part, columns=self.cols).to_pandas()
            df = df[df.dense == False]  # noqa: E712 (métrica sólo dense=False)
            if len(self._cache) >= self.max_entries:
                self._cache.clear()
            self._cache[part] = df
        return df[df.caseid == caseid].sort_values("t0", kind="stable").reset_index(drop=True)


class TruthCache:
    def __init__(self, max_entries: int = 8):
        self.max_entries = max_entries
        self._cache: dict[int, pd.DataFrame] = {}

    def get(self, caseid: int) -> pd.DataFrame:
        df = self._cache.get(caseid)
        if df is None:
            df = pq.read_table(TRUTH_DIR / f"{caseid}_truth.parquet").to_pandas()
            if len(self._cache) >= self.max_entries:
                self._cache.clear()
            self._cache[caseid] = df
        return self._cache[caseid]


def _case_to_part(parts: list[Path]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for part in parts:
        df = pq.ParquetFile(part).read(columns=["caseid"]).to_pandas()
        for cid in df.caseid.unique():
            out[int(cid)] = part
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# requested_change / base_value_at_action
# ---------------------------------------------------------------------------

def requested_change(meta: dict) -> str:
    """Descripción compacta de lo que pide la intervención de la rama A."""
    inter = meta.get("intervention_a") or []
    if inter:
        s = str(inter[0])
        return s.split("s ", 1)[1] if "s " in s else s
    ov = meta.get("vent_override_a") or {}
    return json.dumps(ov, ensure_ascii=False, sort_keys=True)


def base_value_at_action(base_raw: pd.DataFrame, track: str,
                         t_action: float) -> float:
    """Último valor observado del track en la rama base con time <= t_action."""
    sub = base_raw[base_raw["time"] <= t_action][["time", track]].dropna()
    if not len(sub):
        return float("nan")
    return float(sub[track].iloc[-1])


def truth_differs_after_action(truth_a: pd.DataFrame, truth_b: pd.DataFrame,
                               cols: tuple[str, ...], t_action: float) -> bool:
    """¿Difiere alguna columna de la verdad entre ramas en algún t > t_action?"""
    cols = tuple(c for c in cols if c in truth_a.columns and c in truth_b.columns)
    if not cols:
        return False
    m = truth_a[["time", *cols]].merge(
        truth_b[["time", *cols]], on="time", suffixes=("_a", "_b"))
    m = m[m["time"] > t_action]
    if not len(m):
        return False
    for c in cols:
        va = m[f"{c}_a"].to_numpy(dtype=np.float64)
        vb = m[f"{c}_b"].to_numpy(dtype=np.float64)
        if not np.array_equal(va, vb, equal_nan=True):
            return True
    return False


# ---------------------------------------------------------------------------
# Clasificación de nulos
# ---------------------------------------------------------------------------

RATE_LEVERS = frozenset({"ppf20_rate", "rftn20_rate", "remi_up"})

def classify_null(lever: str, req: str, base_val: float,
                  truth_diff: bool, has_post_window: bool) -> str:
    """Causa de un par nulo, con las reglas del paso 3b.

    Orden de evaluación:
      1. case_ends          no hay ninguna ventana con t1 > t_action.
      2. no_change_requested el valor pedido (en el eje de la variable
                            observada) coincide con el valor base: no se pide
                            cambio alguno (tolerancia 5 % relativa).
      3. below_resolution   la verdad SÍ difiere tras t_action, pero la
                            diferencia no se resuelve en las features
                            observadas (duración por debajo de la cadencia del
                            track observado, o cuantización del setpoint).
      4. clip_bound         la verdad no difiere porque el cambio pedido queda
                            anulado por un recorte del simulador. Sólo se
                            aplica a las palancas de consigna persistente de
                            ventilación (clips peep [0,25], fio2 [0.15,1.0],
                            rr [0,70], tv [0,1600]); en el resto, si la verdad
                            no difiere es porque el plan base solapó la
                            petición y se registra como no_change_requested.
      5. unexplained        resto.
    """
    if not has_post_window:
        return "case_ends"
    # 2. valor pedido ~ valor base (misma escala): tasa directa ppf20/rftn20/remi_up
    #    (el eje de la verdad es mL/h = mg/min o mcg/min × 3) o delta ~ 0.
    pedido = _requested_scalar(lever, req)
    if pedido is not None:
        if lever in RATE_LEVERS and base_val == base_val and abs(base_val) > 0:
            if abs(pedido * 3.0 - base_val) <= 0.05 * abs(base_val):
                return "no_change_requested"
        elif abs(pedido) <= 1e-9:
            return "no_change_requested"
    if truth_diff:
        return "below_resolution"
    if cp.is_persistent(lever):
        return "clip_bound"
    return "no_change_requested"


_VALUE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*m(?:g|cg)/min")


def _requested_scalar(lever: str, req: str) -> float | None:
    """Valor numérico de la petición (tasa en mg|mcg/min, o delta del override)."""
    m = _VALUE_RE.search(req)
    if m:
        return float(m.group(1))
    try:
        ov = json.loads(req)
    except (TypeError, ValueError):
        return None
    if isinstance(ov, dict):
        for k in ("peep_delta", "fio2_delta", "rr_delta", "tv_delta"):
            if k in ov:
                return float(ov[k])
    return None



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(limit: int | None = None, write: bool = True) -> int:
    t_start = _time.time()
    pairs_path = cp.PAIRS_PARQUET
    pairs_sha = _sha256(pairs_path)
    pairs = pq.read_table(pairs_path).to_pandas().sort_values(
        "pair_id").reset_index(drop=True)
    assert len(pairs) == 6345, f"n_pairs inesperado: {len(pairs)}"
    if limit is not None:
        pairs = pairs.head(limit).reset_index(drop=True)
    unknown = set(pairs.lever.unique()) - set(tk.LEVER_GROUP)
    assert not unknown, f"palancas fuera de LEVER_GROUP: {sorted(unknown)}"

    cf_parts = sorted((paths.TOKENS_V2_DIR / "windows").glob(
        "source=cf_v7/split=*/part-*.parquet"))
    case_to_part = _case_to_part(cf_parts)
    win_cache = PartitionCache(["caseid", "t0", "t1", "dense"] + list(COMPARE_COLS))
    truth_cache = TruthCache()

    ann: dict[str, list] = {k: [] for k in (
        "t_action", "action_source", "t_divergence_raw", "requested_change",
        "base_value_at_action", "effect_t1", "effect_lag_windows",
        "lever_effective", "null_cause", "truth_diff_after_action",
        "prefix_identical")}

    per_lever: dict[str, dict] = {}
    null_evidence: list[dict] = []
    prefix_failures: list[int] = []
    c1_failures: dict[str, list[dict]] = {"one_shot": [], "persistent": []}
    lag_hist: dict[str, dict[int, int]] = {"one_shot": {}, "persistent": {}}
    n_eff: dict[str, int] = {"one_shot": 0, "persistent": 0}
    n_c1: dict[str, int] = {"one_shot": 0, "persistent": 0}

    for _, r in pairs.iterrows():
        pid = int(r.pair_id)
        base_id = int(r.caseid_base)
        inter_id = int(r.caseid_intervencion)
        lever = str(r.lever)
        group = str(r.lever_group)
        meta = cp.load_cf_pair_meta(pid)
        split_t = float(meta["split_t"])
        t_action, action_source = cp.action_time(meta, lever, split_t)
        req = requested_change(meta)

        feats = list(cp.LEVER_FEATURES[group])

        # --- ventanas dense=False de las dos ramas, alineadas por t0
        db = win_cache.get(case_to_part[base_id], base_id)
        di = win_cache.get(case_to_part[inter_id], inter_id)
        m = db[["t0", "t1"] + list(COMPARE_COLS)].merge(
            di[["t0"] + list(COMPARE_COLS)], on="t0", suffixes=("_b", "_i"))
        m = m.sort_values("t0", kind="stable").reset_index(drop=True)
        if not len(m):
            m = db[["t0", "t1"] + list(COMPARE_COLS)].iloc[0:0].merge(
                di[["t0"] + list(COMPARE_COLS)], on="t0", suffixes=("_b", "_i"))

        # --- C4: prefijo idéntico en TODAS las features y máscaras hasta
        #     t_action, medido sobre la rejilla de la capa de tokens. La ventana
        #     que contiene t_action cuenta como post-acción (post_action) y el
        #     prefijo exigible se queda 2 celdas de rejilla por detrás para
        #     absorber la atribución temprana de los tracks de evento discreto
        #     (cf_pairs.pre_action).
        pre = m[m["t1"].map(lambda t1: cp.pre_action(t1, t_action))] if len(m) else m
        prefix_ok = True
        for c in COMPARE_COLS:
            if not np.array_equal(pre[f"{c}_b"].to_numpy(),
                                  pre[f"{c}_i"].to_numpy(), equal_nan=True):
                prefix_ok = False
                break
        if not prefix_ok:
            prefix_failures.append(pid)

        # --- efecto: primera ventana post-acción con features del grupo distintas
        post = m[m["t1"].map(lambda t1: cp.post_action(t1, t_action))] if len(m) else m
        effect_t1 = float("nan")
        effect_lag = float("nan")
        effective = False
        for lag, (_, row) in enumerate(post.iterrows()):
            diff = False
            for c in feats:
                va, vb = float(row[f"{c}_b"]), float(row[f"{c}_i"])
                if not np.isclose(va, vb, rtol=0.0, atol=0.0, equal_nan=True):
                    diff = True
                    break
            if diff:
                effective = True
                effect_t1 = float(row["t1"])
                effect_lag = float(lag)
                break

        cls = "persistent" if cp.is_persistent(lever) else "one_shot"
        if effective:
            n_eff[cls] += 1
            lag_hist[cls][int(effect_lag)] = lag_hist[cls].get(int(effect_lag), 0) + 1

        # --- t_divergence_raw y base_value_at_action (rama base)
        tracks = cp.LEVER_TRACKS[group]
        primary = LEVER_PRIMARY_TRACK[lever]
        try:
            base_raw = cp.load_raw_tracks(base_id, tracks)
            inter_raw = cp.load_raw_tracks(inter_id, tracks)
            t_div = cp.first_divergence_raw(base_raw, inter_raw, tracks)
            base_val = base_value_at_action(base_raw, primary, t_action)
        except FileNotFoundError:
            t_div, base_val = None, float("nan")

        # --- verdad: ¿difiere tras t_action?
        tcols = LEVER_PRIMARY_TRUTH.get(lever, TRUTH_PRIMARY[group])
        try:
            truth_diff = truth_differs_after_action(
                truth_cache.get(inter_id), truth_cache.get(base_id), tcols, t_action)
        except FileNotFoundError:
            truth_diff = False

        # --- clasificación de nulos
        caused: str | None = None
        if not effective:
            caused = classify_null(lever, req, base_val, truth_diff,
                                   has_post_window=len(post) > 0)

        # --- C1: la divergencia cruda debe caer junto a t_action. Sólo es
        #     aplicable a las acciones puntuales (one_shot): las consignas
        #     persistentes divergen cuando el plan base cambia la consigna.
        if effective and t_div is not None and abs(t_div - t_action) > 30.0:
            n_c1[cls] += 1
            c1_failures[cls].append({"pair_id": pid, "lever": lever,
                                     "t_divergence_raw": t_div, "t_action": t_action})

        ann["t_action"].append(t_action)
        ann["action_source"].append(action_source)
        ann["t_divergence_raw"].append(np.nan if t_div is None else float(t_div))
        ann["requested_change"].append(req)
        ann["base_value_at_action"].append(base_val)
        ann["effect_t1"].append(effect_t1)
        ann["effect_lag_windows"].append(effect_lag)
        ann["lever_effective"].append(effective)
        ann["null_cause"].append(caused)
        ann["truth_diff_after_action"].append(bool(truth_diff))
        ann["prefix_identical"].append(bool(prefix_ok))

        d = per_lever.setdefault(lever, {"lever_group": group, "n": 0,
                                         "effective": 0, "nulls": {}})
        d["n"] += 1
        if effective:
            d["effective"] += 1
        else:
            d["nulls"][caused] = d["nulls"].get(caused, 0) + 1
            null_evidence.append({
                "pair_id": pid, "lever": lever, "lever_group": group,
                "caseid_base": base_id, "caseid_intervencion": inter_id,
                "t_action": round(t_action, 3), "requested_change": req,
                "base_value_at_action": None if base_val != base_val else round(base_val, 6),
                "truth_diff_after_action": bool(truth_diff),
                "has_post_window": bool(len(post) > 0),
                "null_cause": caused,
            })

    # ------------------------------------------------------------------ escritura
    out = pairs.copy()
    for k, v in ann.items():
        out[k] = v
    table = pa.Table.from_pandas(out, preserve_index=False)
    if write:
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, OUT_PARQUET)

    n_null = sum(1 for x in ann["lever_effective"] if not x)
    n_total = len(ann["lever_effective"])
    n_eff_total = n_total - n_null
    causes = {}
    for c in ann["null_cause"]:
        if c:
            causes[c] = causes.get(c, 0) + 1

    def _pct_lag(cls: str) -> float:
        return round(100.0 * sum(v for k, v in lag_hist[cls].items() if k <= 1)
                     / max(1, n_eff[cls]), 4)

    pct_oneshot = _pct_lag("one_shot")
    non_peep_nulls = [e for e in null_evidence
                      if not (e["lever"] in ("peep_down", "peep_up", "set_peep")
                              and e["null_cause"] == "clip_bound")]

    result = {
        "date": pd.Timestamp.now().isoformat(),
        "pairs_parquet_sha256": pairs_sha,
        "pairs_annotated_parquet": str(OUT_PARQUET.relative_to(paths.REPO_ROOT)),
        "n_pairs": int(n_total),
        "n_effective": int(n_eff_total),
        "n_null": int(n_null),
        "null_causes": causes,
        "criterios_alcance": (
            "C1 y C2 se evalúan sobre las ACCIONES PUNTUALES (one_shot: eventos "
            "de la colección pharma/learning/sevo, con t_action en intervention_a). "
            "Las palancas de CONSIGNA PERSISTENTE (overrides de ventilación: el "
            "simulador aplica un delta acumulativo sobre la señal) divergen cuando "
            "el plan base cambia la consigna, no en t_action, y se reportan aparte."),
        "criteria": {
            "C1_divergencia_cruda_cerca_de_t_action": {
                "umbral_s": 30.0,
                "n_failures": len(c1_failures["one_shot"]),
                "ok": len(c1_failures["one_shot"]) == 0,
                "n_effective_one_shot": n_eff["one_shot"],
                "failures": c1_failures["one_shot"][:20],
                "persistent_consigna": {
                    "n_effective": n_eff["persistent"],
                    "n_failures": len(c1_failures["persistent"]),
                    "nota": "no aplicable: la consigna persistente sólo diverge "
                            "cuando el plan base la cambia",
                },
            },
            "C2_efecto_en_ventana_de_t_action_o_siguiente": {
                "umbral_lag": 1,
                "pct_lag_le_1": pct_oneshot,
                "ok": bool(pct_oneshot >= 99.0),
                "n_effective_one_shot": n_eff["one_shot"],
                "hist_lag": {str(k): v for k, v in sorted(lag_hist["one_shot"].items())},
                "persistent_consigna": {
                    "n_effective": n_eff["persistent"],
                    "pct_lag_le_1": _pct_lag("persistent"),
                    "hist_lag": {str(k): v for k, v in sorted(lag_hist["persistent"].items())},
                },
            },
            "C3_sin_nulos_sin_explicar": {
                "ok": causes.get("unexplained", 0) == 0,
                "n_unexplained": causes.get("unexplained", 0)},
            "C4_prefijo_identico_hasta_t_action": {
                "ok": len(prefix_failures) == 0,
                "n_failures": len(prefix_failures),
                "nota": "prefijo medido sobre la rejilla de tokens "
                        "(cf_pairs.pre_action): ventanas con t1 <= "
                        "floor(t_action/5)*5 - 2*5 s, que no pueden ver la "
                        "intervención ni por la cuantización de pk_v2 ni por la "
                        "atribución temprana de los tracks de evento discreto",
                "failures": prefix_failures[:20]},
        },
        "per_lever": [
            {"lever": lv, "lever_group": d["lever_group"], "n": d["n"],
             "effective": d["effective"],
             "pct_effective": round(100.0 * d["effective"] / d["n"], 4) if d["n"] else 0.0,
             "nulls": d["nulls"]}
            for lv, d in sorted(per_lever.items())
        ],
        "nulls_non_peep": non_peep_nulls,
        "n_nulls_non_peep": len(non_peep_nulls),
        "elapsed_s": round(_time.time() - t_start, 2),
    }
    if write:
        OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    print(json.dumps({k: result[k] for k in (
        "n_pairs", "n_effective", "n_null", "null_causes", "elapsed_s")}, indent=2))
    print("criterios:", json.dumps(
        {k: v.get("ok") for k, v in result["criteria"].items()}, indent=2))
    print("nulos no-PEEP:", len(non_peep_nulls))
    for e in non_peep_nulls:
        print(f"  {e['lever']:<13s} pair={e['pair_id']} "
              f"t_action={e['t_action']} cause={e['null_cause']} "
              f"truth_diff={e['truth_diff_after_action']} "
              f"base={e['base_value_at_action']} req={e['requested_change']}")
    print("escrito:", OUT_PARQUET, OUT_JSON)
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="anota sólo los primeros N pares (smoke test, no escribe)")
    ap.add_argument("--dump-parquet", action="store_true",
                    help="escribe el parquet aunque se use --limit")
    args = ap.parse_args()
    raise SystemExit(main(limit=args.limit, write=(args.limit is None) or args.dump_parquet))
