"""PASO 3c — análisis y anotación corregida de los pares CF.

Fase 0 — convención de tiempos (verificada contra window.py, pk_tokens.py y
tokenize.py; se documenta en la sección 1 del informe):

  * La rejilla es ``t = a0 + 5k`` con ``a0 = max(anestart, casestart, t_min)``.
  * Un punto de rejilla ``g`` es el CIERRE de la celda semiabierta ``(g-5, g]``;
    el instante ``g`` pertenece a esa celda.
  * Tracks continuos: ``valor(g)`` = última muestra cruda con ``time <= g``
    (forward-fill hold). Bolos: suma de eventos en ``(g-5, g]``, estampados en
    el primer punto de rejilla ``>= t_evento``.
  * Ventana de tokens: ``t1 = t0 + 60``; sus 12 celdas cierran en
    ``t0+5 .. t0+60`` y cubren ``(t0, t1]``. ``ce_t0 = ce(t0)``.
  * Por tanto "celdas que cierran en o antes de X" = puntos de rejilla
    ``g <= X``, y de ahí que ``pre_action <=> t1 < t_boundary`` y
    ``post_action <=> t1 >= t_boundary`` (la ventana que contiene la celda de
    ``t_boundary`` ya es posterior).

Mide, para cada par de ``data/tokens_v2/pairs.parquet``:

  Fase A  t_effective: t_action (acción puntual) o t_divergence_raw (consigna
          persistente efectiva); NaN si el par es nulo. Clasificación de las
          consignas persistentes en inmediatas / retrasadas por recorte, con
          evidencia del recorte.
  Fase B  B1 (bloqueante): las 14 variables de la imagen de windows_v4 (valores
          y máscaras) idénticas entre ramas en toda celda que cierre en o antes
          de t_effective. B2 (informativo): lo mismo sobre los tracks crudos.
  Fase C  t_div_grid (primer punto de rejilla donde difiere alguna serie que
          alimenta las features del lever_group), lead_s = t_effective −
          t_div_grid, t_boundary = min(t_div_grid, rejilla(t_effective)), y C5
          (prefijo de tokens idéntico).
  Fase D  efecto (features del lever_group) en la ventana que contiene
          t_boundary o la siguiente; lag por palanca.
  Fase E  reclasificación de los pares cuyo efecto desaparece porque el plan
          base sobrescribe la intervención; unidades homogéneas.

Escribe ``data/tokens_v2/pairs_annotated_v2.parquet`` y
``manifests/tokens_v2_cf_pairs_annotation_v2.json``. No toca nada más.

Uso:
  python -m scripts.paso3c_analyze_pairs
"""

from __future__ import annotations

import hashlib
import json
import math
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import paths
from tokens import cf_pairs as cp
from tokens import tokenize as tk

OUT_PARQUET = paths.TOKENS_V2_DIR / "pairs_annotated_v2.parquet"
OUT_JSON = paths.MANIFESTS_DIR / "tokens_v2_cf_pairs_annotation_v2.json"
WINDOWS_V4 = paths.WINDOWS_DIR / "windows"
PK_V2 = paths.PK_V2_DIR / "windows"
TOK_V2 = paths.TOKENS_V2_DIR / "windows"
ART_3B = paths.TOKENS_V2_DIR / "pairs_annotated.parquet"

IMAGE_TRACKS = [
    "BIS/BIS", "Solar8000/HR", "Solar8000/PLETH_SPO2", "Primus/ETCO2",
    "Primus/PEEP_MBAR", "Primus/PIP_MBAR", "Primus/MV", "Primus/TV",
    "Primus/RR_CO2", "Solar8000/BT", "Solar8000/ART_MBP", "Solar8000/ART_SBP",
    "Solar8000/ART_DBP", "BIS/EMG",
]

# Series de la rejilla que alimentan las features del lever_group, por grupo.
PK_SERIES: dict[str, tuple[str, ...]] = {
    "propofol": ("ce_propofol", "bolus_propofol", "dose_cum_propofol", "m_ce_propofol"),
    "remifentanilo": ("ce_remifentanilo", "bolus_remifentanilo",
                      "dose_cum_remifentanilo", "m_ce_remifentanilo"),
    "efedrina": ("ce_efedrina", "bolus_efedrina", "dose_cum_efedrina", "m_ce_efedrina"),
    "fenilefrina": ("ce_fenilefrina", "bolus_fenilefrina", "dose_cum_fenilefrina",
                    "m_ce_fenilefrina"),
    "noradrenalina": ("ce_noradrenalina", "dose_cum_noradrenalina",
                      "m_ce_noradrenalina"),
    "sevoflurano": ("ce_sevoflurano", "m_ce_sevoflurano"),
    "ventilacion": (),
}
WIN_BOLUS: dict[str, tuple[str, ...]] = {
    "propofol": ("ppf_bolus",),
    "remifentanilo": ("remi_bolus",),
    "efedrina": ("eph_bolus",),
    "fenilefrina": ("phen_bolus",),
    "noradrenalina": (),
    "sevoflurano": (),
    "ventilacion": (),
}

VENT_SETPOINT_COLS = tk.VENT_SETPOINT_COLS  # fio2/tv/rr/pip/peep -> Primus/SET_*

# columnas crudas realmente usadas: B2 (imagen), evidencia A2 (consignas de
# ventilación) y E1 (ritmos de infusión de ppf/remi).
RAW_COLS = (["time"] + list(IMAGE_TRACKS) + sorted(set(VENT_SETPOINT_COLS.values()))
            + ["Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"])


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def floor_grid(x: float, g: float = cp.GRID_S) -> float:
    return math.floor(float(x) / g) * g


class Cache:
    def __init__(self, cols: list[str], max_entries: int = 4, dense_only: bool = False):
        self.cols = cols
        self.max = max_entries
        self.dense_only = dense_only
        self._d: dict[Path, pd.DataFrame] = {}

    def get(self, part: Path, caseid: int) -> pd.DataFrame:
        df = self._d.get(part)
        if df is None:
            df = pq.read_table(part, columns=self.cols).to_pandas()
            if self.dense_only:
                df = df[~df.dense]
            if len(self._d) >= self.max:
                self._d.clear()
            self._d[part] = df
        return df[df.caseid == caseid].sort_values("t" if "t" in df.columns else "t0",
                                                  kind="stable").reset_index(drop=True)


class RawCache:
    """Parquets de caso crudos. Lee únicamente las columnas necesarias."""
    def __init__(self, cols: list[str], max_entries: int = 24):
        self.cols = list(cols)
        self.max = max_entries
        self._d: dict[int, pd.DataFrame] = {}
        self._c: dict[Path, list[str]] = {}

    def _cols_for(self, p: Path) -> list[str]:
        cc = self._c.get(p)
        if cc is None:
            have = set(pq.ParquetFile(p).schema_arrow.names)
            cc = [c for c in self.cols if c in have]
            self._c[p] = cc
        return cc

    def get(self, caseid: int) -> pd.DataFrame:
        df = self._d.get(caseid)
        if df is None:
            p = cp.CF_CASES_DIR / f"{caseid}.parquet"
            df = pq.read_table(p, columns=self._cols_for(p)).to_pandas()
            if len(self._d) >= self.max:
                self._d.clear()
            self._d[caseid] = df
        return self._d[caseid]


def _case_to_part(parts: list[Path]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in parts:
        df = pq.ParquetFile(p).read(columns=["caseid"]).to_pandas()
        for cid in df.caseid.unique():
            out[int(cid)] = p
    return out


def first_diff_grid(a: dict[str, np.ndarray], b: dict[str, np.ndarray],
                    t: np.ndarray) -> tuple[float, str] | None:
    """Primer punto de rejilla donde difiere alguna serie (NaN-aware).

    Devuelve ``(t_rejilla, nombre_de_la_serie)``: saber QUÉ serie fija la
    frontera es imprescindible para explicar los lag >= 2 de la fase D.
    """
    best: tuple[float, str] | None = None
    for k in a:
        va, vb = a[k], b[k]
        diff = ~((va == vb) | (np.isnan(va) & np.isnan(vb)))
        if diff.any():
            cand = (float(t[int(np.argmax(diff))]), k)
            if best is None or cand[0] < best[0]:
                best = cand
    return best


def setpoints_on_grid(raw: pd.DataFrame, grid: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    t_raw = raw["time"].to_numpy(np.float64)
    for item, col in VENT_SETPOINT_COLS.items():
        if col not in raw.columns:
            continue
        v = raw[col].to_numpy(np.float64)
        non = np.isfinite(v)
        if not non.any():
            continue
        tt, vv = t_raw[non], v[non]
        idx = np.clip(np.searchsorted(tt, grid, side="right") - 1, 0, len(vv) - 1)
        ok = np.searchsorted(tt, grid, side="right") - 1 >= 0
        arr = np.where(ok, vv[idx], np.nan)
        out[col] = arr
    return out


def main() -> int:  # noqa: C901
    t_start = _time.time()
    pairs = pq.read_table(cp.PAIRS_PARQUET).to_pandas().sort_values("pair_id")
    ann3b = pq.read_table(ART_3B).to_pandas().set_index("pair_id")
    assert len(pairs) == 6345

    v4_parts = sorted(WINDOWS_V4.glob("source=cf_v7/split=*/part-*.parquet"))
    pk_parts = sorted(PK_V2.glob("**/part-*.parquet"))
    tok_parts = sorted(TOK_V2.glob("source=cf_v7/split=*/part-*.parquet"))
    v4_map = _case_to_part(v4_parts)
    pk_map = _case_to_part(pk_parts)
    tok_map = _case_to_part(tok_parts)

    v4_cols = ["caseid", "t"] + IMAGE_TRACKS + [f"m_{c}" for c in IMAGE_TRACKS] \
        + [f"m_raw_{c}" for c in IMAGE_TRACKS] + ["ppf_bolus", "remi_bolus",
                                                 "roc_bolus", "phen_bolus", "eph_bolus"]
    pk_cols = ["caseid", "t"] + sorted({c for s in PK_SERIES.values() for c in s})
    tok_cols = ["caseid", "t0", "t1", "dense"] + list(tk.feature_columns()) \
        + list(tk.mask_columns())
    c_v4 = Cache(v4_cols, max_entries=4)
    c_pk = Cache(pk_cols, max_entries=4)
    c_tok = Cache(tok_cols, max_entries=3, dense_only=True)
    raw = RawCache(RAW_COLS, max_entries=24)

    rows: list[dict] = []
    b1_fail: list[dict] = []
    c5_fail: list[dict] = []
    lag_tab: dict[str, dict[int, int]] = {}
    a2_rows: list[dict] = []
    e1_rows: list[dict] = []

    for r in pairs.itertuples():
        pid = int(r.pair_id)
        lever = str(r.lever)
        group = str(r.lever_group)
        a_cid, b_cid = int(r.caseid_intervencion), int(r.caseid_base)
        meta = cp.load_cf_pair_meta(pid)
        split_t = float(meta["split_t"])
        t_action, action_source = cp.action_time(meta, lever, split_t)
        eff3b = bool(ann3b.loc[pid, "lever_effective"])
        t_div_raw = ann3b.loc[pid, "t_divergence_raw"]
        t_div_raw = float(t_div_raw) if np.isfinite(t_div_raw) else None
        persistent = cp.is_persistent(lever)

        # ---------------- Fase A: t_effective ----------------
        if not eff3b:
            t_eff, a2 = float("nan"), "nulo"
        elif not persistent:
            t_eff, a2 = float(t_action), "one_shot"
        else:
            t_eff = float(t_div_raw) if t_div_raw is not None else float("nan")
            a2 = ("inmediato" if (t_div_raw is not None
                                  and abs(t_div_raw - t_action) <= 30.0)
                  else "retrasado")

        # ---------------- rejillas ----------------
        wa = c_v4.get(v4_map[a_cid], a_cid)
        wb = c_v4.get(v4_map[b_cid], b_cid)
        m = wa.merge(wb, on="t", suffixes=("_a", "_b"))
        g = m["t"].to_numpy(np.float64)
        pa_cid = pk_map.get(a_cid)
        pb_cid = pk_map.get(b_cid)
        ka = c_pk.get(pa_cid, a_cid) if pa_cid is not None else pd.DataFrame()
        kb = c_pk.get(pb_cid, b_cid) if pb_cid is not None else pd.DataFrame()
        mk = ka.merge(kb, on="t", suffixes=("_a", "_b")) if len(ka) and len(kb) else None

        # ---------------- Fase B1: prefijo fisiológico estricto ----------------
        b1_ok = True
        B1_col = None
        B1_t = None
        sel = g <= t_eff if np.isfinite(t_eff) else np.zeros(len(g), bool)
        if np.isfinite(t_eff):
            for c in IMAGE_TRACKS:
                for fam in ("", "m_", "m_raw_"):
                    col = f"{fam}{c}"
                    va = m[f"{col}_a"].to_numpy(np.float64)
                    vb = m[f"{col}_b"].to_numpy(np.float64)
                    d = ~((va == vb) | (np.isnan(va) & np.isnan(vb)))
                    d = d & sel
                    if d.any():
                        b1_ok = False
                        B1_col, B1_t = col, float(g[int(np.argmax(d))])
                        break
                if not b1_ok:
                    break
        if not b1_ok:
            b1_fail.append({"pair_id": pid, "lever": lever, "col": B1_col,
                            "t": B1_t, "t_effective": t_eff,
                            "imagen_menos_t_action": round(B1_t - t_action, 3),
                            "imagen_menos_t_effective": round(B1_t - t_eff, 3)})

        # ---------------- Fase B2 (informativo): crudos ----------------
        b2_ok = True
        if np.isfinite(t_eff):
            ra, rb = raw.get(a_cid), raw.get(b_cid)
            mr = ra.merge(rb, on="time", suffixes=("_a", "_b"))
            mr = mr[mr["time"] <= t_eff]
            for c in IMAGE_TRACKS:
                if f"{c}_a" not in mr.columns:
                    continue
                va = mr[f"{c}_a"].to_numpy(np.float64)
                vb = mr[f"{c}_b"].to_numpy(np.float64)
                if not np.array_equal(va, vb, equal_nan=True):
                    b2_ok = False
                    break

        # ---------------- Fase C1: t_div_grid y adelanto ----------------
        ser_a: dict[str, np.ndarray] = {}
        ser_b: dict[str, np.ndarray] = {}
        if mk is not None:
            gk = mk["t"].to_numpy(np.float64)
            for c in PK_SERIES[group]:
                ser_a[c] = mk[f"{c}_a"].to_numpy(np.float64)
                ser_b[c] = mk[f"{c}_b"].to_numpy(np.float64)
        else:
            gk = np.zeros(0)
        for c in WIN_BOLUS[group]:
            ser_a[c] = m[f"{c}_a"].to_numpy(np.float64)
            ser_b[c] = m[f"{c}_b"].to_numpy(np.float64)
        if group == "ventilacion":
            ga = setpoints_on_grid(raw.get(a_cid), g)
            gb = setpoints_on_grid(raw.get(b_cid), g)
            for c in ga:
                if c in gb:
                    ser_a[c] = ga[c]
                    ser_b[c] = gb[c]

        t_div_grid = None
        t_div_grid_col = None
        if mk is not None:
            td = first_diff_grid({c: ser_a[c] for c in PK_SERIES[group]},
                                 {c: ser_b[c] for c in PK_SERIES[group]}, gk)
            if td is not None:
                t_div_grid, t_div_grid_col = td
        td2 = first_diff_grid({c: v for c, v in ser_a.items() if c not in PK_SERIES[group]},
                              {c: v for c, v in ser_b.items() if c not in PK_SERIES[group]}, g)
        if td2 is not None and (t_div_grid is None or td2[0] < t_div_grid):
            t_div_grid, t_div_grid_col = td2

        lead = (t_eff - t_div_grid) if (np.isfinite(t_eff) and t_div_grid is not None) else np.nan
        t_boundary = (min(t_div_grid, floor_grid(t_eff))
                      if (np.isfinite(t_eff) and t_div_grid is not None)
                      else (floor_grid(t_eff) if np.isfinite(t_eff) else np.nan))

        # ---------------- Fase C5 y D: ventanas de tokens ----------------
        pre_ok, eff_v2, lag, eff_t1 = True, False, np.nan, np.nan
        feats = list(cp.LEVER_FEATURES[group])
        if np.isfinite(t_eff) and eff3b:
            ta = c_tok.get(tok_map[a_cid], a_cid)
            tb = c_tok.get(tok_map[b_cid], b_cid)
            mm = ta.merge(tb, on="t0", suffixes=("_a", "_b")).sort_values("t0")
            # Convención 0.1: la celda (g-5, g] cierra en g y g le pertenece.
            # pre_action  <=> todas las celdas cierran estrictamente antes de
            #                t_boundary  <=> t1 < t_boundary
            # post_action <=> contiene la celda de t_boundary (o posterior)
            #                <=> t1 >= t_boundary
            pre = mm[mm["t1_a"].to_numpy(np.float64) < t_boundary]
            for c in list(tk.feature_columns()) + list(tk.mask_columns()):
                if c in ("cf_role",):
                    continue
                va = pre[f"{c}_a"].to_numpy(np.float64)
                vb = pre[f"{c}_b"].to_numpy(np.float64)
                if not np.array_equal(va, vb, equal_nan=True):
                    pre_ok = False
                    c5_fail.append({"pair_id": pid, "lever": lever, "col": c})
                    break
            post = mm[mm["t1_a"].to_numpy(np.float64) >= t_boundary]
            for i, (_, row) in enumerate(post.iterrows()):
                if any(not np.isclose(float(row[f"{c}_a"]), float(row[f"{c}_b"]),
                                      rtol=0.0, atol=0.0, equal_nan=True) for c in feats):
                    eff_v2, lag, eff_t1 = True, float(i), float(row["t1_a"])
                    break

        # ---------------- Fase E1: evidencia de sobrescritura ----------------
        if (str(ann3b.loc[pid, "null_cause"]) == "below_resolution"
                and lever in ("rftn20_rate", "remi_up", "ppf20_rate")):
            rate_col = {"propofol": "Orchestra/PPF20_RATE",
                        "remifentanilo": "Orchestra/RFTN20_RATE"}[group]
            ra = raw.get(a_cid)[["time", rate_col]].dropna()
            rb = raw.get(b_cid)[["time", rate_col]].dropna()
            # cambios de la rama intervenida a partir de t_action
            cha = ra[ra["time"] >= t_action].reset_index(drop=True)
            chg = cha[cha[rate_col].to_numpy() != cha[rate_col].shift().to_numpy()]
            first = chg.iloc[0] if len(chg) else None
            # ¿el valor nuevo aparece igual en la base (plan base lo sobrescribe)?
            base_same = None
            if first is not None:
                tb_ = rb[rb["time"] <= float(first["time"])]
                base_same = (round(float(tb_[rate_col].iloc[-1]), 4)
                             if len(tb_) else None)
            # mantenimiento de la diferencia: primer instante posterior a t_action
            # en el que las dos series vuelven a coincidir y ya no se separan.
            mant_s, t_reconv = None, None
            if first is not None:
                mgo = (ra.merge(rb, on="time", suffixes=("_a", "_b")))
                mgo = mgo[mgo["time"] >= t_action].reset_index(drop=True)
                neq = (mgo[f"{rate_col}_a"].to_numpy(np.float64)
                       != mgo[f"{rate_col}_b"].to_numpy(np.float64))
                if neq.any():
                    last = int(np.argmax(neq[::-1]))  # último índice con diferencia
                    last = len(neq) - 1 - last
                    t_reconv = round(float(mgo["time"].iloc[min(last + 1, len(mgo) - 1)]), 2)
                    mant_s = round(t_reconv - float(first["time"]), 2)
            # ¿son idénticas las series observadas desde t_action?
            mgr = ra.merge(rb, on="time", suffixes=("_a", "_b"))
            mgr = mgr[mgr["time"] >= t_action]
            ident = bool(np.array_equal(mgr[f"{rate_col}_a"].to_numpy(np.float64),
                                        mgr[f"{rate_col}_b"].to_numpy(np.float64),
                                        equal_nan=True))
            e1_rows.append({
                "pair_id": pid, "lever": lever, "t_action": round(float(t_action), 2),
                "peticion": str(meta["intervention_a"]),
                "primer_cambio_intervencion_t": (round(float(first["time"]), 2)
                                                 if first is not None else None),
                "primer_cambio_intervencion_v": (round(float(first[rate_col]), 4)
                                                 if first is not None else None),
                "delta_s": (round(float(first["time"]) - t_action, 2)
                            if first is not None else None),
                "base_v_en_ese_instante": base_same,
                "series_identicas_desde_t_action": ident,
                "t_reconvergencia": t_reconv,
                "mantenimiento_s": mant_s,
            })

        # ---------------- Fase A2: evidencia del recorte ----------------
        if a2 == "retrasado" and t_div_grid is not None:
            ov = dict(meta.get("vent_override_a") or {})
            base_at_action = None
            if group == "ventilacion":
                key = ("peep" if "peep" in lever else
                       "fio2" if "fio2" in lever else
                       "rr" if "rr" in lever else "tv")
                col = VENT_SETPOINT_COLS[key]
                gb2 = setpoints_on_grid(raw.get(b_cid), g)
                ga2 = setpoints_on_grid(raw.get(a_cid), g)
                if col in gb2:
                    selb = g <= t_action
                    selb2 = g <= t_div_grid
                    base_at_action = (float(gb2[col][selb][-1]) if selb.any() else None)
                    base_at_eff = (float(gb2[col][selb2][-1])
                                   if selb2.any() else None)
                    appl_at_eff = (float(ga2[col][selb2][-1])
                                   if selb2.any() else None)
                else:
                    base_at_eff = appl_at_eff = None
            else:
                base_at_eff = appl_at_eff = None
            a2_rows.append({
                "pair_id": pid, "lever": lever, "t_action": t_action,
                "t_divergence_raw": t_div_raw, "retraso_s": round(t_div_raw - t_action, 2),
                "override": ov, "base_setpoint_at_action": base_at_action,
                "base_setpoint_at_effective": base_at_eff,
                "intervencion_setpoint_at_effective": appl_at_eff,
            })

        lag_tab.setdefault(lever, {})
        if eff_v2:
            lag_tab[lever][int(lag)] = lag_tab[lever].get(int(lag), 0) + 1

        rows.append({
            "pair_id": pid, "lever": lever, "lever_group": group,
            "persistent": persistent, "action_source": action_source,
            "t_action": float(t_action), "t_divergence_raw": t_div_raw,
            "t_effective": t_eff, "a2_class": a2,
            "t_div_grid": np.nan if t_div_grid is None else t_div_grid,
            "t_div_grid_col": t_div_grid_col,
            "lead_s": lead, "t_boundary": t_boundary,
            "b1_prefix_ok": b1_ok, "b1_first_col": B1_col, "b1_first_t": B1_t,
            "b2_raw_prefix_ok": b2_ok,
            "c5_pre_ok": pre_ok,
            "lever_effective": eff3b, "lever_effective_v2": eff_v2,
            "effect_lag_boundary": lag, "effect_t1_boundary": eff_t1,
        })
        if (len(rows) % 1000) == 0:
            print(f"  {len(rows)} pares... {_time.time()-t_start:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    # ``pairs`` ya trae lever/lever_group: se descartan para no duplicar
    # columnas con sufijos _x/_y en el artefacto anotado.
    out = pairs.drop(columns=["lever", "lever_group"], errors="ignore") \
               .merge(df, on="pair_id", how="left")
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), OUT_PARQUET)

    n_eff_v2 = int(df.lever_effective_v2.sum())
    leads = df.lead_s.dropna()
    res = {
        "date": pd.Timestamp.now().isoformat(),
        "pairs_parquet_sha256": _sha256(cp.PAIRS_PARQUET),
        "pairs_annotated_3b_sha256": _sha256(ART_3B),
        "pairs_annotated_v2_sha256": _sha256(OUT_PARQUET),
        "n_pairs": int(len(df)),
        "n_effective_3b": int(df.lever_effective.sum()),
        "n_effective_v2": n_eff_v2,
        "A": {
            "n_one_shot": int((df.a2_class == "one_shot").sum()),
            "n_inmediato": int((df.a2_class == "inmediato").sum()),
            "n_retrasado": int((df.a2_class == "retrasado").sum()),
            "n_nulo": int((df.a2_class == "nulo").sum()),
            "retrasados": a2_rows,
            "retraso_mediano_s": (float(np.median([x["retraso_s"] for x in a2_rows]))
                                  if a2_rows else None),
            "retraso_max_s": (float(max(x["retraso_s"] for x in a2_rows))
                              if a2_rows else None),
        },
        "B1": {"ok": len(b1_fail) == 0, "n_failures": len(b1_fail),
               "n_efectivos": int(df.lever_effective.sum()),
               "pct_fallo": (round(100.0 * len(b1_fail) / int(df.lever_effective.sum()), 3)
                             if int(df.lever_effective.sum()) else None),
               "clases": {
                   "boundary_t_igual_t_eff": int(
                       (df.loc[~df.b1_prefix_ok, "b1_first_t"]
                        == df.loc[~df.b1_prefix_ok, "t_effective"]).sum()),
                   "gt_0_le_5": int(((df.loc[~df.b1_prefix_ok, "t_effective"]
                                      - df.loc[~df.b1_prefix_ok, "b1_first_t"]) > 0).sum()
                                    - ((df.loc[~df.b1_prefix_ok, "t_effective"]
                                        - df.loc[~df.b1_prefix_ok, "b1_first_t"]) > 5).sum()),
                   "gt_5_le_10": int(((df.loc[~df.b1_prefix_ok, "t_effective"]
                                       - df.loc[~df.b1_prefix_ok, "b1_first_t"]) > 5).sum()
                                     - ((df.loc[~df.b1_prefix_ok, "t_effective"]
                                         - df.loc[~df.b1_prefix_ok, "b1_first_t"]) > 10).sum()),
                   "gt_10": int(((df.loc[~df.b1_prefix_ok, "t_effective"]
                                  - df.loc[~df.b1_prefix_ok, "b1_first_t"]) > 10).sum()),
               },
               "por_palanca": {k: int(v) for k, v in
                               df.loc[~df.b1_prefix_ok].lever.value_counts().items()},
               "min_imagen_menos_t_action_s": (
                   float(min(x["imagen_menos_t_action"] for x in b1_fail))
                   if b1_fail else None),
               "n_imagen_antes_de_t_action": int(
                   sum(1 for x in b1_fail if x["imagen_menos_t_action"] < 0)),
               "failures": b1_fail[:50]},
        "B2": {"ok_raw_prefix": int(df.b2_raw_prefix_ok.sum()),
               "n_failures": int((~df.b2_raw_prefix_ok.fillna(False)).sum())},
        "C": {
            "n_con_div_grid": int(df.t_div_grid.notna().sum()),
            "lead_mediana_s": float(np.median(leads)) if len(leads) else None,
            "lead_p95_s": float(np.percentile(leads, 95)) if len(leads) else None,
            "lead_max_s": float(leads.max()) if len(leads) else None,
            "clases": {
                "lt_5": int((leads < 5).sum()),
                "llamado_5_10": int(((leads >= 5) & (leads <= 10)).sum()),
                "gt_10": int((leads > 10).sum()),
            },
            "hist_leads_s": {str(b): int((leads <= b).sum() - (leads <= a).sum())
                             for a, b in ((-1e9, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.0),
                                          (3.0, 4.0), (4.0, 5.0), (5.0, 5.5), (5.5, 10.0),
                                          (10.0, 1e9))},
            "por_palanca": {str(k): {"n": int(v["n"]), "lead_max_s": v["lead_max_s"]}
                            for k, v in pd.DataFrame({
                                "lever": df.lever, "lead": df.lead_s})
                            .groupby("lever")["lead"]
                            .agg(["count", "max"]).reset_index()
                            .rename(columns={"count": "n", "max": "lead_max_s"})
                            .set_index("lever").to_dict("index").items()},
            "lead_mayor_10": df.loc[df.lead_s > 10, ["pair_id", "lever", "lead_s",
                                                     "t_effective", "t_div_grid"]]
                                .to_dict("records")[:20],
            "c5_ok": len(c5_fail) == 0, "c5_failures": c5_fail[:20],
        },
        "D": {"lag_por_palanca": {k: {str(i): v for i, v in sorted(d.items())}
                                  for k, d in lag_tab.items()},
              "cumple_lag_0_1_pct_por_palanca": {
                  k: round(100.0 * ((d.get(0, 0) + d.get(1, 0)) / max(sum(d.values()), 1)), 2)
                  for k, d in lag_tab.items()},
              "palancas_que_fallan_99_pct": sorted(
                  k for k, d in lag_tab.items()
                  if max(sum(d.values()), 1) and
                  100.0 * ((d.get(0, 0) + d.get(1, 0)) / sum(d.values())) < 99.0),
              "lag_ge_2": int((df.effect_lag_boundary >= 2).sum()),
              "lag_ge_2_pares": df.loc[df.effect_lag_boundary >= 2,
                                       ["pair_id", "lever", "effect_lag_boundary",
                                        "effect_t1_boundary", "t_action",
                                        "t_effective", "t_boundary", "t_div_grid_col"]]
                                 .to_dict("records")[:60]},
        "E1": {"n": len(e1_rows), "pares": e1_rows},
        "decided_post_hoc": {
            "post_3b": {
                "decision": "t_action de los overrides de ventilación = split_t",
                "motivo": ("el generador aplica el override a t > split_t "
                           "(src/anessim/simulate.py:333) y el enunciado de 3c lo "
                           "confirma; 3b usó split_t sin justificarlo por escrito."),
                "efecto": "ninguno sobre las anotaciones ya publicadas de 3b",
            },
            "post_3c_A": {
                "decision": ("t_effective = t_action para acciones puntuales; "
                             "t_effective = t_divergence_raw para consignas "
                             "persistentes efectivas"),
                "motivo": ("la divergencia de una consigna persistente la fija el "
                           "plan base, no el acto; se conserva la definición de 3b "
                           "para no cambiar el criterio (R3)."),
            },
            "post_3c_C": {
                "decision": ("t_boundary = min(t_div_grid, rejilla(t_effective)) y "
                             "pre/post_action definidos por t_boundary con "
                             "convención semiabierta (g-5, g]"),
                "motivo": ("t_div_grid es el primer punto de rejilla donde difiere "
                           "alguna serie que alimenta las features del grupo; la "
                           "ventana que contiene la celda de t_boundary es la "
                           "primera posterior."),
            },
            "post_3c_alcance": {
                "decision": ("3b restringió C1/C2 a las palancas puntuales sin "
                             "autorización; 3c evalúa TODOS los pares efectivos"),
                "motivo": "R3: el criterio no se estrecha ni se amplía sin decirlo.",
            },
        },
        "elapsed_s": round(_time.time() - t_start, 1),
    }
    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_pairs", "n_effective_3b", "n_effective_v2",
                                          "elapsed_s")}, indent=2))
    print("A:", json.dumps(res["A"], ensure_ascii=False)[:600])
    print("B1:", res["B1"]["ok"], res["B1"]["n_failures"], res["B1"]["failures"][:5])
    print("B2 fallos:", res["B2"]["n_failures"])
    print("C:", json.dumps(res["C"]["clases"]), "lead max", res["C"]["lead_max_s"],
          "C5 ok", res["C"]["c5_ok"], res["C"]["c5_failures"][:5])
    print("D lag>=2:", res["D"]["lag_ge_2"], res["D"]["lag_ge_2_pares"][:6])
    print("escrito:", OUT_PARQUET, OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
