"""Tests del tokenizador v1 (src/tokens/tokenize.py).

Unitarios (a-h) sobre datos construidos; integración (i-p) sobre
data/tokens_v1/ (generado con `python -m tokens.tokenize run`).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tokens import tokenize as tk

ROOT = Path(__file__).resolve().parents[1]
TOKENS_DIR = ROOT / "data" / "tokens_v1" / "windows"
MANIFEST = ROOT / "data" / "tokens_v1" / "manifest_tokens.json"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_pk(n: int) -> dict:
    pk: dict[str, np.ndarray] = {}
    for d in tk.DRUGS:
        pk[f"ce_{d}"] = np.arange(n, dtype=float)
        pk[f"dose_cum_{d}"] = np.arange(n, dtype=float) * 10.0
        pk[f"bolus_{d}"] = np.ones(n, dtype=float)
        pk[f"bolus_obs_{d}"] = np.ones(n, dtype=float)
        pk[f"m_ce_{d}"] = np.zeros(n, dtype=np.uint8)
    return pk


def make_proxies(n: int) -> dict:
    return {v: (np.arange(n, dtype=float), np.ones(n, dtype=np.uint8))
            for v in tk.VENT_ITEMS}


def make_setpoints(values=None) -> dict:
    return {v: (values.copy() if values is not None else None) for v in tk.VENT_ITEMS}


def grid13():
    return np.arange(100, 165, 5, dtype=np.int64)


def phase_all(t):
    return np.full(len(t), "maintenance", dtype=object)


def tso(t):
    return (t - t[0]).astype(np.float64)


def build(t, phase=None, tso_arr=None, split="train", dense=False, pk=None,
          sp=None, pr=None):
    phase = phase if phase is not None else phase_all(t)
    tso_arr = tso_arr if tso_arr is not None else tso(t)
    pk = pk if pk is not None else make_pk(len(t))
    sp = sp if sp is not None else make_setpoints(np.full(len(t), 50.0))
    pr = pr if pr is not None else make_proxies(len(t))
    return tk.build_case_windows(t, phase, tso_arr, split, dense, pk, sp, pr)


def full_out(w, caseid=1, source="synthetic_v5", split="train"):
    ctx_ids = tk.load_vocab_v1()
    m = len(w["t0"])
    w2 = dict(w)
    for iid in ctx_ids:
        w2[f"ctx_{iid}"] = np.full(m, 1.0, dtype=np.float32)
    w2["n_ctx"] = np.full(m, len(ctx_ids), dtype=np.uint8)
    w2["caseid"] = np.full(m, caseid, dtype=np.int32)
    w2["source"] = np.full(m, source, dtype=object)
    w2["split"] = np.full(m, split, dtype=object)
    w2["pair_id"] = np.full(m, np.nan, dtype=np.float64)
    w2["cf_role"] = np.full(m, None, dtype=object)
    w2["split_t"] = np.full(m, np.nan, dtype=np.float64)
    w2["lever"] = np.full(m, None, dtype=object)
    w2["post_split"] = np.zeros(m, dtype=bool)
    return w2


# --------------------------------------------------------------------------
# Unitarios
# --------------------------------------------------------------------------

def test_a_window_12_cells():
    t = grid13()
    w, disc = build(t)
    assert list(w["t0"]) == [100]
    assert list(w["t1"]) == [160]
    assert w["drug_propofol_ce_t0"][0] == 0.0
    assert w["drug_propofol_ce_t1"][0] == 12.0
    assert w["drug_propofol_ce_max"][0] == 12.0
    assert w["drug_propofol_bolo"][0] == 12.0
    assert w["drug_propofol_dose_cum"][0] == 120.0
    assert disc["emitido"] == 1


def test_b_mask_worst_of_12():
    t = grid13()
    pk = make_pk(len(t))
    mce = np.zeros(len(t), dtype=np.uint8)
    mce[5] = 2
    pk["m_ce_propofol"] = mce
    w, _ = build(t, pk=pk)
    assert w["drug_propofol_mask"][0] == 2
    mce[7] = 1
    pk["m_ce_propofol"] = mce
    w, _ = build(t, pk=pk)
    assert w["drug_propofol_mask"][0] == 1
    pk["m_ce_propofol"] = np.zeros(len(t), dtype=np.uint8)
    w, _ = build(t, pk=pk)
    assert w["drug_propofol_mask"][0] == 0


def test_c_efedrina_mask1_absent_mask0_present():
    t = grid13()
    pk = make_pk(len(t))
    pk["ce_efedrina"] = np.arange(len(t), dtype=float) + 1.0
    pk["m_ce_efedrina"] = np.ones(len(t), dtype=np.uint8)
    w, _ = build(t, pk=pk)
    assert w["drug_efedrina_mask"][0] == 1
    out = tk.apply_normalization(w, {})
    assert out["drug_efedrina_ce_t0"][0] == 0.0

    pk["m_ce_efedrina"] = np.zeros(len(t), dtype=np.uint8)
    w, _ = build(t, pk=pk)
    assert w["drug_efedrina_mask"][0] == 0
    out = tk.apply_normalization(w, {})
    assert out["drug_efedrina_ce_t0"][0] != 0.0


def test_d_vent_setpoint_null_both_cells_uses_proxy():
    t = grid13()
    w, _ = build(t, sp=make_setpoints(None))
    assert w["vent_fio2_proxy"][0] == 1
    assert w["vent_fio2_t0"][0] == 0.0
    assert w["vent_fio2_t1"][0] == 12.0

    w, _ = build(t, sp=make_setpoints(np.full(len(t), 50.0)))
    assert w["vent_fio2_proxy"][0] == 0
    assert w["vent_fio2_t0"][0] == 50.0
    assert w["vent_fio2_t1"][0] == 50.0


def test_e_log1p_zscore_and_masked_zero():
    t = grid13()
    w, _ = build(t)
    stats = {tk._stat_key("drug", "propofol", "ce_t0"): {"mean": 5.0, "std": 1.0}}
    out = tk.apply_normalization(w, stats)
    # ce_t0 raw = 0 (mask 0) -> log1p(0)=0 -> (0-5)/1 = -5 != 0
    assert out["drug_propofol_ce_t0"][0] != 0.0
    assert abs(out["drug_propofol_ce_t0"][0] + 5.0) < 1e-6

    pk = make_pk(len(t))
    pk["m_ce_propofol"] = np.full(len(t), 2, dtype=np.uint8)
    w2, _ = build(t, pk=pk)
    out2 = tk.apply_normalization(w2, stats)
    assert out2["drug_propofol_ce_t0"][0] == 0.0


def test_f_grid_hole_not_emitted():
    t = np.array([100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 170],
                 dtype=np.int64)
    w, disc = build(t)
    assert disc["hueco"] == len(t)
    assert disc["emitido"] == 0
    assert len(w["t0"]) == 0


def test_g_phase_filter_t1_ge_opend_discarded():
    t = np.arange(100, 225, 5, dtype=np.int64)
    phase = np.array(["maintenance" if x < 200 else "emergence" for x in t],
                     dtype=object)
    w, disc = build(t, phase=phase)
    assert disc["emergencia"] == 5
    assert list(w["t0"]) == [100]


def _write_setpoint_parquet(path, time, **setpoints):
    cols = {"time": np.asarray(time, dtype=np.float64)}
    for name, arr in setpoints.items():
        if arr is not None:
            cols[name] = np.asarray(arr, dtype=np.float64)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(cols)), path)


def test_setpoints_a_1s_axis_to_5s_grid(tmp_path):
    # eje crudo a 1 s; rejilla a 5 s: el valor en t es el último registrado con time <= t
    t = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=float)
    fio2 = np.array([30, 30, 40, 40, 40, 50, 50, 50, 50, 50, 60], dtype=float)
    p = tmp_path / "case.parquet"
    _write_setpoint_parquet(p, t, **{"Primus/SET_FIO2": fio2})
    grid = np.array([10, 15, 20], dtype=np.int64)
    sp = tk.read_setpoints_from_path(p, grid)
    np.testing.assert_allclose(sp["fio2"], [30, 50, 60])


def test_setpoints_b_5s_axis_same_result(tmp_path):
    t = np.array([10, 15, 20], dtype=float)
    fio2 = np.array([30, 50, 60], dtype=float)
    p = tmp_path / "case.parquet"
    _write_setpoint_parquet(p, t, **{"Primus/SET_FIO2": fio2})
    grid = np.array([10, 15, 20], dtype=np.int64)
    sp = tk.read_setpoints_from_path(p, grid)
    np.testing.assert_allclose(sp["fio2"], [30, 50, 60])


def test_setpoints_c_change_between_cells(tmp_path):
    # el cambio en t=13 se refleja en la celda t=15 (no en la t=10)
    t = np.array([10, 11, 12, 13, 14, 15, 16], dtype=float)
    fio2 = np.array([30, 30, 30, 40, 40, 40, 40], dtype=float)
    p = tmp_path / "case.parquet"
    _write_setpoint_parquet(p, t, **{"Primus/SET_FIO2": fio2})
    grid = np.array([10, 15, 20], dtype=np.int64)
    sp = tk.read_setpoints_from_path(p, grid)
    np.testing.assert_allclose(sp["fio2"], [30, 40, 40])


def test_h_determinism_bytes_identical():
    t = np.arange(100, 225, 5, dtype=np.int64)
    stats = {tk._stat_key("drug", "propofol", "ce_t0"): {"mean": 2.0, "std": 1.5},
             tk._stat_key("vent", "fio2", "t0"): {"mean": 10.0, "std": 3.0},
             tk._stat_key("time", "t_since_opstart", "value"): {"mean": 5.0, "std": 2.0}}
    w1, _ = build(t)
    w2, _ = build(t)
    out1 = tk.apply_normalization(full_out(w1), stats)
    out2 = tk.apply_normalization(full_out(w2), stats)
    b1 = io.BytesIO()
    b2 = io.BytesIO()
    pq.write_table(tk.build_table(out1, tk.load_vocab_v1()), b1, compression="zstd")
    pq.write_table(tk.build_table(out2, tk.load_vocab_v1()), b2, compression="zstd")
    assert b1.getvalue() == b2.getvalue()


# --------------------------------------------------------------------------
# Integración
# --------------------------------------------------------------------------

def _require_output():
    if not TOKENS_DIR.exists() or not MANIFEST.exists():
        raise AssertionError("data/tokens_v1 no existe: ejecuta 'python -m tokens.tokenize run'")
    parts = sorted(TOKENS_DIR.glob("source=*/split=*/part-*.parquet"))
    if not parts:
        raise AssertionError("data/tokens_v1 vacío")
    return parts


TOKEN_COLS = [c for c in tk.output_columns(tk.load_vocab_v1())
              if c not in {"caseid", "t0", "t1", "source", "split", "dense",
                           "pair_id", "cf_role", "split_t", "lever", "post_split"}]

LEVER_DRUG = {
    "propofol_bolus": "propofol", "noradrenaline": "noradrenalina",
    "remi_up": "remifentanilo", "ephedrine": "efedrina",
    "sevo_up": "sevoflurano", "ppf20_rate": "propofol",
    "rftn20_rate": "remifentanilo", "ppf_bolus": "propofol",
    "remi_bolus": "remifentanilo", "nepi_rate": "noradrenalina",
    "phen_rate": "fenilefrina", "phen_bolus": "fenilefrina",
    "eph_bolus": "efedrina", "sevo_mac": "sevoflurano",
}


def _aeq(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _cf_case_map(parts):
    """Mapea caseid -> partición de salida escaneando las particiones cf_v5.

    El mapeo caseid -> partición no se puede derivar con la fórmula de suma
    acumulada de celdas: window.py hace flush cuando el buffer supera 200 000
    celdas, así que cada partición tiene >= 200 000 celdas y los límites se
    desplazan. Escanear las particiones de salida es la forma correcta.
    """
    caseids: set[int] = set()
    case_to_part: dict[int, Path] = {}
    for part in parts:
        if "source=cf_v5" not in str(part):
            continue
        df = pq.ParquetFile(part).read(columns=["caseid"]).to_pandas()
        for cid in df.caseid.unique():
            cid = int(cid)
            caseids.add(cid)
            case_to_part[cid] = part
    return caseids, case_to_part


def _load_case_tokens_from_map(caseid: int, case_to_part: dict) -> pd.DataFrame:
    tbl = pq.read_table(case_to_part[caseid]).to_pandas()
    return tbl[tbl.caseid == caseid].sort_values("t0", kind="stable").reset_index(drop=True)


def test_i_gate5_split_matches():
    parts = _require_output()
    cases = pq.read_table(ROOT / "data" / "windows_v2" / "cases.parquet").to_pandas()
    split_map = pq.read_table(ROOT / "data" / "windows_v2" / "split.parquet").to_pandas()
    subj_split = dict(zip(split_map.subjectid.astype(str), split_map.split.astype(str)))
    case_expected = {int(r.caseid): subj_split[str(r.subjectid)] for _, r in cases.iterrows()}

    seen: dict[str, set] = {}
    for part in parts:
        df = pq.ParquetFile(part).read(columns=["caseid", "split"]).to_pandas()
        for split, grp in df.groupby("split"):
            split = str(split)
            cids = set(int(c) for c in grp.caseid)
            seen.setdefault(split, set()).update(cids)
    assert seen.get("train", set()).isdisjoint(seen.get("val", set()))
    for split, cids in seen.items():
        for cid in cids:
            assert case_expected.get(cid) == split, f"split incorrecto para {cid}"


def test_j_gate2_cf_prefix():
    parts = _require_output()
    cf_meta = tk.load_cf_meta()
    available, case_to_part = _cf_case_map(parts)
    pairs_info: dict[int, list] = {}
    for cid, m in cf_meta.items():
        if m["cf_role"] == "intervencion":
            pairs_info.setdefault(m["pair_id"], [m["pair_id"], None])
    for cid, m in cf_meta.items():
        if m["cf_role"] == "base":
            pairs_info[m["pair_id"]][1] = cid
    pairs = sorted(p for p, (a, b) in pairs_info.items()
                   if a in available and b in available)
    drug_pairs = []
    seen_levers = set()
    for p in pairs:
        lever = cf_meta[p]["lever"]
        if lever not in LEVER_DRUG:
            continue
        if lever not in seen_levers:
            seen_levers.add(lever)
            drug_pairs.append(p)
    rest = [p for p in pairs if cf_meta[p]["lever"] in LEVER_DRUG and p not in set(drug_pairs)]
    selected = (drug_pairs + rest)[:20]
    assert len(selected) == 20

    for p in selected:
        meta = cf_meta[p]
        a = int(pairs_info[p][0])
        b = int(pairs_info[p][1])
        da = _load_case_tokens_from_map(a, case_to_part)
        db = _load_case_tokens_from_map(b, case_to_part)
        da = da[da.dense == False].sort_values("t0").reset_index(drop=True)
        db = db[db.dense == False].sort_values("t0").reset_index(drop=True)
        m = da.merge(db[["t0"] + TOKEN_COLS], on="t0", suffixes=("_a", "_b"))
        # contrato v4: post_split = t1 > split_t
        pre = m[~m.post_split]
        post = m[m.post_split]

        # prefijo (post_split=False): idéntico bit a bit
        for c in TOKEN_COLS:
            assert _aeq(pre[f"{c}_a"], pre[f"{c}_b"]), f"prefijo difiere en {c} (par {p})"

        lever_cols = [c for c in TOKEN_COLS if c.startswith(f"drug_{LEVER_DRUG[meta['lever']]}_")]
        first_diff = None
        for _, row in post.iterrows():
            diff = [c for c in TOKEN_COLS if not _aeq(m.loc[m.t0 == row.t0, f"{c}_a"].iloc[0],
                                                      m.loc[m.t0 == row.t0, f"{c}_b"].iloc[0])]
            if diff:
                first_diff = (int(row.t0), diff)
                break
        assert first_diff is not None, f"par {p}: sin divergencia post-split"
        assert set(first_diff[1]) <= set(lever_cols), \
            f"par {p}: divergencia fuera de la palanca: {set(first_diff[1]) - set(lever_cols)}"
        # correcci¾n 1: post_split = t1 > split_t. La divergencia no puede
        # aparecer en el prefijo; puede empezar en la primera ventana post_split
        # (la que CONTIENE split_t) o en una posterior (intervenci¾n al final de
        # la ventana, p. ej. nora a t=11932 en la ventana 11880-11940).
        first_post = int(post.t0.iloc[0]) if len(post) else None
        assert first_post is not None, f"par {p}: sin ventanas post_split"
        assert first_diff[0] >= first_post, \
            f"par {p}: divergencia ({first_diff[0]}) antes de la primera post_split ({first_post})"


def test_k_gate9_stats_train_only():
    parts = _require_output()
    mf = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stats = mf["normalization_stats"]
    expected_keys = []
    for d in tk.DRUGS:
        for f in ("ce_t0", "ce_t1", "ce_max", "bolo", "dose_cum"):
            expected_keys.append(tk._stat_key("drug", d, f))
    for v in tk.VENT_ITEMS:
        expected_keys.append(tk._stat_key("vent", v, "t0"))
        expected_keys.append(tk._stat_key("vent", v, "t1"))
    expected_keys.append(tk._stat_key("time", "t_since_opstart", "value"))
    assert set(expected_keys) <= set(stats.keys())

    sums: dict[str, float] = {}
    sq: dict[str, float] = {}
    n: dict[str, int] = {}
    train_parts = sorted(TOKENS_DIR.glob("source=*/split=train/part-*.parquet"))
    for part in train_parts:
        cols = []
        for d in tk.DRUGS:
            cols += [f"drug_{d}_{f}" for f in ("ce_t0", "ce_t1", "ce_max", "bolo", "dose_cum")]
            cols.append(f"drug_{d}_mask")
        for v in tk.VENT_ITEMS:
            cols += [f"vent_{v}_t0", f"vent_{v}_t1", f"vent_{v}_mask"]
        cols.append("time_t_since_opstart")
        df = pq.read_table(part, columns=cols).to_pandas()
        for d in tk.DRUGS:
            mask = df[f"drug_{d}_mask"].to_numpy() == 0
            for f in ("ce_t0", "ce_t1", "ce_max", "bolo", "dose_cum"):
                key = tk._stat_key("drug", d, f)
                v = df[f"drug_{d}_{f}"].to_numpy()[mask].astype(np.float64)
                sums[key] = sums.get(key, 0.0) + float(v.sum())
                sq[key] = sq.get(key, 0.0) + float((v * v).sum())
                n[key] = n.get(key, 0) + int(len(v))
        for v in tk.VENT_ITEMS:
            mask = df[f"vent_{v}_mask"].to_numpy() == 0
            for f in ("t0", "t1"):
                key = tk._stat_key("vent", v, f)
                arr = df[f"vent_{v}_{f}"].to_numpy()[mask].astype(np.float64)
                sums[key] = sums.get(key, 0.0) + float(arr.sum())
                sq[key] = sq.get(key, 0.0) + float((arr * arr).sum())
                n[key] = n.get(key, 0) + int(len(arr))
        key = tk._stat_key("time", "t_since_opstart", "value")
        arr = df["time_t_since_opstart"].to_numpy().astype(np.float64)
        sums[key] = sums.get(key, 0.0) + float(arr.sum())
        sq[key] = sq.get(key, 0.0) + float((arr * arr).sum())
        n[key] = n.get(key, 0) + int(len(arr))

    for key in expected_keys:
        mean = sums[key] / n[key]
        std = float(np.sqrt(max(0.0, sq[key] / n[key] - mean * mean)))
        assert abs(mean) < 1e-3, f"{key}: media train {mean:.5f} != 0"
        # features constantes (p. ej. bolo/dose_cum de sevoflurano) tienen std 0
        assert abs(std - 1.0) < 0.02 or std < 1e-6, \
            f"{key}: std train {std:.5f} ni 1 ni 0"


def test_l_gate8_context_v1_only():
    parts = _require_output()
    ids = tk.load_vocab_v1()
    assert len(ids) == 12
    schema = pq.read_schema(parts[0])
    ctx_cols = [n for n in schema.names if n.startswith("ctx_")]
    assert sorted(ctx_cols) == sorted("ctx_" + i for i in ids)


def test_m_gate4_no_excluded_columns():
    from tokens.context_vocab import EXCLUDED_COLUMNS, CONTEXT_COLUMNS
    out_cols = set(TOKEN_COLS)
    assert out_cols.isdisjoint(EXCLUDED_COLUMNS)
    ctx_ids = set(tk.load_vocab_v1())
    base_cols = {i.split(":")[0] for i in ctx_ids}
    assert base_cols <= set(CONTEXT_COLUMNS)
    # columnas que tokenize lee como feature: ventanas, pk y setpoints
    read_cols = (set(tk.WINDOWS_READ_COLS) | set(tk.PK_READ_COLS)
                 | set(tk.VENT_SETPOINT_COLS.values()) | {"time"})
    inter = read_cols & set(EXCLUDED_COLUMNS)
    assert inter <= {"caseid"}, f"columnas excluidas leídas: {inter}"


def test_n_gate7_ks_informative():
    from scipy.stats import ks_2samp
    parts = _require_output()
    cols = ["source"]
    for d in tk.DRUGS:
        cols += [f"drug_{d}_ce_t0", f"drug_{d}_mask"]
    for v in tk.VENT_ITEMS:
        cols += [f"vent_{v}_t0", f"vent_{v}_mask"]
    train_parts = sorted(TOKENS_DIR.glob("source=*/split=train/part-*.parquet"))
    acc = {f"drug.{d}": {"real": [], "synth": []} for d in tk.DRUGS}
    acc.update({f"vent.{v}": {"real": [], "synth": []} for v in tk.VENT_ITEMS})
    for part in train_parts:
        df = pq.read_table(part, columns=cols).to_pandas()
        real = df[df.source == "real"]
        synth = df[df.source == "synthetic_v5"]
        for d in tk.DRUGS:
            r = real[real[f"drug_{d}_mask"] == 0][f"drug_{d}_ce_t0"].to_numpy()
            s = synth[synth[f"drug_{d}_mask"] == 0][f"drug_{d}_ce_t0"].to_numpy()
            acc[f"drug.{d}"]["real"].append(r)
            acc[f"drug.{d}"]["synth"].append(s)
        for v in tk.VENT_ITEMS:
            r = real[real[f"vent_{v}_mask"] == 0][f"vent_{v}_t0"].to_numpy()
            s = synth[synth[f"vent_{v}_mask"] == 0][f"vent_{v}_t0"].to_numpy()
            acc[f"vent.{v}"]["real"].append(r)
            acc[f"vent.{v}"]["synth"].append(s)
    rows = []
    for key, d in acc.items():
        r = np.concatenate(d["real"]) if d["real"] else np.zeros(0)
        s = np.concatenate(d["synth"]) if d["synth"] else np.zeros(0)
        if len(r) == 0 or len(s) == 0:
            rows.append((key, None, len(r), len(s)))
        else:
            stat, _ = ks_2samp(r, s)
            rows.append((key, float(stat), len(r), len(s)))
    assert len(rows) == len(tk.DRUGS) + len(tk.VENT_ITEMS)
    # informativo: se imprime la tabla ordenada por estadístico
    for key, stat, nr, ns in sorted(rows, key=lambda x: -(x[1] if x[1] is not None else 0)):
        print(f"{key:14s} KS={stat if stat is None else round(stat,6)} "
              f"n_real={nr} n_synth={ns}")


def test_o_gate1_coverage():
    parts = _require_output()
    drug_counts: dict = {}
    vent_counts: dict = {}
    ctx_counts: dict = {}
    total: dict = {}
    for part in parts:
        rel = part.parent.relative_to(TOKENS_DIR)
        key = str(rel).replace("\\", "/")
        cols = ["source"]
        for d in tk.DRUGS:
            cols.append(f"drug_{d}_mask")
        for v in tk.VENT_ITEMS:
            cols.append(f"vent_{v}_mask")
        ctx_ids = tk.load_vocab_v1()
        cols += [f"ctx_{i}" for i in ctx_ids]
        df = pq.read_table(part, columns=cols).to_pandas()
        for d in tk.DRUGS:
            for st, grp in df.groupby("source"):
                vc = grp[f"drug_{d}_mask"].value_counts()
                for mstate in (0, 1, 2):
                    drug_counts[(d, str(st), key, mstate)] = \
                        drug_counts.get((d, str(st), key, mstate), 0) + int(vc.get(mstate, 0))
        for v in tk.VENT_ITEMS:
            for st, grp in df.groupby("source"):
                vc = grp[f"vent_{v}_mask"].value_counts()
                for mstate in (0, 1, 2):
                    vent_counts[(v, str(st), key, mstate)] = \
                        vent_counts.get((v, str(st), key, mstate), 0) + int(vc.get(mstate, 0))
        for i in ctx_ids:
            for st, grp in df.groupby("source"):
                emitted = int(grp[f"ctx_{i}"].notna().sum())
                ctx_counts[(i, str(st), key, "emitido")] = \
                    ctx_counts.get((i, str(st), key, "emitido"), 0) + emitted
        for st, grp in df.groupby("source"):
            total[(str(st), key)] = total.get((str(st), key), 0) + len(grp)

    # aserciones estructurales: todos los tipos e items presentes
    for d in tk.DRUGS:
        assert any(k[0] == d for k in drug_counts)
    for v in tk.VENT_ITEMS:
        assert any(k[0] == v for k in vent_counts)
    for i in tk.load_vocab_v1():
        assert any(k[0] == i for k in ctx_counts)

    # tabla completa (informativa)
    print("\nCobertura (fracción de ventanas por estado de máscara):")
    for d in tk.DRUGS:
        for (src, key) in sorted(total):
            ntot = total[(src, key)]
            row = [drug_counts.get((d, src, key, 0), 0) / ntot,
                   drug_counts.get((d, src, key, 1), 0) / ntot,
                   drug_counts.get((d, src, key, 2), 0) / ntot]
            print(f"  drug {d:14s} {src:14s} {key:24s} 0={row[0]:.3f} 1={row[1]:.3f} 2={row[2]:.3f}")
    for v in tk.VENT_ITEMS:
        for (src, key) in sorted(total):
            ntot = total[(src, key)]
            row = [vent_counts.get((v, src, key, 0), 0) / ntot,
                   vent_counts.get((v, src, key, 1), 0) / ntot,
                   vent_counts.get((v, src, key, 2), 0) / ntot]
            print(f"  vent {v:6s} {src:14s} {key:24s} 0={row[0]:.3f} 1={row[1]:.3f} 2={row[2]:.3f}")
    for i in tk.load_vocab_v1():
        for (src, key) in sorted(total):
            ntot = total[(src, key)]
            frac = ctx_counts.get((i, src, key, "emitido"), 0) / ntot
            print(f"  ctx  {i:12s} {src:14s} {key:24s} emitido={frac:.3f}")


def test_p_cf_metadata():
    parts = _require_output()
    cf_meta = tk.load_cf_meta()
    roles_by_pair: dict[int, set] = {}
    cols = ["caseid", "t0", "t1", "pair_id", "cf_role", "split_t", "lever", "post_split"]
    for part in parts:
        if "source=cf_v5" not in str(part):
            continue
        df = pq.read_table(part, columns=cols).to_pandas()
        for _, r in df.iterrows():
            pair = int(r.pair_id)
            roles_by_pair.setdefault(pair, set()).add(str(r.cf_role))
            assert str(r.cf_role) in ("base", "intervencion")
            meta = cf_meta[int(r.caseid)]
            assert abs(float(r.split_t) - float(meta["split_t"])) < 1e-6
            assert r.lever == meta["lever"]
            # contrato v4: post_split = t1 > split_t
            assert bool(r.post_split) == (int(r.t1) > float(meta["split_t"]))
    assert roles_by_pair, "no se encontraron filas cf_v5"
    for pair, roles in roles_by_pair.items():
        assert roles == {"base", "intervencion"}, f"par {pair}: roles {roles}"


def test_q_no_phase_marks_real_cases_excluded():
    parts = _require_output()
    excluded = set(tk.load_cases_without_phase_marks())
    # contrato v5: 100 % maintenance en windows_v2 -> 35 caseids (los 24 de la
    # regresión de la iteración 3 vuelven a quedar fuera).
    assert len(excluded) == 35
    found: set[int] = set()
    for part in parts:
        if "source=real" not in str(part):
            continue
        df = pq.ParquetFile(part).read(columns=["caseid"]).to_pandas()
        found.update(int(c) for c in df.caseid.unique())
    assert excluded.isdisjoint(found), \
        f"casos sin marcas de fase en la salida: {sorted(excluded & found)}"


def test_cf_branch_pairing():
    """Corrección 2: las ramas base e intervención tienen el mismo conjunto de
    t0 dense=False (sin eso no se puede emparejar ventana a ventana)."""
    parts = _require_output()
    cf_meta = tk.load_cf_meta()
    available, case_to_part = _cf_case_map(parts)
    pairs_info: dict[int, list] = {}
    for cid, m in cf_meta.items():
        if m["cf_role"] == "intervencion":
            pairs_info.setdefault(m["pair_id"], [m["pair_id"], None])
    for cid, m in cf_meta.items():
        if m["cf_role"] == "base":
            pairs_info[m["pair_id"]][1] = cid
    pairs = sorted(p for p, (a, b) in pairs_info.items()
                   if a in available and b in available)
    drug_pairs, seen = [], set()
    for p in pairs:
        lever = cf_meta[p]["lever"]
        if lever not in LEVER_DRUG:
            continue
        if lever not in seen:
            seen.add(lever)
            drug_pairs.append(p)
    rest = [p for p in pairs if cf_meta[p]["lever"] in LEVER_DRUG and p not in set(drug_pairs)]
    selected = (drug_pairs + rest)[:20]
    assert len(selected) == 20
    for p in selected:
        a = int(pairs_info[p][0])
        b = int(pairs_info[p][1])
        da = _load_case_tokens_from_map(a, case_to_part)
        db = _load_case_tokens_from_map(b, case_to_part)
        ta = sorted(int(x) for x in da[da.dense == False].t0)
        tb = sorted(int(x) for x in db[db.dense == False].t0)
        assert ta == tb, f"par {p}: t0 dense=False distintos ({len(ta)} vs {len(tb)})"


def test_pairs_parquet():
    """Corrección 2: data/tokens_v1/pairs.parquet, una fila por par CF."""
    path = ROOT / "data" / "tokens_v1" / "pairs.parquet"
    assert path.exists(), "pairs.parquet no existe"
    df = pq.read_table(path).to_pandas()
    assert list(df.columns) == [
        "pair_id", "caseid_base", "caseid_intervencion", "split_t", "lever",
        "lever_group", "n_windows_pre", "n_windows_post", "n_windows_total",
        "n_windows_base", "n_windows_intervencion", "usable"]
    assert df.pair_id.is_unique
    assert (df.n_windows_total == df.n_windows_pre + df.n_windows_post).all()
    # Pares sin ventanas emitidas (excluidos a nivel de windows_v2) quedan con
    # n_windows_total=0; el resto debe ser >0.
    available, _ = _cf_case_map(_require_output())
    zero = df[df.n_windows_total == 0]
    assert set(int(p) for p in zero.pair_id).isdisjoint(available), \
        "par con n_windows_total=0 pero presente en la salida"
    assert (df.loc[df.n_windows_total != 0, "n_windows_total"] > 0).all()
    # corrección 4: todo par con ventanas debe tener ambas ramas con el mismo
    # conjunto de t0 dense=False (usable=True).
    bad = df[(df.n_windows_total > 0) & (~df.usable)]
    assert bad.empty, \
        f"pares con ventanas pero no usables: {bad[['pair_id', 'lever']].to_dict('records')}"
    cf_meta = tk.load_cf_meta()
    for _, r in df.iterrows():
        assert int(r.caseid_intervencion) == int(r.pair_id)
        assert cf_meta[int(r.caseid_base)]["cf_role"] == "base"
        assert cf_meta[int(r.caseid_intervencion)]["cf_role"] == "intervencion"
        # corrección 5: lever_group según el objetivo de la palanca.
        assert r.lever_group == tk.LEVER_GROUP[r.lever], \
            f"lever_group mal para {r.lever}: {r.lever_group}"
    assert set(df.lever_group.unique()) == set(tk.LEVER_GROUP.values())


def test_gate4_mechanical_read_columns(tmp_path, monkeypatch):
    """Corrección 3: ninguna lectura de parquet omite columns= y la unión de
    columnas solicitadas no intersecta con la tabla Excluidas (salvo caseid,
    opstart y opend, que son claves/marcas de rejilla)."""
    from tokens.context_vocab import EXCLUDED_COLUMNS
    excluded = set(EXCLUDED_COLUMNS)
    allowed = {"caseid", "opstart", "opend"}
    calls: list[tuple] = []
    real_read_table = pq.read_table
    real_pf_read = pq.ParquetFile.read

    def spy_read_table(source, columns=None, **kw):
        if columns is None:
            calls.append(("NO_COLUMNS", str(source)))
        else:
            calls.append(("cols", set(columns)))
        return real_read_table(source, columns=columns, **kw)

    def spy_pf_read(self, columns=None, **kw):
        if columns is None:
            calls.append(("NO_COLUMNS", "ParquetFile"))
        else:
            calls.append(("cols", set(columns)))
        return real_pf_read(self, columns=columns, **kw)

    monkeypatch.setattr(pq, "read_table", spy_read_table)
    monkeypatch.setattr(pq.ParquetFile, "read", spy_pf_read)
    monkeypatch.setattr(tk, "OUT_DIR", tmp_path / "windows")
    monkeypatch.setattr(tk, "OUT_ROOT", tmp_path)

    for fn in (tk.load_cases, tk.load_context_map, tk.load_cases_without_phase_marks):
        fn.cache_clear()
    tk.run_full(verbose=False, limit_parts_per_group=1)
    tk.verify_output()

    no_cols = [c for c in calls if c[0] == "NO_COLUMNS"]
    assert not no_cols, f"llamadas sin columns=: {no_cols}"
    all_cols: set = set()
    for kind, cols in calls:
        if kind == "cols":
            all_cols.update(cols)
    inter = all_cols & excluded
    assert inter <= allowed, f"columnas excluidas leídas: {inter - allowed}"


def test_column_lists_disjoint_and_complete():
    """Corrección 4: feature_columns / mask_columns / metadata_columns disjuntas
    y con unión exactamente igual al esquema; es_proxy viaja como metadato."""
    f = set(tk.feature_columns())
    m = set(tk.mask_columns())
    md = set(tk.metadata_columns())
    assert not (f & m) and not (f & md) and not (m & md)
    schema = set(tk.output_columns(tk.load_vocab_v1()))
    assert f | m | md == schema
    for v in tk.VENT_ITEMS:
        assert f"vent_{v}_proxy" in md
        assert f"vent_{v}_proxy" not in f
    mf = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(mf["feature_columns"]) == f
    assert set(mf["mask_columns"]) == m
    assert set(mf["metadata_columns"]) == md


def test_partition_invariant():
    """Corrección 5: ningún caseid aparece en más de una partición windows_v2."""
    parts = sorted((ROOT / "data" / "windows_v2" / "windows").glob(
        "source=*/split=*/part-*.parquet"))
    seen: set[int] = set()
    dup: set[int] = set()
    min_part = None
    for part in parts:
        df = pq.ParquetFile(part).read(columns=["caseid"]).to_pandas()
        uniq = {int(c) for c in df.caseid.unique()}
        inter = seen & uniq
        if inter:
            dup.update(inter)
        seen.update(uniq)
        min_part = len(df) if min_part is None else min(min_part, len(df))
    assert not dup, f"caseids repartidos entre particiones: {sorted(dup)}"
    cases = pq.read_table(ROOT / "data" / "windows_v2" / "cases.parquet").to_pandas()
    max_cells = int(cases.n_windows.max())
    print(f"max celdas/caso={max_cells}, min celdas/particion={min_part}, "
          f"n_caseids_total={len(seen)}")


def test_r_phase_mark_criterion():
    """Corrección 1: se excluye el caso 100 % maintenance; no se excluye el que
    tiene al menos una celda induction/emergence."""
    agg = {101: (10, 10), 102: (12, 11), 103: (8, 0)}
    excl = set(tk._phase_mark_caseids(agg))
    assert 101 in excl, "caso 100 % maintenance debe excluirse"
    assert 102 not in excl, "caso con 1 celda no maintenance no debe excluirse"
    assert 103 not in excl, "caso sin celdas maintenance no debe excluirse"


def test_s_phase_marks_match_windows_v2():
    """Corrección 1: load_cases_without_phase_marks() coincide con el conjunto
    de casos reales 100 % maintenance de windows_v2 (diagnóstico 1)."""
    agg: dict[int, tuple[int, int]] = {}
    for part in sorted((ROOT / "data" / "windows_v2" / "windows").glob(
            "source=real/split=*/part-*.parquet")):
        df = pq.read_table(part, columns=["caseid", "phase_from_clinical"]).to_pandas()
        for cid, grp in df.groupby("caseid"):
            cid = int(cid)
            n = int(len(grp))
            m = int((grp["phase_from_clinical"] == "maintenance").sum())
            prev = agg.get(cid, (0, 0))
            agg[cid] = (prev[0] + n, prev[1] + m)
    expected = {cid for cid, (n, m) in agg.items() if n > 0 and m == n}
    tk.load_cases_without_phase_marks.cache_clear()
    actual = set(tk.load_cases_without_phase_marks())
    assert actual == expected
    assert len(actual) == 35  # diagnóstico 1: la cohorte 100 % maint es exactamente la de 35


def test_t_lever_group_covers_all():
    """Corrección 5: LEVER_GROUP cubre exactamente las 21 palancas de cf_meta."""
    cf_meta = tk.load_cf_meta()
    levers = {m["lever"] for m in cf_meta.values()}
    assert set(tk.LEVER_GROUP) == levers
    assert len(tk.LEVER_GROUP) == 21
    assert set(tk.LEVER_GROUP.values()) == {
        "propofol", "remifentanilo", "efedrina", "fenilefrina",
        "noradrenalina", "sevoflurano", "ventilacion"}


def test_u_report_no_obsolete_strings(tmp_path, monkeypatch):
    """Corrección 3: write_report() no deja cadenas obsoletas en el informe."""
    monkeypatch.setattr(tk, "REPORT_PATH", tmp_path / "REPORT.txt")
    tk.write_report()
    text = (tmp_path / "REPORT.txt").read_text(encoding="utf-8")
    assert "t0 >= split_t" not in text
    assert "35 casos" not in text


def test_v_sin_celdas_case_4476():
    """Corrección 2: el caseid real 4476 (n_windows=0 en cases.parquet) no está
    en ninguna partición y se registra como causa de descarte sin_celdas."""
    cases = pq.read_table(ROOT / "data" / "windows_v2" / "cases.parquet",
                          columns=["caseid", "n_windows"]).to_pandas()
    zero = {int(c) for c in cases[cases.n_windows == 0].caseid}
    assert zero == {4476}
    mf = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(mf["sin_celdas_caseids"]) == zero
    assert mf["discards_by_cause"]["sin_celdas"] == 1
    found: set[int] = set()
    for part in _require_output():
        df = pq.ParquetFile(part).read(columns=["caseid"]).to_pandas()
        found.update(int(c) for c in df.caseid.unique())
    assert 4476 not in found
