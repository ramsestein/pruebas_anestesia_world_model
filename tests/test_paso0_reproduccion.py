"""Tests de reproducción de los números conocidos (FASE D del PASO 0).

Cada test fija en la suite un criterio de aceptación de la tarea D1-D5, de modo
que cualquier cambio futuro que rompa la reproducción falle en rojo. Todos leen
datos a través de src/paths.py.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import paths

INTEGRATION = pytest.mark.integration


# ---------------------------------------------------------------------------
# D1 — VENTANAS: conteos de windows_v4 (lectura vía paths)
# ---------------------------------------------------------------------------

@INTEGRATION
def test_d1_windows_counts_v4():
    wm = json.loads((paths.WINDOWS_DIR / "manifest.json").read_text(encoding="utf-8"))
    nw = wm["n_windows_by_source_split"]
    nc = wm["n_cases_by_source_split"]

    def cells(src, split):
        return int(nw[f"source={src}/split={split}"])

    def cases(src, split):
        return int(nc[f"{src}|{split}"])

    # real
    assert cells("real", "train") == 12_061_757
    assert cells("real", "val") == 2_063_800
    assert cases("real", "train") == 5_437
    assert cases("real", "val") == 951
    # sintéticas v7
    assert cells("synthetic_v7", "train") == 21_728_215
    assert cells("synthetic_v7", "val") == 3_896_864
    assert cells("vaso_reinf_v7", "train") == 1_151_944
    assert cells("vaso_reinf_v7", "val") == 184_662
    assert cells("cf_v7", "train") == 27_863_930
    assert cells("cf_v7", "val") == 4_845_942


# ---------------------------------------------------------------------------
# D2 — AUTOENCODER: gates 2 y 3 del ae_bal congelado sobre la real de val
# ---------------------------------------------------------------------------

@INTEGRATION
def test_d2_ae_gates_23_reproduce_manifest():
    import torch

    from ae import physio_ae as pa

    m = json.loads((paths.AE_DIR / "ae_bal" / "manifest_ae.json").read_text(encoding="utf-8"))
    ref2 = m["gates"]["gate2"]["real"]
    ref3 = m["gates"]["gate3"]

    device = torch.device("cpu")
    model = pa.load_model("ae_bal", "cpu").to(device)
    norm_stats = pa.load_norm_stats("ae_bal")
    excluded = pa.excluded_caseids()

    val = pa.load_cells(["real"], "val", excluded)
    gates = pa.evaluate(model, norm_stats, val, device)
    g2 = gates["gate2"]["real"]
    g3 = gates["gate3"]

    # valores físicos con tolerancia +-0.0005 (orden de acumulación)
    for t in ["Solar8000/HR", "Solar8000/ART_MBP", "BIS/BIS", "Primus/ETCO2"]:
        assert abs(g2[t] - ref2[t]) <= 0.0005, f"gate2 {t}: {g2[t]:.5f} vs {ref2[t]:.5f}"
    # recuentos exactos
    assert g3["n_cases"] == ref3["n_cases"] == 943
    assert g3["n_cells"] == ref3["n_cells"] == 1_481_996
    assert abs(g3["err_hr_with_art"] - ref3["err_hr_with_art"]) <= 0.0005
    assert abs(g3["err_hr_without_art"] - ref3["err_hr_without_art"]) <= 0.0005


# ---------------------------------------------------------------------------
# D3 — PK: gate 3 de pk_tokens (equivalencia integrador vs anessim) en v7
# ---------------------------------------------------------------------------

@INTEGRATION
def test_d3_pk_gate3_integrator_equivalence_v7():
    from anessim.pk.base import Infusion
    from anessim.pk.propofol import PropofolSchnider
    from anessim.pk.remifentanil import RemifentanilMinto
    from tokens import pk_tokens as pk

    synth = paths.COHORTS["synthetic_v7"]
    cases = sorted((synth / "cases").glob("*.parquet"))
    demo = pk.load_clinical_map()["synthetic_v7"]
    DT_S = 5.0

    def anessim_ce(model, grid, rate_hold, bolus_grid):
        rate_min = rate_hold / 3.0
        infs = []
        i = 1
        while i < len(grid):
            j = i
            while j + 1 < len(grid) and rate_min[j + 1] == rate_min[i]:
                j += 1
            if rate_min[i] != 0.0:
                infs.append(Infusion(start_s=grid[i - 1], end_s=grid[j] - 1e-9,
                                     rate=rate_min[i], drug=model.drug))
            i = j + 1
        boluses = [(grid[i - 1] if i > 0 else grid[0], float(bolus_grid[i]))
                   for i in np.where(bolus_grid > 0)[0]]
        st = model.simulate(infs, grid, boluses=boluses)
        return st[:, 3]

    prop_errs: list[float] = []
    remi_errs: list[float] = []
    for case_path in cases[:20]:
        cid = int(case_path.stem)
        row = demo[cid]
        d = dict(weight=float(row["weight"]), age=float(row["age"]),
                 height=float(row["height"]), sex=str(row["sex"]))
        schema = pq.read_schema(case_path).names
        cols = ["time"]
        for c in ("Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE",
                  "ppf_bolus_mg", "remi_bolus_ug"):
            if c in schema:
                cols.append(c)
        df = pq.read_table(case_path, columns=cols).to_pandas()
        t = df["time"].to_numpy(float)
        grid = np.arange(t[0], t[-1], DT_S)
        if len(grid) < 2:
            continue
        models = [
            ("propofol", "Orchestra/PPF20_RATE", "ppf_bolus_mg",
             PropofolSchnider(weight_kg=d["weight"], age_y=d["age"],
                              height_cm=d["height"], sex=d["sex"])),
            ("remifentanilo", "Orchestra/RFTN20_RATE", "remi_bolus_ug",
             RemifentanilMinto(weight_kg=d["weight"], height_cm=d["height"],
                               age_y=d["age"], sex=d["sex"])),
        ]
        for drug, rate_col, bol_col, model in models:
            if rate_col not in df.columns:
                continue
            rate_hold = pk._forward_fill_hold(t, df[rate_col].to_numpy(float), grid)
            bolus = pk._bolus_on_grid(t, df[bol_col].to_numpy(float), grid)
            ce_my = pk.compute_ce(drug, rate_hold, bolus, d, dt_s=DT_S)
            ce_an = anessim_ce(model, grid, rate_hold, bolus)
            ok = ce_an > 0.1
            if ok.sum() > 10:
                err = np.abs(ce_my[ok] - ce_an[ok]) / ce_an[ok]
                (prop_errs if drug == "propofol" else remi_errs).extend(err.tolist())

    mean_prop = float(np.mean(prop_errs))
    mean_remi = float(np.mean(remi_errs))
    # validación v7: propofol 0.086 %, remifentanilo 0.081 %; siempre < 1 %.
    assert mean_prop < 0.01, f"propofol error relativo {mean_prop:.5f} >= 1 %"
    assert mean_remi < 0.01, f"remi error relativo {mean_remi:.5f} >= 1 %"


# ---------------------------------------------------------------------------
# D4 — CONTEXTO: vocabulario de 86 ítems con la MISMA lista de item_id
# ---------------------------------------------------------------------------

# Lista vigente (12 v1 + 74 v2), congelada antes del repunte (sha vocab.json
# 945b7884… registrado en tokens_v1_manifest.json como sha256_context_vocab).
FROZEN_ITEM_IDS = [
    "age", "height", "weight", "bmi", "preop_gluc", "preop_pao2", "preop_paco2",
    "preop_sao2", "preop_cr", "preop_bun", "preop_na", "preop_k", "preop_alb",
    "preop_ast", "preop_alt", "preop_pt", "preop_aptt", "preop_plt", "preop_hb",
    "preop_ph", "preop_hco3", "preop_be", "tubesize", "sex:F", "preop_htn",
    "preop_dm", "preop_ecg:anormal", "preop_pft:anormal", "asa:1", "asa:2",
    "asa:3", "asa:4", "asa:otros", "emop:0", "emop:1", "optype:biliary_pancreas",
    "optype:breast", "optype:colorectal", "optype:general", "optype:hepatic",
    "optype:major_resection", "optype:minor_resection", "optype:others",
    "optype:stomach", "optype:thyroid", "optype:transplantation",
    "optype:vascular", "approach:open", "approach:robotic", "approach:videoscopic",
    "position:left_lateral_decubitus", "position:lithotomy", "position:otros",
    "position:prone", "position:reverse_trendelenburg",
    "position:right_lateral_decubitus", "position:supine", "ane_type:general",
    "ane_type:mac", "ane_type:neuraxial", "ane_type:otros",
    "ane_type:sedationalgesia", "ane_type:spinal", "top_diagnosis_chapter:c",
    "top_diagnosis_chapter:d", "top_diagnosis_chapter:e", "top_diagnosis_chapter:g",
    "top_diagnosis_chapter:h", "top_diagnosis_chapter:i", "top_diagnosis_chapter:j",
    "top_diagnosis_chapter:k", "top_diagnosis_chapter:l", "top_diagnosis_chapter:m",
    "top_diagnosis_chapter:n", "top_diagnosis_chapter:o", "top_diagnosis_chapter:otros",
    "top_diagnosis_chapter:r", "top_diagnosis_chapter:s", "top_diagnosis_chapter:t",
    "top_diagnosis_chapter:z", "airway:oral", "airway:otros", "cormack:i",
    "cormack:ii", "cormack:iiia", "cormack:otros",
]


@INTEGRATION
def test_d4_context_vocab_86_items_same_item_ids(tmp_path):
    from tokens import context_vocab as cv

    summary = cv.run(tmp_path)
    vocab = summary["items"]
    item_ids = [it["item_id"] for it in vocab]

    assert len(item_ids) == 86
    n_v1 = sum(1 for it in vocab if it.get("scope") == "v1")
    n_v2 = sum(1 for it in vocab if it.get("scope") == "v2")
    assert n_v1 == 12 and n_v2 == 74
    assert item_ids == FROZEN_ITEM_IDS, (
        f"la lista de item_id difiere de la vigente: "
        f"faltan {sorted(set(FROZEN_ITEM_IDS) - set(item_ids))}, "
        f"sobran {sorted(set(item_ids) - set(FROZEN_ITEM_IDS))}")


# ---------------------------------------------------------------------------
# D5 — TOKENIZADOR (prueba de humo): 50 casos reales de val con semilla fija
# ---------------------------------------------------------------------------

@INTEGRATION
def test_d5_tokenizer_smoke_50_real_val_cases():
    from tokens import tokenize as tk

    manifest = json.loads(
        (paths.TOKENS_DIR / "manifest_tokens.json").read_text(encoding="utf-8"))
    frozen_stats = manifest["normalization_stats"]
    dense_val = manifest["dense_val_enabled"]

    excluded = frozenset(tk.load_cases_without_phase_marks())
    context_map = tk.load_context_map()
    cf_meta = tk.load_cf_meta()

    cases = tk.load_cases()
    real_val_ids = sorted(int(c) for c in cases[
        (cases.source == "real") & (cases.split == "val")
        & (~cases.caseid.isin(excluded)) & (cases.n_windows > 0)].caseid)
    rng = np.random.default_rng(20260930)
    sel = set(int(c) for c in rng.choice(real_val_ids, 50, replace=False))

    # filas vigentes en data/tokens_v1 para esos 50 casos
    out_val = paths.TOKENS_DIR / "windows" / "source=real" / "split=val"
    existing = pd.concat(
        [pq.read_table(p).to_pandas() for p in sorted(out_val.glob("part-*.parquet"))],
        ignore_index=True)
    existing = existing[existing.caseid.isin(sel)]

    # reconstruir desde windows_v4 (paths) con los estadísticos CONGELADOS
    win_val = paths.WINDOWS_DIR / "windows" / "source=real" / "split=val"
    frames = []
    for part in sorted(win_val.glob("part-*.parquet")):
        w, _ = tk.process_partition_windows(part, dense_val, context_map, cf_meta,
                                            excluded_caseids=excluded)
        if len(w["t0"]) == 0:
            continue
        norm = tk.apply_normalization(w, frozen_stats)
        tbl = tk.build_table(norm, tk.load_vocab_v1()).to_pandas()
        tbl = tbl[tbl.caseid.isin(sel)]
        if len(tbl):
            frames.append(tbl)
    built = pd.concat(frames, ignore_index=True)

    assert set(built.caseid.unique()) == set(sel), "los 50 casos deben emitir ventanas"
    built = built.sort_values(["caseid", "t0"], kind="stable").reset_index(drop=True)
    existing = existing.sort_values(["caseid", "t0"], kind="stable").reset_index(drop=True)

    assert list(built.columns) == list(existing.columns)
    assert len(built) == len(existing), \
        f"n filas {len(built)} vs {len(existing)}"

    def _col_equal(a, b) -> bool:
        a = a.reset_index(drop=True)
        b = b.reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = a.to_numpy(dtype=np.float64, na_value=np.nan)
            bv = b.to_numpy(dtype=np.float64, na_value=np.nan)
            return bool(np.allclose(av, bv, rtol=1e-6, atol=1e-6, equal_nan=True))
        na = a.isna().to_numpy()
        nb = b.isna().to_numpy()
        if not np.array_equal(na, nb):
            return False
        if not (~na).any():
            return True  # ambas totalmente nulas
        return bool((a.astype(str).to_numpy()[~na] == b.astype(str).to_numpy()[~nb]).all())

    # Columnas propias del tokenizador (fármaco/ventilación/tiempo + máscaras +
    # metadatos): coincidencia EXACTA con los estadísticos congelados del
    # manifest. Las columnas ctx_* son valores YA normalizados por context_vocab
    # (data/context_v1): sus estadísticos se recalcularon con las cohortes v7 en
    # D4, así que el patrón de emisión debe coincidir pero no el valor numérico.
    ctx_cols = [c for c in built.columns if c.startswith("ctx_")]
    exact_cols = [c for c in built.columns if c not in ctx_cols]
    for c in exact_cols:
        assert _col_equal(built[c], existing[c]), f"columna {c} difiere"
    for c in ctx_cols:
        a = built[c].to_numpy(dtype=np.float64, na_value=np.nan)
        b = existing[c].to_numpy(dtype=np.float64, na_value=np.nan)
        # mismo patrón de emisión (NaN vs valor) y valores finitos donde se emite
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"columna {c}: patrón de emisión"
        assert np.all(np.isfinite(a[~np.isnan(a)])), f"columna {c}: no finitos"
