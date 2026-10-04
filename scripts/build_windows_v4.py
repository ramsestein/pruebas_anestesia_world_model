"""build_windows_v4.py — genera las ventanas de las cohortes v7 (PASO 4).

Reutiliza window.py sin modificarlo (N1 del enunciado): monkeypatchea
OUT_DIR/SOURCES/EXCLUSIONS_PATH para apuntar a las cohortes v7 y llama a
window.generate().

Uso:
  python scripts/build_windows_v4.py --workers 14
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import paths  # noqa: E402

import autoencoder.window as W  # noqa: E402


def _is_cf_source(source: str) -> bool:
    base = W.SOURCES.get(source)
    if base is None:
        return False
    return bool(sorted((base / "metadata").glob("cf_pair_*.json")))


def _read_clinical_v6(source: str):
    """Versión agnóstica de nombre: detecta la estructura por ficheros."""
    base = W.SOURCES[source]
    if source == "real":
        df = pq.read_table(base / "clinical_data_enriched.parquet").to_pandas()
        df, diffs = W._dedup_clinical(df)
        return df, diffs
    if (base / "clinical_data.parquet").exists():
        df = pq.read_table(base / "clinical_data.parquet").to_pandas()
        return df, []
    # cohortes con fila clínica por caso (vaso_reinf_*, cf_*)
    rows = []
    for p in sorted((base / "cases").glob("*.parquet")):
        caseid = int(p.stem)
        clin = base / "clinical" / f"{caseid}_clinical.parquet"
        if clin.exists():
            try:
                t = pq.read_table(clin).to_pandas()
                if len(t):
                    rows.append(t.iloc[0])
            except Exception:
                pass
    return pd.DataFrame(rows), []


def list_cases_v6(source: str, exclusions=None):
    """Versión agnóstica de list_cases: usa _is_cf_source en vez de 'cf_v5'."""
    import json

    logs: list[str] = []
    clinical, diffs = _read_clinical_v6(source)
    base = W.SOURCES[source]
    case_paths = sorted(base.glob("cases/*.parquet"))
    disk_ids = {int(p.stem) for p in case_paths}
    ex = exclusions or {}

    clin = clinical[clinical["caseid"].isin(disk_ids)].copy()
    missing = disk_ids - set(clin["caseid"].astype(int).tolist())
    for cid in sorted(missing):
        logs.append(f"{source}: caso {cid} sin fila clínica -> excluido")

    if _is_cf_source(source):
        pairs: list[dict] = []
        for p in sorted((base / "metadata").glob("cf_pair_*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            pairs.append({
                "caseid_a": int(d["caseid_a"]),
                "caseid_b": int(d["caseid_b"]),
                "split_t": float(d.get("split_t", d.get("split_t_s", np.nan))),
                "lever": d.get("lever"),
                "collection": d.get("collection"),
            })
        pair_by_case: dict[int, dict] = {}
        parity_mismatches = 0
        for pr in pairs:
            a, b = pr["caseid_a"], pr["caseid_b"]
            pair_by_case[a] = pr
            pair_by_case[b] = pr
            if b != a + 1 or a % 2 == 0:
                parity_mismatches += 1
        if parity_mismatches:
            logs.append(f"{source}: {parity_mismatches} pares cuya metadata contradice la paridad impar/par")
        clin = clin.copy()
        clin["cf_pair_id"] = clin["caseid"].map(
            lambda cid: min(pair_by_case[cid]["caseid_a"], pair_by_case[cid]["caseid_b"])
            if cid in pair_by_case else None)
        clin["caseid_a"] = clin["caseid"].map(
            lambda cid: pair_by_case[cid]["caseid_a"] if cid in pair_by_case else None)
        clin["caseid_b"] = clin["caseid"].map(
            lambda cid: pair_by_case[cid]["caseid_b"] if cid in pair_by_case else None)
        clin["split_t"] = clin["caseid"].map(
            lambda cid: pair_by_case[cid]["split_t"] if cid in pair_by_case else None)
        clin["lever"] = clin["caseid"].map(
            lambda cid: pair_by_case[cid]["lever"] if cid in pair_by_case else None)
        clin["subjectid"] = clin["cf_pair_id"].map(
            lambda p: f"cf:{int(p)}" if p is not None else None)

    excluded = [c for c in clin["caseid"].astype(int).tolist() if c in ex.get(source, set())]
    if _is_cf_source(source):
        excl_set = set(ex.get(source, set()))
        to_add = set()
        for cid in list(excl_set):
            row = clin[clin["caseid"] == cid]
            if len(row) and row.iloc[0]["cf_pair_id"] is not None:
                to_add.add(int(row.iloc[0]["caseid_a"]))
                to_add.add(int(row.iloc[0]["caseid_b"]))
        excl_set |= to_add
        excluded = sorted(excl_set)
    n_before = len(clin)
    clin = clin[~clin["caseid"].astype(int).isin(set(excluded))].copy()
    if excluded:
        logs.append(f"{source}: {n_before - len(clin)} casos excluidos por exclusions_path (de {n_before})")

    keep = ["caseid", "subjectid", "casestart", "caseend",
            "anestart", "aneend", "opstart", "opend"]
    for extra in ("cf_pair_id", "caseid_a", "caseid_b", "split_t", "lever"):
        if extra in clin.columns:
            keep.append(extra)
    out = clin[keep].copy()
    out["source"] = source
    out["caseid"] = out["caseid"].astype(np.int64)
    return out, logs, diffs


def _sample_cases_v6(cases: pd.DataFrame) -> pd.DataFrame:
    """Muestra de la fase A con los nombres de fuente actuales (v6)."""
    rng = np.random.default_rng(20240917)
    keys = list(W.SOURCES.keys())
    synth_key = next((k for k in keys if "synthetic" in k), None)
    vaso_key = next((k for k in keys if "vaso" in k), None)
    cf_key = next((k for k in keys if k.startswith("cf")), None)

    parts = []
    real = cases[cases.source == "real"]
    if len(real):
        parts.append(real.iloc[np.sort(rng.choice(len(real), min(40, len(real)), replace=False))])
    if synth_key and synth_key in cases.source.unique():
        s = cases[cases.source == synth_key]
        parts.append(s.iloc[np.sort(rng.choice(len(s), min(20, len(s)), replace=False))])
    if vaso_key and vaso_key in cases.source.unique():
        v = cases[cases.source == vaso_key]
        parts.append(v.iloc[np.sort(rng.choice(len(v), min(10, len(v)), replace=False))])
    if cf_key and cf_key in cases.source.unique():
        cf = cases[cases.source == cf_key]
        pairs = sorted(cf["cf_pair_id"].dropna().astype(int).unique().tolist())
        if pairs:
            sel = [int(p) for p in rng.choice(pairs, min(5, len(pairs)), replace=False)]
            parts.append(cf[cf["cf_pair_id"].astype(float).isin([float(p) for p in sel])].copy())
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Destinos de la construcción. Por defecto, las cohortes y el directorio
# vigentes (comportamiento original). Las dos variables de entorno permiten
# reconstruir las ventanas con OTRA cohorte CF sin tocar el núcleo (paso 3d);
# son necesarias porque ``_init_worker`` corre en procesos NUEVOS (spawn en
# Windows) que re-importan este módulo y NO heredan ningún monkeypatch hecho en
# el padre, mientras que el entorno sí se hereda.
# ---------------------------------------------------------------------------
_CF_ENV = os.environ.get("ANESTESIA_CF_V7_DIR")
_WIN_OUT_ENV = os.environ.get("ANESTESIA_WINDOWS_OUT")

SOURCES_V7: dict[str, Path] = (
    {**paths.COHORTS, "cf_v7": Path(_CF_ENV)} if _CF_ENV else dict(paths.COHORTS)
)
OUT_DIR_V7 = Path(_WIN_OUT_ENV) if _WIN_OUT_ENV else paths.WINDOWS_DIR


def _init_worker() -> None:
    """Se ejecuta en CADA worker (spawn): reaplica SOURCES de v7."""
    import autoencoder.window as _W
    _W.SOURCES = SOURCES_V7
    _W.OUT_DIR = OUT_DIR_V7
    _W.EXCLUSIONS_PATH = paths.AUDIT_DIR / "_nonexistent_v7.csv"


def generate_v3(max_workers: int = 14) -> None:
    """Reimplementa window.generate() con initializer en el pool de workers
    (en Windows los workers son procesos nuevos y no heredan el monkeypatch)."""
    import json
    import time as _time
    from concurrent.futures import ProcessPoolExecutor

    W.SOURCES = SOURCES_V7
    W.OUT_DIR = OUT_DIR_V7
    W.EXCLUSIONS_PATH = paths.AUDIT_DIR / "_nonexistent_v7.csv"
    W._read_clinical = _read_clinical_v6
    W.list_cases = list_cases_v6
    W._sample_cases = _sample_cases_v6

    print("=" * 78)
    print("FASE B — generación del dataset (v7)")
    print("=" * 78)

    exclusions = W._load_exclusions()
    all_cases = []
    for source in W.SOURCES:
        df, logs, _ = W.list_cases(source, exclusions)
        all_cases.append(df)
        for line in logs:
            print("  [log]", line)
    cases = W.make_split(pd.concat(all_cases, ignore_index=True), W.SEED)
    cases = cases.sort_values(["source", "split", "caseid"], kind="stable").reset_index(drop=True)

    inventory = W._inventory_from_sources()
    sample_df = W._sample_cases(cases)
    presence = W._presence_in_sample(sample_df)
    registry = W.build_registry(inventory, presence)

    OUT_DIR_V7.mkdir(parents=True, exist_ok=True)
    win_dir = OUT_DIR_V7 / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)

    cases_path = OUT_DIR_V7 / "cases.parquet"
    errors_path = OUT_DIR_V7 / "errors.csv"
    done: set[int] = set()
    if cases_path.exists():
        try:
            done = set(int(c) for c in pq.read_table(cases_path, columns=["caseid"])["caseid"].to_pylist())
            print(f"    resume: {len(done)} casos ya procesados")
        except Exception:
            done = set()
    todo = cases[~cases["caseid"].astype(int).isin(done)].reset_index(drop=True)
    print(f"    casos a procesar: {len(todo)} (total {len(cases)})")

    case_rows = [dict(r) for _, r in todo.iterrows()]

    buffers: dict[tuple[str, str], list[pd.DataFrame]] = {}
    buffer_rows: dict[tuple[str, str], int] = {}
    part_idx: dict[tuple[str, str], int] = {}
    for key in [(s, sp) for s in W.SOURCES for sp in ("train", "val")]:
        sub = win_dir / f"source={key[0]}" / f"split={key[1]}"
        if sub.exists():
            existing = sorted(sub.glob("part-*.parquet"))
            if existing:
                part_idx[key] = max(int(p.stem.split("-")[1]) for p in existing) + 1
    cases_out: list[dict] = []
    errors_out: list[dict] = []
    n_done = 0
    t_start = _time.time()

    def flush(key: tuple[str, str]) -> None:
        if key not in buffers or not buffers[key]:
            return
        df = pd.concat(buffers[key], ignore_index=True)
        buffers[key] = []
        buffer_rows[key] = 0
        if key not in part_idx:
            part_idx[key] = 0
        out_sub = win_dir / f"source={key[0]}" / f"split={key[1]}"
        out_sub.mkdir(parents=True, exist_ok=True)
        path = out_sub / f"part-{part_idx[key]:05d}.parquet"
        pq.write_table(W._windows_to_table(df), path, compression="zstd")
        part_idx[key] += 1

    if len(case_rows) == 0:
        print("    nada que procesar (resume completo)")
    else:
        with ProcessPoolExecutor(max_workers=max_workers,
                                 initializer=_init_worker) as ex:
            for i, (stat, w) in enumerate(ex.map(W._worker_process, case_rows, chunksize=8)):
                n_done += 1
                if "error" in stat:
                    errors_out.append({"caseid": stat["caseid"],
                                       "source": stat.get("source", ""),
                                       "error": stat["error"]})
                else:
                    stat["subjectid"] = str(case_rows[i].get("subjectid"))
                    stat["source"] = str(case_rows[i].get("source"))
                    stat["split"] = str(case_rows[i].get("split"))
                    stat["t_first"] = int(w["t"].iloc[0]) if w is not None and len(w) else None
                    stat["t_last"] = int(w["t"].iloc[-1]) if w is not None and len(w) else None
                    cases_out.append(stat)
                    if w is not None and len(w):
                        key = (str(case_rows[i]["source"]), str(case_rows[i]["split"]))
                        buffers.setdefault(key, []).append(w)
                        buffer_rows[key] = buffer_rows.get(key, 0) + len(w)
                        if buffer_rows[key] >= W.PART_SIZE:
                            flush(key)
                if n_done % 500 == 0:
                    print(f"    {n_done}/{len(case_rows)} casos, "
                          f"{_time.time() - t_start:.0f}s")

        for key in list(buffers.keys()):
            flush(key)

    if done:
        try:
            old_cases = pq.read_table(cases_path).to_pandas()
            all_cases_out = pd.concat([old_cases, pd.DataFrame(cases_out)], ignore_index=True)
        except Exception:
            all_cases_out = pd.DataFrame(cases_out)
    else:
        all_cases_out = pd.DataFrame(cases_out)
    if len(all_cases_out):
        all_cases_out = all_cases_out.drop_duplicates(subset=["caseid"], keep="last")
        all_cases_out.to_parquet(cases_path, index=False)

    if errors_out:
        pd.DataFrame(errors_out).to_csv(errors_path, index=False)
        print(f"    errores: {len(errors_out)} en {errors_path.name}")

    split_df = cases[["subjectid", "source", "split"]].drop_duplicates().copy()
    split_df["subjectid"] = split_df["subjectid"].astype(str)
    split_df.to_parquet(OUT_DIR_V7 / "split.parquet", index=False)

    reg_out = registry.copy()
    if "duplicate_of" in reg_out.columns:
        reg_out["duplicate_of"] = reg_out["duplicate_of"].fillna("").astype(str)
    reg_out.to_parquet(OUT_DIR_V7 / "registry.parquet", index=False)

    manifest = {
        "date": pd.Timestamp.now().isoformat(),
        "seed": W.SEED,
        "grid_s": W.GRID_S,
        "max_age": W.MAX_AGE,
        "plausible_range": {k: list(v) for k, v in W.PLAUSIBLE_RANGE.items()},
        "image_tracks": W.IMAGE_TRACKS,
        "bolus_tracks": W.BOLUS_SHORT,
        "bolus_specs": {s: dict(spec) for s, spec in W.BOLUS_SPECS.items()},
        "exclusions_path": str(W.EXCLUSIONS_PATH),
        "exclusions_sha256": None,
        "n_cases_by_source_split": {
            f"{s}|{sp}": int(n)
            for (s, sp), n in cases.groupby(["source", "split"]).size().items()
        },
        "n_windows_by_source_split": W._count_windows_by_partition(win_dir),
        "total_bytes": W._dir_bytes(OUT_DIR_V7),
        "window_py_sha256": W._sha256(Path(W.__file__)),
        "n_cases_processed": int(len(pq.read_table(cases_path))),
        "n_errors": int(len(errors_out)),
    }
    (OUT_DIR_V7 / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("    manifest.json escrito")
    print(f"    TOTAL: {len(cases_out)} casos procesados, {len(errors_out)} errores, "
          f"{_time.time() - t_start:.0f}s")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    generate_v3(max_workers=args.workers)


if __name__ == "__main__":
    main()
