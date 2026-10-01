"""Inventario de artefactos con sha256 + manifiesto global (PARTE 1.2/1.4)."""
import hashlib
import json
import pathlib
import shutil
from datetime import datetime, timezone

import paths

ROOT = pathlib.Path(".")


def sha256(p):
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_sha(dirpath, pattern="*"):
    """sha256 de la concatenación de los sha de los archivos bajo dirpath."""
    p = pathlib.Path(dirpath)
    if not p.exists():
        return "missing"
    hashes = []
    for f in sorted(p.rglob(pattern)):
        if f.is_file():
            hashes.append(f"{f.relative_to(p)}:{sha256(f)}")
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()


inv = {}

# ── Cohortes (directorios de casos) ──
cohorts = {
    "real": str(paths.COHORTS["real"]),
    "synthetic_v5": str(paths.LOST_COHORT_DIRS["synthetic_v5"]),
    "vaso_reinf_v5": str(paths.LOST_COHORT_DIRS["vaso_reinf_v5"]),
    "cf_v5": str(paths.LOST_COHORT_DIRS["cf_v5"]),
    "synthetic_v6": str(paths.LOST_COHORT_DIRS["synthetic_v6"]),
    "vaso_reinf_v6": str(paths.LOST_COHORT_DIRS["vaso_reinf_v6"]),
    "cf_v6": str(paths.LOST_COHORT_DIRS["cf_v6"]),
    "synthetic_v7": str(paths.COHORTS["synthetic_v7"]),
    "vaso_reinf_v7": str(paths.COHORTS["vaso_reinf_v7"]),
    "cf_v7": str(paths.COHORTS["cf_v7"]),
}
inv["cohorts"] = {}
for name, path in cohorts.items():
    p = pathlib.Path(path)
    status = "presente" if p.exists() else "perdido"
    inv["cohorts"][name] = {
        "status": status,
        "n_cases_parquet": len(list((p / "cases").glob("*.parquet"))) if p.exists() else 0,
        "sha256_dir_cases": dir_sha(p / "cases", "*.parquet") if p.exists() else "missing",
    }

# ── Ventanas ──
windows = {
    "windows_v2": str(paths.WINDOWS_V2_DIR),
    "windows_v3": str(paths.WINDOWS_V3_DIR),
    "windows_v4": str(paths.WINDOWS_DIR),
}
inv["windows"] = {}
for name, path in windows.items():
    p = pathlib.Path(path)
    if p.exists():
        manifest = p / "manifest.json"
        inv["windows"][name] = {
            "status": "presente",
            "sha256_manifest": sha256(manifest),
            "n_partitions": len(list((p / "windows").rglob("part-*.parquet"))),
        }
    else:
        inv["windows"][name] = {"status": "perdido", "sha256_manifest": "missing"}

# ── Conjuntos de tokens / AE ──
inv["tokens_v1"] = {"status": "presente" if paths.TOKENS_DIR.exists() else "perdido",
                    "sha256_manifest": sha256(paths.TOKENS_DIR / "manifest_tokens.json")}
inv["ae_v1_ae_bal"] = {"status": "presente",
                       "sha256_norm_stats": sha256(paths.AE_V1_DIR / "ae_bal" / "norm_stats.json"),
                       "sha256_manifest": sha256(paths.AE_V1_DIR / "ae_bal" / "manifest_ae.json")}

# ── Scripts cerrados ──
scripts = {
    "autoencoder/window.py": "src/autoencoder/window.py",
    "tokens/pk_tokens.py": "src/tokens/pk_tokens.py",
    "tokens/context_vocab.py": "src/tokens/context_vocab.py",
    "tokens/tokenize.py": "src/tokens/tokenize.py",
    "ae/physio_ae.py": "src/ae/physio_ae.py",
    "diagnostics/v6_validate.py": "src/diagnostics/v6_validate.py",
    "diagnostics/v7_validate.py": "src/diagnostics/v7_validate.py",
    "diagnostics/v7_attribution.py": "src/diagnostics/v7_attribution.py",
    "diagnostics/v7_transfer_probe.py": "src/diagnostics/v7_transfer_probe.py",
    "diagnostics/cohort_gap.py": "src/diagnostics/cohort_gap.py",
    "diagnostics/gap_addendum.py": "src/diagnostics/gap_addendum.py",
    "anessim/simulate.py": "src/anessim/simulate.py",
    "anessim/respiratory.py": "src/anessim/respiratory.py",
}
inv["scripts"] = {}
for name, path in scripts.items():
    p = pathlib.Path(path)
    inv["scripts"][name] = {"status": "presente" if p.exists() else "perdido",
                            "sha256": sha256(p)}

manifest = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "inventory": inv,
}
(paths.DIAGNOSTICS_DIR / "v9_manifest_global.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

# ── Copia de seguridad de scripts cerrados fuera del árbol de trabajo ──
backup_dir = pathlib.Path("D:/data/anestesia_world/backup_scripts_20260928")
backup_dir.mkdir(parents=True, exist_ok=True)
copied = []
for name, path in scripts.items():
    src = pathlib.Path(path)
    if not src.exists():
        continue
    dst = backup_dir / name.replace("/", "__")
    shutil.copy2(src, dst)
    copied.append(str(dst))
manifest["backup_copied"] = copied
(paths.DIAGNOSTICS_DIR / "v9_manifest_global.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print("scripts copiados a", backup_dir, ":", len(copied))
print("manifiesto global:", paths.DIAGNOSTICS_DIR / "v9_manifest_global.json")
for name, d in inv["cohorts"].items():
    print(f"  cohort {name}: {d['status']} ({d['n_cases_parquet']} cases)")
for name, d in inv["windows"].items():
    print(f"  window {name}: {d['status']}")
