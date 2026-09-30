"""v10 3: copia de seguridad de lo irreemplazable.

Copia data/windows_v4, data/synthetic_v7, data/synthetic_vaso_reinf_v7 y
data/cf_v7 a D:/data/anestesia_world/backup_v7_20260928/ con un manifiesto
sha256 por fichero, y verifica la copia releyendo los sha desde el destino.
"""
import hashlib
import json
import pathlib
import shutil
import sys
import time

SRC = pathlib.Path("data")
DST = pathlib.Path("D:/data/anestesia_world/backup_v7_20260928")
DIRS = ["windows_v4", "synthetic_v7", "synthetic_vaso_reinf_v7", "cf_v7"]

CHUNK = 1 << 22  # 4 MiB


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    DST.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}  # relpath -> {sha, size}
    total_bytes = 0

    # Fase 1: copia + sha de origen.
    for d in DIRS:
        src = SRC / d
        if not src.exists():
            print(f"[backup] FALTA {src}", flush=True)
            continue
        n_this = 0
        for f in sorted(src.rglob("*")):
            if not f.is_file():
                continue
            rel = str(f.relative_to(SRC))
            sha = sha256(f)
            size = f.stat().st_size
            manifest[rel] = {"sha256": sha, "size": size}
            total_bytes += size
            n_this += 1
            dst_f = DST / rel
            dst_f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst_f)
        print(f"[backup] copiado {d} ({n_this} ficheros)", flush=True)

    manifest_path = DST / "manifest_sha256.json"
    manifest_path.write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "total_bytes": total_bytes, "files": manifest},
        indent=2, ensure_ascii=False), encoding="utf-8")

    # Fase 2: verificación releyendo sha desde el destino.
    n_bad = 0
    n_ok = 0
    for rel, meta in sorted(manifest.items()):
        dst_f = DST / rel
        if not dst_f.exists():
            print(f"[verify] FALTA {rel}", flush=True)
            n_bad += 1
            continue
        if sha256(dst_f) != meta["sha256"]:
            print(f"[verify] SHA DIFERENTE {rel}", flush=True)
            n_bad += 1
        else:
            n_ok += 1

    result = {"ok": n_ok, "bad": n_bad, "total_files": len(manifest),
              "total_bytes": total_bytes, "elapsed_s": round(time.time() - t0, 1)}
    (DST / "verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[backup] VERIFICACIÓN: {n_ok} OK, {n_bad} MAL, "
          f"{len(manifest)} ficheros, {total_bytes/1e9:.2f} GB, "
          f"{result['elapsed_s']} s", flush=True)


if __name__ == "__main__":
    main()
