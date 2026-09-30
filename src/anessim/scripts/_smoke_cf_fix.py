"""Smoke test for corrected CF generation (1 pair per affected lever)."""
import json
import shutil
import time
from pathlib import Path

from anessim.scripts.generate_f2_cf import _worker
from anessim.config import SimulatorConfig

src = Path("D:/data/anestesia_world/cf_learning_v5")
master = json.loads((src / "metadata" / "counterfactual_manifest.json").read_text())
cfg_dict = SimulatorConfig.from_yaml(src / "config.yaml").to_dict()

affected = ["set_peep", "set_fio2", "sevo_mac", "remi_bolus", "ppf20_rate", "rftn20_rate"]
tmp = Path("D:/tmp/cf_fix_smoke")
if tmp.exists():
    shutil.rmtree(tmp)
for sub in ("cases", "truth", "metadata", "clinical"):
    (tmp / sub).mkdir(parents=True)
shutil.copy(src / "config.yaml", tmp / "config.yaml")

for lever in affected:
    entry = next(e for e in master["pairs"] if e["lever"] == lever and not e.get("skipped"))
    task = {
        "out_dir": str(tmp),
        "caseid": int(entry["caseid_a"]),
        "lever": lever,
        "seed": int(entry["seed"]),
        "collection": "learning",
        "config_dict": cfg_dict,
        "force": True,
    }
    t0 = time.time()
    r = _worker(task)
    status = "OK" if not r.get("error") else f"ERROR {r['error'][:120]}"
    print(f"{lever}: {status} ({time.time()-t0:.0f}s)", flush=True)

print("done", flush=True)
