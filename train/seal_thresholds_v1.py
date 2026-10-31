"""Seal the dev-stage thresholds of every model in models.json before any test-side inference (P5 gate)."""
import os
import datetime
import hashlib
import json
import sys
from pathlib import Path

E, models = Path(sys.argv[1]), json.load(open(sys.argv[2], encoding="utf-8"))
dst = E / "dev_thresholds_sealed.json"
if dst.exists():
    raise SystemExit("already sealed: " + str(dst))
if (E / "test").exists() and any((E / "test").iterdir()):
    raise SystemExit("test outputs exist before sealing")
sealed = {}
for name, spec in models.items():
    rec = json.loads((E / "dev" / name / "threshold.json").read_text(encoding="utf-8"))
    if rec["ckpt_sha256"] != hashlib.sha256(Path(os.path.expandvars(spec["ckpt"])).read_bytes()).hexdigest():
        raise SystemExit("checkpoint changed after dev stage: " + name)
    sealed[name] = {k: rec[k] for k in ["ckpt", "ckpt_sha256", "native_r95", "native_r95_float32_bits",
                                        "native_val_component_recall"]}
out = {"sealed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "models": sealed,
       "program_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "configs/"
                                             "experiment_program_v1.yaml").read_bytes()).hexdigest(),
       "test_outputs_present": False}
dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({"sealed": len(sealed), "at": out["sealed_at"]}))
