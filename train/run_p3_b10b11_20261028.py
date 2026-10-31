"""P3 scheduler (prereg_v6): B10/B11 x seeds 0-2 with student_v4, 80k iters, on GPUs that are idle when checked.

Idle = memory.used < 2500 MiB and utilization < 10% on two checks 60 s apart, and not already used by this
scheduler. Before launch the draft run manifest is re-hashed; any change aborts. A failed run is recorded and not
retried. Stops after the six runs; P4 evaluation is not started here.
"""
import sys
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
C, D = R, Path(os.environ.get("CRACKCF_DATA", "data"))
PY = sys.executable
S = D / "students_v4"
DRAFT = D / "b10b11_p0p2_20261028/run_manifest_draft.json"
JOBS = [("B11", 0), ("B10", 0), ("B11", 1), ("B10", 1), ("B11", 2), ("B10", 2)]


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def log(**kw):
    with open(S / "scheduler.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": now(), **kw}, ensure_ascii=False) + "\n")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def idle_gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    return {int(i) for i, m, u in (l.split(", ") for l in out.strip().splitlines()) if int(m) < 2500 and int(u) < 10}


S.mkdir(parents=False, exist_ok=False)
manifest = json.loads(DRAFT.read_text(encoding="utf-8"))
changed = [p for p, h in manifest["sha256"].items() if sha(p) != h]
if changed:
    log(event="abort_inputs_changed", files=changed)
    raise SystemExit(1)
manifest.update(status="P3_launched", launched_at=now(), scheduler=str(C / "train/run_p3_b10b11_20261028.py"),
                scheduler_sha256=sha(C / "train/run_p3_b10b11_20261028.py"))
(S / "run_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
log(event="start", jobs=JOBS)

env = {**os.environ}
pending, running, done = list(JOBS), {}, []
while pending or running:
    for gpu, (job, proc, fh) in list(running.items()):
        rc = proc.poll()
        if rc is not None:
            fh.close()
            ok = rc == 0 and (S / f"{job[0]}_s{job[1]}" / "final.pt").exists()
            log(event="end", arm=job[0], seed=job[1], gpu=gpu, rc=rc, ok=ok)
            done.append({"arm": job[0], "seed": job[1], "gpu": gpu, "rc": rc, "ok": ok})
            del running[gpu]
    if pending:
        first = idle_gpus() - set(running)
        time.sleep(60) if first else None
        free = sorted((first & idle_gpus()) - set(running)) if first else []
        for gpu in free:
            if not pending:
                break
            arm, seed = pending.pop(0)
            out = S / f"{arm}_s{seed}"
            fh = open(S / f"{arm}_s{seed}.log", "w", encoding="utf-8")
            cmd = [PY, "-X", "utf8", str(C / "train/student_v4.py"), str(C), str(D), arm, str(seed), str(out),
                   "--gpu", str(gpu), "--quad-dir", str(D / "quadruplets/v4"),
                   "--tau-prereg", str(C / "configs/prereg_v6.yaml")]
            running[gpu] = ((arm, seed), subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), fh)
            log(event="launch", arm=arm, seed=seed, gpu=gpu, pid=running[gpu][1].pid)
    time.sleep(120)
(S / "P3_DONE.json").write_text(json.dumps({"finished_at": now(), "runs": done}, indent=1), encoding="utf-8")
log(event="all_done", ok=sum(r["ok"] for r in done), total=len(done))
