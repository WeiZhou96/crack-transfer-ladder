"""Addendum-3 scheduler (experiment_program_v1_addendum3): trainer control B0_v6 and second backbone R50_{B0,B2,B11},
student_v6, 80k iters, seeds 0-2 (12 runs, seed-major order). Starts at once and takes GPUs idle when checked.

Idle = memory.used < 2500 MiB and utilization < 10% on two checks 60 s apart, and not already used by this
scheduler. Before launch the draft run manifest is re-hashed; any change aborts. A failed run is recorded and not
retried. Writes ADD3_DONE.json when all runs have ended; evaluation is chained separately.
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
S = D / "students_v6_add3"
DRAFT = D / "b10b11_p0p2_20261028/run_manifest_draft.json"
PROGRAM = C / "configs/experiment_program_v1.yaml"
ARMS = C / "configs/e1_arms_ablation_v1.yaml"
ADDENDUM = C / "configs/experiment_program_v1_addendum3.yaml"
JOBS = []
QA = ["--quad-dir", str(D / "quadruplets/v4"), "--tau-prereg", str(C / "configs/prereg_v6.yaml")]
for s_ in (0, 1, 2):
    JOBS += [(f"B0_v6_s{s_}", "B0", s_, []),
             (f"R50_B0_s{s_}", "B0", s_, ["--backbone", "upernet_resnet50"]),
             (f"R50_B11_s{s_}", "B11", s_, ["--backbone", "upernet_resnet50"] + QA),
             (f"R50_B2_s{s_}", "B2", s_, ["--backbone", "upernet_resnet50",
                                          "--neg-list", str(D / "negpool_r20/admitted.csv")])]


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
manifest.update(status="add3_launched", launched_at=now(), scheduler=str(C / "train/run_add3_20261028.py"),
                scheduler_sha256=sha(C / "train/run_add3_20261028.py"), trainer="train/student_v6.py",
                trainer_sha256=sha(C / "train/student_v6.py"), program_sha256=sha(PROGRAM), arms_sha256=sha(ARMS),
                addendum_sha256=sha(ADDENDUM))
(S / "run_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
log(event="start", jobs=JOBS)

env = {**os.environ, "OMP_NUM_THREADS": "6"}
pending, running, done = list(JOBS), {}, []
while pending or running:
    for gpu, (job, proc, fh) in list(running.items()):
        rc = proc.poll()
        if rc is not None:
            fh.close()
            ok = rc == 0 and (S / job[0] / "final.pt").exists()
            log(event="end", run=job[0], gpu=gpu, rc=rc, ok=ok)
            done.append({"run": job[0], "gpu": gpu, "rc": rc, "ok": ok})
            del running[gpu]
    if pending:
        first = idle_gpus() - set(running)
        time.sleep(60) if first else None
        free = sorted((first & idle_gpus()) - set(running)) if first else []
        for gpu in free:
            if not pending:
                break
            name, arm, seed, extra = pending.pop(0)
            out = S / name
            fh = open(S / f"{name}.log", "w", encoding="utf-8")
            cmd = [PY, "-X", "utf8", str(C / "train/student_v6.py"), str(C), str(D), arm, str(seed), str(out),
                   "--gpu", str(gpu)] + extra
            running[gpu] = ((name,), subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), fh)
            log(event="launch", run=name, gpu=gpu, pid=running[gpu][1].pid, cmd=cmd)
    time.sleep(120)
(S / "ADD3_DONE.json").write_text(json.dumps({"finished_at": now(), "runs": done}, indent=1), encoding="utf-8")
log(event="all_done", ok=sum(r["ok"] for r in done), total=len(done))
