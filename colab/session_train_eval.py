"""Notebook cell for a Tier S training session that also evaluates the result.

Expects /content/s1_train.tgz with s1/, eval/jevbench/{paths,run_s1}.py, data/train/<targets> and
data/eval/{jevbench_subset,custom_v0}.jsonl. Trains with the cost guard, then scores both held-out
suites with the best checkpoint on the same GPU, so the Mac only downloads results.

Keeps the kernel busy throughout and holds the VM until /content/downloaded.flag appears (or 30 min).
Edit the constants, then: colab exec -s <session> -f colab/session_train_eval.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

RATE = 5.30  # units/hour of this session (from `colab usage`)
MAX_UNITS = 7.0  # stop if projected training cost exceeds this (keeps the project within the approved 50 units)
GUARD_STEP = 50
TARGETS = "data/train/targets_v3.jsonl"
EPOCHS = 1
INIT_FROM = "/content/init.pt"  # v2 best checkpoint, uploaded separately
RUN_ID = "s1-v3"
W = Path("/content/s1")
LOG = Path("/content/train.log")
STATUS = Path("/content/session.log")
FLAG = Path("/content/downloaded.flag")


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
subprocess.run(f"mkdir -p {W} && tar xzf /content/s1_train.tgz -C {W} && pip install -q 'transformers>=5.0'", shell=True, check=True)
status(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip())

cmd = [sys.executable, "-m", "s1.train", "--targets", TARGETS, "--out", "/content/ckpt", "--epochs", str(EPOCHS),
       "--batch-tokens", "8192", "--accum", "4", "--max-len", "8192", "--eval-every", "300", "--save-every", "300",
       "--grad-checkpoint"] + (["--init-from", INIT_FROM] if INIT_FROM else [])
if INIT_FROM and not Path(INIT_FROM).exists():
    status(f"ABORTED: {INIT_FROM} missing; refusing to train from scratch by accident")
    raise SystemExit(1)
stopped = False
with LOG.open("w") as log:
    job = subprocess.Popen(cmd, cwd=W, stdout=log, stderr=subprocess.STDOUT)
    guarded = False
    while job.poll() is None:
        time.sleep(30)
        if guarded:
            continue
        lines = [json.loads(l) for l in LOG.read_text().splitlines() if l.startswith("{")]
        total = next((l["total_steps"] for l in lines if "total_steps" in l), None)
        at = [l for l in lines if l.get("step", 0) >= GUARD_STEP and "elapsed_s" in l]
        if total and at:
            projected = at[0]["elapsed_s"] / at[0]["step"] * total / 3600 * RATE
            status(f"guard: {total} steps, {projected:.1f} units projected for training (cap {MAX_UNITS})")
            guarded = True
            if projected > MAX_UNITS:
                job.terminate()
                job.wait()
                stopped = True
                status("STOPPED by cost guard")
status(f"training exit {job.returncode}")

if not stopped and Path("/content/ckpt/best.pt").exists():
    for suite in ["custom", "jevbench"]:
        rc = subprocess.run([sys.executable, "eval/jevbench/run_s1.py", "--checkpoint", "/content/ckpt/best.pt",
                             "--run-id", RUN_ID, "--suite", suite], cwd=W, stdout=open(f"/content/eval_{suite}.log", "w"),
                            stderr=subprocess.STDOUT).returncode
        status(f"eval {suite} exit {rc}")
    status("TRAIN+EVAL DONE")
else:
    status("ABORTED before evaluation")

t0 = time.time()
while not FLAG.exists() and time.time() - t0 < 1800:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation after 30 min; releasing")
