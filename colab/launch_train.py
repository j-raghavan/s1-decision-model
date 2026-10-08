"""Notebook cell for the full Tier S training run, with a cost guard and a busy kernel.

Expects /content/s1_train.tgz (the s1/ package and data/train/targets_v1.jsonl).
After GUARD_STEP optimizer steps it projects the run's total cost from measured
step time; if the projection exceeds MAX_UNITS at RATE units/hour, it stops the
run before spending more. Like launch_label.py, it keeps the kernel busy for the
whole run and then holds the VM until the Mac uploads /content/downloaded.flag.

Edit the three constants below, then: colab exec -s <session> -f colab/launch_train.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

RATE = 5.30  # units/hour of this session, read from `colab usage` after `colab new`
MAX_UNITS = 11.5  # stop if the projected training cost exceeds this (12 approved, ~0.5 spent on the OOM attempt)
GUARD_STEP = 50
EPOCHS = 3

FLAG = Path("/content/downloaded.flag")
LOG = Path("/content/train.log")

FLAG.unlink(missing_ok=True)  # a flag left by an earlier attempt on this VM must not release this one early
print(subprocess.run("mkdir -p /content/s1 && tar xzf /content/s1_train.tgz -C /content/s1 && "
                     "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && "
                     "pip install -q 'transformers>=5.0' 2>&1 | tail -1", shell=True, capture_output=True, text=True).stdout,
      flush=True)

cmd = [sys.executable, "-m", "s1.train", "--targets", "/content/s1/data/train/targets_v1.jsonl", "--out", "/content/ckpt",
       "--epochs", str(EPOCHS), "--batch-tokens", "8192", "--accum", "4", "--max-len", "2048", "--eval-every", "300",
       "--save-every", "300"]
with LOG.open("w") as log:
    job = subprocess.Popen(cmd, cwd="/content/s1", stdout=log, stderr=subprocess.STDOUT)
    guarded = False
    while job.poll() is None:
        time.sleep(30)
        if guarded:
            continue
        lines = [json.loads(l) for l in LOG.read_text().splitlines() if l.startswith("{")]
        total = next((l["total_steps"] for l in lines if "total_steps" in l), None)
        at = [l for l in lines if l.get("step", 0) >= GUARD_STEP and "elapsed_s" in l]
        if total and at:
            s_per_step = at[0]["elapsed_s"] / at[0]["step"]
            projected = s_per_step * total / 3600 * RATE
            print(f"guard: {s_per_step:.2f} s/step x {total} steps -> {projected:.1f} units projected (cap {MAX_UNITS})",
                  flush=True)
            guarded = True
            if projected > MAX_UNITS:
                job.terminate()
                job.wait()
                print("STOPPED by cost guard", flush=True)
print(f"training exited with code {job.returncode}", flush=True)

t0 = time.time()
while not FLAG.exists() and time.time() - t0 < 1800:
    time.sleep(15)
print("download confirmed" if FLAG.exists() else "no download confirmation after 30 min; releasing", flush=True)
