"""Notebook cell: evaluate the final training checkpoint (last.pt) on both suites, beside the loss-selected best.pt.

Run on the same VM right after session_train_eval.py has released its cell. Writes
results/raw/<suite>/s1-v2-final/ and holds the VM until /content/downloaded.flag appears (or 30 min).
"""

import subprocess
import sys
import time
from pathlib import Path

W = Path("/content/s1")
FLAG = Path("/content/downloaded.flag")
FLAG.unlink(missing_ok=True)
for suite in ["custom", "jevbench"]:
    rc = subprocess.run([sys.executable, "eval/jevbench/run_s1.py", "--checkpoint", "/content/ckpt/last.pt", "--run-id",
                         "s1-v2-final", "--suite", suite], cwd=W, stdout=open(f"/content/eval_final_{suite}.log", "w"),
                        stderr=subprocess.STDOUT).returncode
    print(f"eval final {suite} exit {rc}", flush=True)
    with open("/content/session.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} eval final {suite} exit {rc}\n")
with open("/content/session.log", "a") as f:
    f.write(f"{time.strftime('%H:%M:%S')} FINAL EVAL DONE\n")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < 1800:
    time.sleep(15)
