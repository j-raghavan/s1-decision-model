"""Notebook cell: is Unsloth's faster path numerically the same training? Run on the VM left by the pilot-2 cell
(model cached, Unsloth installed in /content/unsloth_pkgs).

1. Forward + backward equality on 16 fixed rows with dropout off and seeded non-zero LoRA weights: plain,
   the same rows as two batches of 8 (bf16 noise floor), and Unsloth (colab/unsloth_equality.py).
2. Training trajectory with dropout off: 20 identical steps plain and with Unsloth; losses at steps 10 and 20
   must agree within 0.02.

Expects /content/s1_verify.tgz with the updated s1/ and colab/unsloth_equality.py.

    colab exec -s <session> -f colab/session_unsloth_verify.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/content")
W = BASE / "s1"
STATUS = BASE / "verify.log"
FLAG = BASE / "verified.flag"
UNSLOTH = {"PYTHONPATH": f"{BASE / 'unsloth_pkgs'}:{W}", "S1_UNSLOTH": "1", "UNSLOTH_MOE_GROUPED_TRITON": "1"}


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def run(args: list[str], log: str, env: dict | None = None, timeout_min: float = 15) -> int:
    try:
        return subprocess.run(args, cwd=W, stdout=open(BASE / log, "w"), stderr=subprocess.STDOUT,
                              timeout=timeout_min * 60, env={**os.environ, **(env or {})}).returncode
    except subprocess.TimeoutExpired:
        return -9


def losses(log: str) -> dict[int, float]:
    out = {}
    for line in open(BASE / log, errors="replace"):
        if line.startswith("{") and '"loss"' in line and "elapsed_s" in line:
            r = json.loads(line)
            out[r["step"]] = r["loss"]
    return out


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
subprocess.run(f"tar xzf {BASE}/s1_verify.tgz -C {W}", shell=True, check=True)
try:
    for mode, env in (("plain", None), ("split", None), ("unsloth", UNSLOTH)):
        rc = run([sys.executable, "colab/unsloth_equality.py", mode, str(BASE / f"eq_{mode}.pt")], f"eq_{mode}.log", env)
        status(f"equality pass {mode}: exit {rc}")
    run([sys.executable, "colab/unsloth_equality.py", "compare", *(str(BASE / f"eq_{m}.pt") for m in ("plain", "split", "unsloth"))],
        "eq_compare.log")
    status("equality: " + " ".join(open(BASE / "eq_compare.log").read().split()))

    common = ["-m", "s1.train_decoder", "--model", "google/gemma-4-26B-A4B-it", "--targets", "data/train/targets_pilot2.jsonl",
              "--max-steps", "20", "--batch", "16", "--accum", "2", "--lr", "1e-4", "--rank", "64", "--lora-dropout", "0",
              "--eval-every", "100000", "--val-limit", "16", "--skip-initial-eval", "--no-save"]
    for name, env in (("traj_plain", None), ("traj_unsloth", UNSLOTH)):
        rc = run([sys.executable, *common, "--out", str(BASE / name)], f"{name}.log", env)
        status(f"{name}: exit {rc}, losses {losses(f'{name}.log')}")
    p, u = losses("traj_plain.log"), losses("traj_unsloth.log")
    same = all(abs(p[s] - u[s]) <= 0.02 for s in (10, 20)) if {10, 20} <= set(p) & set(u) else False
    status(f"trajectory with dropout off: {'MATCH' if same else 'DIFFER'} (plain {p}, unsloth {u})")
except Exception as exc:
    status(f"ABORTED: {exc!r}"[:300])
status("VERIFY DONE")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < 900:
    time.sleep(15)
