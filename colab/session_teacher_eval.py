"""Notebook cell: measure a candidate teacher on both eval suites with vLLM (next-token logprobs, one pass).

Expects /content/s1_eval.tgz with eval/jevbench/{run_ollama,paths}.py and data/eval/*.jsonl. Keeps the kernel
busy throughout (Colab reclaims idle kernels) and holds the VM until /content/downloaded.flag appears.

    colab exec -s <session> -f colab/session_teacher_eval.py
"""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Set per run. Measured so far: openai/gpt-oss-120b (harmony, run gptoss-120b); google/gemma-4-31B-it (gemma4, run gemma4-31b).
MODEL, TEMPLATE, RUN_ID = "google/gemma-4-31B-it", "gemma4", "gemma4-31b"
W = Path("/content/s1")
STATUS = Path("/content/session.log")
FLAG = Path("/content/downloaded.flag")


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
subprocess.run(f"mkdir -p {W} && tar xzf /content/s1_eval.tgz -C {W}", shell=True, check=True)
status(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True,
                      text=True).stdout.strip())
try:
    import vllm  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "httpx", "tqdm"], check=True)
status("vllm ready")

server = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", "8000",
                           "--max-model-len", "8192", "--max-logprobs", "20", "--gpu-memory-utilization", "0.92",
                           "--enforce-eager"], stdout=open("/content/vllm.log", "w"), stderr=subprocess.STDOUT)
t0, ready = time.time(), False
while time.time() - t0 < 1800 and server.poll() is None:
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
            ready = r.status == 200
    except Exception:
        pass
    if ready:
        break
    time.sleep(10)
if not ready:
    status(f"ABORTED: server not ready (exit {server.poll()}); see /content/vllm.log")
else:
    status(f"server ready in {time.time() - t0:.0f}s")
    for suite in ["jevbench", "custom"]:
        rc = subprocess.run([sys.executable, "eval/jevbench/run_ollama.py", "--backend", "vllm", "--template", TEMPLATE,
                             "--model", MODEL, "--run-id", RUN_ID, "--suite", suite, "--workers", "64"], cwd=W,
                            stdout=open(f"/content/eval_{suite}.log", "w"), stderr=subprocess.STDOUT).returncode
        status(f"eval {suite} exit {rc}")
    status("TEACHER EVAL DONE")
server.terminate()

t0 = time.time()
while not FLAG.exists() and time.time() - t0 < 1800:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation after 30 min; releasing")
