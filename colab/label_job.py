"""Colab job: serve the Gemma 4 teacher with vLLM and label uploaded rows.

Runs on the Colab VM (uploaded with the repo's pipeline/ and eval/jevbench/ code).
It picks precision from the GPU it lands on: full BF16 on an 80 GB card, FP8
weights (weight-only, supported on A100) on a 40 GB card. Labels stream to
/content/s1/data/labels so they can be downloaded while the job runs.

    python colab/label_job.py --rows /content/s1/data/train/throughput_2k.jsonl --max-minutes 12

Stops on its own after --max-minutes of labeling, so a test run cannot overspend.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "google/gemma-4-26B-A4B-it"
WORK = Path("/content/s1")


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def gpu_memory_gb() -> float:
    return float(sh("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits").split()[0]) / 1024


def wait_for_server(url: str, server: subprocess.Popen, timeout_s: int = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if server.poll() is not None:
            raise SystemExit(f"vLLM server exited with code {server.returncode} before becoming healthy; see /content/vllm.log")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(10)
    raise SystemExit("vLLM server did not become healthy in time; see /content/vllm.log")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--max-minutes", type=float, default=12)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--compile", action="store_true",
                    help="allow torch.compile; off by default because vLLM 0.30 + Gemma 4 crashed in Inductor on A100")
    args = ap.parse_args()

    print(sh("nvidia-smi --query-gpu=name,memory.total --format=csv"), flush=True)
    mem = gpu_memory_gb()
    quant = [] if mem >= 70 else ["--quantization", "fp8"]
    print(f"GPU memory {mem:.0f} GB -> {'BF16' if not quant else 'FP8 weights'}", flush=True)

    t0 = time.time()
    try:
        import vllm  # noqa: F401  (already installed on a reused VM)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "httpx", "tqdm", "numpy"], check=True)
    print(f"pip install done in {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    log = open("/content/vllm.log", "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", str(args.port),
         "--max-model-len", "8192", "--max-logprobs", "20", "--gpu-memory-utilization", "0.92", *quant,
         *([] if args.compile else ["--enforce-eager"])],
        stdout=log, stderr=subprocess.STDOUT)
    url = f"http://localhost:{args.port}"
    wait_for_server(url, server)
    print(f"vLLM ready in {time.time() - t1:.0f}s (download + load)", flush=True)

    t2 = time.time()
    label = subprocess.Popen(
        [sys.executable, str(WORK / "pipeline" / "label.py"), args.rows, "--backend", "vllm", "--model", MODEL,
         "--base-url", url, "--workers", str(args.workers), "--out", str(WORK / "data" / "labels")],
        cwd=WORK)
    try:
        label.wait(timeout=args.max_minutes * 60)
    except subprocess.TimeoutExpired:
        label.terminate()
        label.wait()
        print(f"stopped labeling at the {args.max_minutes}-minute limit", flush=True)
    labeling_s = time.time() - t2

    labels = list((WORK / "data" / "labels").glob("*.labels.jsonl"))
    n = sum(1 for p in labels for _ in p.open())
    tokens = sum(json.loads(l).get("input_tokens", 0) for p in labels for l in p.open())
    summary = {"gpu_memory_gb": round(mem), "precision": "bf16" if not quant else "fp8", "rows_labeled": n,
               "labeling_seconds": round(labeling_s), "rows_per_s": round(n / labeling_s, 2),
               "prompt_tokens_per_s": round(tokens / labeling_s), "setup_seconds": round(t2 - t0)}
    print("SUMMARY " + json.dumps(summary), flush=True)
    (WORK / "data" / "labels" / "throughput_summary.json").write_text(json.dumps(summary, indent=2))
    server.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
