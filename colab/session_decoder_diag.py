"""Notebook cell: find why the HF scoring path disagrees with the teacher, then re-score the pilot adapter.

The HF bf16 path agreed with the Ollama 4-bit teacher on 91-98% of safety and routing cases but only 69-81% on
knowledge questions, where both were confidently different; batched and single scoring also disagreed on the
GPU while a tiny float32 copy of the architecture is exact on CPU. This cell, on one GPU:
  1. scores MedMCQA + MMLU-Pro (400 cases) through HF with each kernel variant (default, eager experts,
     eager attention, batch 1) and compares accuracy with the teacher on the same cases
  2. if a variant fixes accuracy, re-scores the pilot adapter on both suites with it
  3. installs vLLM and scores the untuned model in bf16 on both suites (the trusted reference), and the pilot
     adapter through vLLM LoRA if vLLM accepts it

Expects /content/s1_diag.tgz (s1/, eval/jevbench/, data/eval/, teacher predictions) and the pilot adapter in
/content/adapter_best/. Keeps the kernel busy and holds the VM until /content/downloaded.flag appears.

    colab exec -s <session> -f colab/session_decoder_diag.py
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = os.environ.get("DIAG_MODEL", "google/gemma-4-26B-A4B-it")
BASE = Path(os.environ.get("DIAG_BASE", "/content"))
LIMIT = ["--limit", os.environ["DIAG_LIMIT"]] if os.environ.get("DIAG_LIMIT") else []
CUDA = os.environ.get("DIAG_DEVICE", "cuda") == "cuda"
W = BASE / "s1"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
ADAPTER = BASE / "adapter_best"
KNOWLEDGE = ["medmcqa", "mmlu_pro"]
VARIANTS = {  # run id -> extra run_decoder arguments
    "diag-default-b16": ["--batch", "16"],
    "diag-eagerexperts-b16": ["--batch", "16", "--experts-impl", "eager"],
    "diag-eagerattn-b16": ["--batch", "16", "--attn-impl", "eager"],
    "diag-default-b1": ["--batch", "1"],
}


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def run(args: list[str], log: str, timeout_min: float) -> int:
    try:
        return subprocess.run(args, cwd=W, stdout=open(BASE / log, "w"), stderr=subprocess.STDOUT,
                              timeout=timeout_min * 60).returncode
    except subprocess.TimeoutExpired:
        return -9


def preds(suite: str, run_id: str) -> dict:
    p = W / f"results/raw/{suite}/{run_id}/predictions.jsonl"
    return {r["case_id"]: r for r in map(json.loads, p.open())} if p.exists() else {}


def accuracy(suite: str, run_id: str, slices: list[str] | None = None) -> float:
    by: dict[str, list[float]] = {}
    for r in preds(suite, run_id).values():
        if r["task_type"] == "score" or r.get("error") or (slices and r["slice"] not in slices):
            continue
        by.setdefault(r["slice"], []).append(float(str(r["pred"]).lower() == str(r["gold"]).lower()))
    return sum(sum(v) / len(v) for v in by.values()) / max(1, len(by))


def agreement(run_id: str, ref: str) -> float:
    a, b = preds("jevbench", run_id), preds("jevbench", ref)
    shared = [k for k in a if k in b and b[k].get("pred") is not None and not a[k].get("error")]
    return sum(str(a[k]["pred"]) == str(b[k]["pred"]) for k in shared) / max(1, len(shared))


def hf_stage() -> list[str]:
    """Kernel variants on knowledge slices; returns the extra arguments of the variant to use for the adapter."""
    for run_id, extra in VARIANTS.items():
        t0 = time.time()
        rc = run([sys.executable, "eval/jevbench/run_decoder.py", "--model", MODEL, "--run-id", run_id,
                  "--suite", "jevbench", "--slices", *KNOWLEDGE, *extra, *LIMIT], f"{run_id}.log", 15)
        status(f"{run_id}: exit {rc}, {time.time() - t0:.0f}s, knowledge acc {accuracy('jevbench', run_id, KNOWLEDGE):.3f}, "
               f"agree w/ teacher {agreement(run_id, 'gemma4-26b'):.2f}, w/ default {agreement(run_id, 'diag-default-b16'):.2f}")
    teacher = accuracy("jevbench", "gemma4-26b", KNOWLEDGE)
    base = accuracy("jevbench", "diag-default-b16", KNOWLEDGE)
    best = max(VARIANTS, key=lambda r: accuracy("jevbench", r, KNOWLEDGE))
    status(f"teacher knowledge acc {teacher:.3f}; default {base:.3f}; best variant {best} "
           f"{accuracy('jevbench', best, KNOWLEDGE):.3f}")
    return VARIANTS[best] if accuracy("jevbench", best, KNOWLEDGE) >= base + 0.04 else []


def rescore_adapter(extra: list[str]) -> None:
    if not extra:
        status("no HF variant fixes accuracy; skipping HF adapter re-score")
        return
    for run_id, adapter in (("g26-hf-zeroshot-fixed", None), ("g26-pilot-fixed", ADAPTER)):
        for suite in ("jevbench", "custom"):
            args = [sys.executable, "eval/jevbench/run_decoder.py", "--model", MODEL, "--run-id", run_id,
                    "--suite", suite, *extra, *LIMIT] + (["--adapter", str(adapter)] if adapter else [])
            rc = run(args, f"{run_id}_{suite}.log", 25)
            status(f"{run_id} {suite} exit {rc}")
        status(f"{run_id} JevBench accuracy {accuracy('jevbench', run_id):.3f}")


def vllm_stage() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "httpx", "tqdm"], check=True)
    status("vllm installed")
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", "8000",
           "--max-model-len", "8192", "--max-logprobs", "20", "--gpu-memory-utilization", "0.90", "--enforce-eager",
           "--enable-lora", "--max-lora-rank", "64", "--lora-modules", f"pilot={ADAPTER}"]
    for attempt, args in enumerate((cmd, cmd[: cmd.index("--enable-lora")])):
        server = subprocess.Popen(args, stdout=open(BASE / f"vllm{attempt}.log", "w"), stderr=subprocess.STDOUT)
        t0, ready = time.time(), False
        while time.time() - t0 < 900 and server.poll() is None:
            try:
                with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
                    ready = r.status == 200
            except Exception:
                pass
            if ready:
                break
            time.sleep(10)
        if ready:
            break
        server.terminate()
        server.wait(timeout=60)
        status(f"vLLM {'with LoRA ' if attempt == 0 else ''}did not start (exit {server.poll()}); see vllm{attempt}.log")
    if not ready:
        return
    lora = attempt == 0
    status(f"vLLM ready in {time.time() - t0:.0f}s, LoRA {'on' if lora else 'off'}")
    for model, run_id in ((MODEL, "gemma4-26b-bf16-vllm"), ("pilot", "g26-pilot-vllm")) if lora else ((MODEL, "gemma4-26b-bf16-vllm"),):
        for suite in ("jevbench", "custom"):
            rc = run([sys.executable, "eval/jevbench/run_ollama.py", "--backend", "vllm", "--template", "gemma4",
                      "--model", model, "--run-id", run_id, "--suite", suite, "--workers", "64"], f"{run_id}_{suite}.log", 15)
            status(f"{run_id} {suite} exit {rc}")
        status(f"{run_id} JevBench accuracy {accuracy('jevbench', run_id):.3f}")
    server.terminate()


def main() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_diag.tgz -C {W}", shell=True, check=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):  # uploaded flat by scripts/colab_upload.sh
        if (BASE / name).exists():
            ADAPTER.mkdir(exist_ok=True)
            (BASE / name).rename(ADAPTER / name)
    if CUDA:
        status(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                              capture_output=True, text=True).stdout.strip())
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.18.0", "peft==0.21.2",
                        "accelerate==1.15.0"], check=True)
        # Colab preinstalls torchao 0.10, which PEFT rejects when it builds LoRA layers; nothing here uses it.
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    t0 = time.time()
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "tokenizer*"])
    status(f"model downloaded in {time.time() - t0:.0f}s; adapter present: {(ADAPTER / 'adapter_model.safetensors').exists()}")
    rescore_adapter(hf_stage())
    if CUDA:
        vllm_stage()
    rc = subprocess.run(f"cd {W} && tar czf {BASE}/diag_out.tgz results/raw/*/diag-* results/raw/*/*-fixed "
                        f"results/raw/*/*-vllm -C {BASE} $(cd {BASE} && ls *.log)", shell=True).returncode
    status(f"packed diag_out.tgz exit {rc}")
    status("DIAG DONE")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
T0 = time.time()
try:
    main()
except Exception as exc:  # logged so the watcher sees it; the VM is still held for downloads
    status(f"ABORTED: {exc!r}"[:300])
status(f"elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("DIAG_HOLD_MIN", 30)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
