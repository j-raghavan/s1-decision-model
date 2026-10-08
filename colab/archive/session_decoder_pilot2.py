"""Notebook cell: training speed test, then decoder pilot 2 with the fastest verified setup, scored through vLLM.

1. Speed test: 20 identical steps (same seed, same batches, length-bucketed) per configuration:
     ckpt     gradient checkpointing on (pilot 1 setup, but bucketed)
     nockpt   gradient checkpointing off
     unsloth  checkpointing off + Unsloth's Triton grouped GEMM (replaces PyTorch's per-expert loop on sm_120),
              installed into a separate folder so the main environment is untouched
   Seconds per step are taken between steps 10 and 20; a configuration is usable only if its losses at steps
   10 and 20 match the checkpointed run within 0.02 (the maths is the same, so they should).
2. Pilot 2: LoRA on data/train/targets_pilot2.jsonl with the fastest usable configuration, 25-minute cap,
   best adapter chosen on held-out families.
3. vLLM bf16 serves the base model with the adapter as a LoRA and scores both suites (HF fallback if vLLM
   fails). The untuned bf16 baseline was scored in session s1-diag26.

Expects /content/s1_pilot2.tgz. Keeps the kernel busy and holds the VM until /content/downloaded.flag appears.

    colab exec -s <session> -f colab/session_decoder_pilot2.py

P2_* environment variables shrink everything for a local dry run with a small model.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = os.environ.get("P2_MODEL", "google/gemma-4-26B-A4B-it")
BASE = Path(os.environ.get("P2_BASE", "/content"))
CUDA = os.environ.get("P2_DEVICE", "cuda") == "cuda"
TRAIN_MINUTES = float(os.environ.get("P2_MINUTES", 25))
SPEED_STEPS = int(os.environ.get("P2_SPEED_STEPS", 20))
BATCH = os.environ.get("P2_BATCH", "16")
VAL_LIMIT = os.environ.get("P2_VAL_LIMIT", "400")
LIMIT = ["--limit", os.environ["P2_LIMIT"]] if os.environ.get("P2_LIMIT") else []
W = BASE / "s1"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
UNSLOTH_DIR = BASE / "unsloth_pkgs"
TARGETS = "data/train/targets_pilot2.jsonl"


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def run(args: list[str], log: str, timeout_min: float, env: dict | None = None) -> int:
    try:
        return subprocess.run(args, cwd=W, stdout=open(BASE / log, "w"), stderr=subprocess.STDOUT,
                              timeout=timeout_min * 60, env={**os.environ, **(env or {})}).returncode
    except subprocess.TimeoutExpired:
        return -9


def records(log: str) -> list[dict]:
    out = []
    for line in open(BASE / log, errors="replace"):
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def accuracy(suite: str, run_id: str) -> float:
    by: dict[str, list[float]] = {}
    p = W / f"results/raw/{suite}/{run_id}/predictions.jsonl"
    for r in map(json.loads, p.open()) if p.exists() else []:
        if r["task_type"] == "score" or r.get("error"):
            continue
        by.setdefault(r["slice"], []).append(float(str(r["pred"]).lower() == str(r["gold"]).lower()))
    return sum(sum(v) / len(v) for v in by.values()) / max(1, len(by))


def install_unsloth() -> dict | None:
    """Unsloth and unsloth-zoo (Triton grouped GEMM merged 2026-10-07, not yet released) into their own folder."""
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(UNSLOTH_DIR),
                         "unsloth", "git+https://github.com/unslothai/unsloth-zoo.git"],
                        stdout=open(BASE / "unsloth_install.log", "w"), stderr=subprocess.STDOUT).returncode
    # light dependencies only; torch, transformers and peft stay as pinned
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tyro", "msgspec", "cut_cross_entropy", "hf_transfer",
                    "sentencepiece", "protobuf", "bitsandbytes", "datasets", "trl", "--no-deps"],
                   stdout=open(BASE / "unsloth_deps.log", "w"), stderr=subprocess.STDOUT)
    status(f"unsloth install exit {rc}")
    if rc != 0:
        return None
    return {"PYTHONPATH": f"{UNSLOTH_DIR}:{W}", "S1_UNSLOTH": "1", "UNSLOTH_MOE_GROUPED_TRITON": "1"}


def speed_test() -> tuple[list[str], dict | None]:
    common = ["--model", MODEL, "--targets", TARGETS, "--max-steps", str(SPEED_STEPS), "--batch", BATCH, "--accum", "2",
              "--lr", "1e-4", "--rank", "64", "--eval-every", "100000", "--val-limit", "16", "--skip-initial-eval",
              "--no-save"]
    configs = [("ckpt", [], None), ("nockpt", ["--no-grad-checkpoint"], None)]
    results = {}
    for name, extra, env in configs + [("unsloth", None, "unsloth")]:
        if env == "unsloth":
            if not CUDA:
                continue
            # same memory setting as the fastest plain run that worked
            extra = ["--no-grad-checkpoint"] if "nockpt" in results else []
            env = install_unsloth()
            if env is None:
                continue
        rc = run([sys.executable, "-m", "s1.train_decoder", *common, *extra, "--out", str(BASE / f"speed_{name}")],
                 f"speed_{name}.log", 15, env)
        recs = records(f"speed_{name}.log")
        steps = {r["step"]: r for r in recs if "loss" in r and "elapsed_s" in r}
        done = next((r for r in recs if r.get("done")), {})
        info = next((r for r in recs if "grouped_mm_impl" in r), None)
        if rc != 0 or 10 not in steps or SPEED_STEPS not in steps:
            status(f"speed {name}: FAILED rc={rc} (see speed_{name}.log)")
            continue
        sps = (steps[SPEED_STEPS]["elapsed_s"] - steps[10]["elapsed_s"]) / (SPEED_STEPS - 10)
        results[name] = {"sps": sps, "l10": steps[10]["loss"], "l20": steps[SPEED_STEPS]["loss"], "extra": extra, "env": env}
        status(f"speed {name}: {sps:.2f} s/step, loss@10 {steps[10]['loss']:.4f} @{SPEED_STEPS} {steps[SPEED_STEPS]['loss']:.4f}, "
               f"peak {done.get('peak_mem_gb')} GB" + (f", {info}" if info else ""))
    ref = results.get("ckpt")
    usable = {n: r for n, r in results.items()
              if ref is None or (abs(r["l10"] - ref["l10"]) <= 0.02 and abs(r["l20"] - ref["l20"]) <= 0.02)}
    for n in set(results) - set(usable):
        status(f"speed {n}: losses differ from the checkpointed run by more than 0.02; not used")
    if not usable:
        return [], None
    best = min(usable, key=lambda n: usable[n]["sps"])
    status(f"fastest usable: {best} ({usable[best]['sps']:.2f} s/step"
           + (f", {ref['sps'] / usable[best]['sps']:.1f}x vs checkpointed)" if ref else ")"))
    return usable[best]["extra"], usable[best]["env"]


def pilot(extra: list[str], env: dict | None) -> Path | None:
    rc = run([sys.executable, "-m", "s1.train_decoder", "--model", MODEL, "--targets", TARGETS, "--out",
              str(BASE / "adapter2"), "--max-steps", "5000", "--batch", BATCH, "--accum", "2", "--lr", "1e-4",
              "--rank", "64", "--eval-every", "250", "--val-limit", VAL_LIMIT, "--max-minutes", str(TRAIN_MINUTES), *extra],
             "train2.log", TRAIN_MINUTES + 25, env)
    for r in records("train2.log"):
        if any(k in r for k in ("sec_per_step", "unseen_loss", "done", "device")):
            status("train " + json.dumps(r))
    status(f"train exit {rc}")
    for name in ("best", "last"):
        if (BASE / "adapter2" / name / "adapter_model.safetensors").exists():
            return BASE / "adapter2" / name
    return None


def hf_eval(adapter: Path) -> None:
    for suite in ("jevbench", "custom"):
        rc = run([sys.executable, "eval/jevbench/run_decoder.py", "--model", MODEL, "--adapter", str(adapter),
                  "--run-id", "g26-pilot2", "--suite", suite, "--batch", BATCH, *LIMIT], f"hf_{suite}.log", 25)
        status(f"HF eval {suite} exit {rc}")
    status(f"pilot 2 (HF) JevBench accuracy {accuracy('jevbench', 'g26-pilot2'):.3f}")


def vllm_eval(adapter: Path) -> bool:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "httpx", "tqdm"], check=True,
                   stdout=open(BASE / "vllm_install.log", "w"), stderr=subprocess.STDOUT)
    server = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", "8000",
                               "--max-model-len", "8192", "--max-logprobs", "20", "--gpu-memory-utilization", "0.90",
                               "--enforce-eager", "--enable-lora", "--max-lora-rank", "64", "--lora-modules", f"pilot2={adapter}"],
                              stdout=open(BASE / "vllm.log", "w"), stderr=subprocess.STDOUT)
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
    if not ready:
        status(f"vLLM did not start (exit {server.poll()}); see vllm.log")
        server.terminate()
        return False
    status(f"vLLM ready in {time.time() - t0:.0f}s")
    for suite in ("jevbench", "custom"):
        rc = run([sys.executable, "eval/jevbench/run_ollama.py", "--backend", "vllm", "--template", "gemma4", "--model",
                  "pilot2", "--run-id", "g26-pilot2-vllm", "--suite", suite, "--workers", "64"], f"vllm_{suite}.log", 15)
        status(f"vLLM eval {suite} exit {rc}")
    server.terminate()
    status(f"pilot 2 (vLLM) JevBench accuracy {accuracy('jevbench', 'g26-pilot2-vllm'):.3f} (untuned bf16 0.742)")
    return accuracy("jevbench", "g26-pilot2-vllm") > 0


def main() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_pilot2.tgz -C {W}", shell=True, check=True)
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
    status(f"model downloaded in {time.time() - t0:.0f}s")
    extra, env = speed_test()
    if not extra and env is None and not (BASE / "speed_ckpt.log").exists():
        status("ABORTED: no training configuration worked")
        return
    adapter = pilot(extra, env)
    if adapter is None:
        status("ABORTED: no adapter saved; see train2.log")
        return
    status(f"adapter {adapter}")
    if not (CUDA and vllm_eval(adapter)):
        hf_eval(adapter)
    rc = subprocess.run(f"cd {W} && tar czf {BASE}/pilot2_out.tgz results/raw/*/g26-pilot2* -C {BASE} "
                        f"adapter2/log.jsonl adapter2/{adapter.name} $(cd {BASE} && ls *.log)", shell=True).returncode
    status(f"packed pilot2_out.tgz exit {rc}, {(BASE / 'pilot2_out.tgz').stat().st_size / 1e6:.0f} MB")
    status("PILOT2 DONE")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
T0 = time.time()
try:
    main()
except Exception as exc:  # logged so the watcher sees it; the VM is still held for downloads
    status(f"ABORTED: {exc!r}"[:300])
status(f"elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("P2_HOLD_MIN", 30)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
