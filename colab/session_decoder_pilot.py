"""Notebook cell: Gemma 4 26B-A4B decision-model LoRA pilot on one GPU, evaluated in the same session.

Stages, each logged to /content/session.log:
  1. install pinned transformers/peft, download the model
  2. GPU self-check: batched vs single scoring in bf16
  3. zero-shot eval of the untuned model through the same readout (JevBench + custom); aborts before any
     training if JevBench accuracy is far below the vLLM/Ollama teacher measurement (0.770), since that
     would mean the readout is wrong
  4. time-capped LoRA training on data/train/targets_pilot.jsonl, best adapter chosen on held-out families
  5. eval of the tuned model on both suites; results and adapter packed into /content/pilot_out.tgz

Expects /content/s1_pilot.tgz (s1/, eval/jevbench/, data/eval/, data/train/targets_pilot.jsonl). Keeps the
kernel busy throughout and holds the VM until /content/downloaded.flag appears.

    colab exec -s <session> -f colab/session_decoder_pilot.py

PILOT_* environment variables override the model, base folder and sizes, only for a local dry run with a
small model (PILOT_MODEL=google/gemma-4-E2B-it PILOT_BASE=/tmp/pilot PILOT_LIMIT=24 PILOT_STEPS=12 ...).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = os.environ.get("PILOT_MODEL", "google/gemma-4-26B-A4B-it")
TRAIN_MINUTES = float(os.environ.get("PILOT_MINUTES", 30))
STEPS, BATCH = os.environ.get("PILOT_STEPS", "800"), os.environ.get("PILOT_BATCH", "16")
VAL_LIMIT = os.environ.get("PILOT_VAL_LIMIT", "600")
LIMIT = ["--limit", os.environ["PILOT_LIMIT"]] if os.environ.get("PILOT_LIMIT") else []
MIN_ZEROSHOT = float(os.environ.get("PILOT_MIN_ZEROSHOT", 0.70))
DEVICE = os.environ.get("PILOT_DEVICE", "cuda")
BASE = Path(os.environ.get("PILOT_BASE", "/content"))
W = BASE / "s1"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
HOLD_MIN = float(os.environ.get("PILOT_HOLD_MIN", 30))
T_START = time.time()


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


def accuracy(run_id: str) -> float:
    """Mean per-slice accuracy over yes/no and choice slices (the headline the teachers were compared on)."""
    by: dict[str, list[float]] = {}
    for line in open(W / f"results/raw/jevbench/{run_id}/predictions.jsonl"):
        r = json.loads(line)
        if r["task_type"] == "score" or r.get("error"):
            continue
        by.setdefault(r["slice"], []).append(float(str(r["pred"]).lower() == str(r["gold"]).lower()))
    return sum(sum(v) / len(v) for v in by.values()) / max(1, len(by))


def main() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_pilot.tgz -C {W}", shell=True, check=True)
    if DEVICE == "cuda":
        status(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                              capture_output=True, text=True).stdout.strip())
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.18.0", "peft==0.21.2",
                        "accelerate==1.15.0"], check=True)
        # Colab preinstalls torchao 0.10, which PEFT rejects when it builds LoRA layers; nothing here uses it.
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
        import torch
        status(f"torch {torch.__version__}, cuda {torch.version.cuda}, capability {torch.cuda.get_device_capability()}")

    t0 = time.time()
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "tokenizer*"])
    status(f"model downloaded in {time.time() - t0:.0f}s")

    check = (
        "import sys, json, torch; sys.path.insert(0, 'eval/jevbench')\n"
        "from s1.decoder import load, Readout, option_log_probs\n"
        "from paths import SUITES, read_jsonl\n"
        f"tok, m = load('{MODEL}', '{DEVICE}', torch.bfloat16); m.eval(); r = Readout(tok)\n"
        "cases = read_jsonl(SUITES['jevbench']['cases'])[::100][:16]\n"
        "with torch.no_grad():\n"
        f"    b = option_log_probs(m, tok, r, cases, '{DEVICE}')\n"
        f"    s = [option_log_probs(m, tok, r, [c], '{DEVICE}')[0] for c in cases]\n"
        "d = max(abs(x[k] - y[k]) for x, y in zip(b, s) for k in x)\n"
        "agree = sum(max(x, key=x.get) == max(y, key=y.get) for x, y in zip(b, s))\n"
        "print(f'SELFCHECK max_dlogp={d:.4f} argmax_agree={agree}/{len(cases)} mem_gb={torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0:.1f}')\n"
    )
    rc = run([sys.executable, "-c", check], "selfcheck.log", 20)
    line = next((l for l in open(BASE / "selfcheck.log") if l.startswith("SELFCHECK")), f"SELFCHECK failed rc={rc}")
    status(line.strip())
    if rc != 0:
        status(f"ABORTED: self-check failed; see {BASE}/selfcheck.log")
        return

    for suite in ("jevbench", "custom"):
        done = W / f"results/raw/{suite}/g26-hf-zeroshot/predictions.jsonl"
        if done.exists() and not LIMIT and sum(1 for _ in done.open()) == sum(1 for _ in open(W / f"data/eval/{'jevbench_subset' if suite == 'jevbench' else 'custom_v0'}.jsonl")):
            status(f"zero-shot {suite} already complete in this session; reused")
            continue
        rc = run([sys.executable, "eval/jevbench/run_decoder.py", "--model", MODEL, "--run-id", "g26-hf-zeroshot",
                  "--suite", suite, "--batch", BATCH, *LIMIT], f"zeroshot_{suite}.log", 25)
        status(f"zero-shot {suite} exit {rc}")
    zs = accuracy("g26-hf-zeroshot")
    status(f"zero-shot JevBench accuracy {zs:.3f} (teacher measurement 0.770)")
    if zs < MIN_ZEROSHOT:
        status("ABORTED: zero-shot readout is far below the teacher measurement; not training")
        return

    rc = run([sys.executable, "-m", "s1.train_decoder", "--model", MODEL, "--targets", "data/train/targets_pilot.jsonl",
              "--out", str(BASE / "adapter"), "--max-steps", STEPS, "--batch", BATCH, "--accum", "2", "--lr", "1e-4",
              "--rank", "64", "--eval-every", "100", "--val-limit", VAL_LIMIT, "--max-minutes", str(TRAIN_MINUTES)],
             "train.log", TRAIN_MINUTES + 20)
    for line in open(BASE / "train.log"):
        if line.startswith("{") and any(k in line for k in ("sec_per_step", "unseen_loss", "done", "device")):
            status("train " + line.strip())
    status(f"train exit {rc}")
    adapter = BASE / "adapter" / ("best" if (BASE / "adapter" / "best").exists() else "last")
    if not adapter.exists():
        status(f"ABORTED: no adapter saved; see {BASE}/train.log")
        return
    status(f"evaluating {adapter}")

    for suite in ("jevbench", "custom"):
        rc = run([sys.executable, "eval/jevbench/run_decoder.py", "--model", MODEL, "--adapter", str(adapter),
                  "--run-id", "g26-pilot", "--suite", suite, "--batch", BATCH, *LIMIT], f"pilot_{suite}.log", 25)
        status(f"pilot eval {suite} exit {rc}")
    status(f"pilot JevBench accuracy {accuracy('g26-pilot'):.3f} vs zero-shot {zs:.3f}")
    rc = subprocess.run(f"cd {W} && tar czf {BASE}/pilot_out.tgz results/raw/*/g26-* -C {BASE} adapter/log.jsonl "
                        f"adapter/{adapter.name} train.log", shell=True).returncode
    status(f"packed {BASE}/pilot_out.tgz exit {rc}, {(BASE / 'pilot_out.tgz').stat().st_size / 1e6:.0f} MB")
    status("PILOT DONE")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
try:
    main()
except Exception as exc:  # logged so the watcher sees it, then the VM is still held for inspection
    status(f"ABORTED: {exc!r}"[:300])
status(f"elapsed {(time.time() - T_START) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < HOLD_MIN * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else f"no download confirmation after {HOLD_MIN:.0f} min; releasing")
