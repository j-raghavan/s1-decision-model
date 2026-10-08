"""Notebook cell for the v2 data session: generate synthetic task families, then label everything with the teacher.

Runs on one GPU session (G4, 96 GB) and keeps the kernel busy throughout, so
Colab cannot reclaim the VM (see COMPUTE_LOG.md). Expects /content/s1_data.tgz
with pipeline/, eval/jevbench/ and data/train/sni_v2.jsonl.

  1. serve openai/gpt-oss-120b with vLLM; generate FAMILIES task families
  2. stop it and delete its weights (VM disk is ~70 GB)
  3. serve google/gemma-4-26B-A4B-it; label sni_v2 + synthetic_v2 with label.py
  4. hold the VM until the Mac uploads /content/downloaded.flag (or 30 minutes)

Each server start aborts as soon as the server process dies, so a crash cannot
hold a paid GPU idle. Progress goes to /content/data_session.log.

    colab exec -s <session> -f colab/session_data.py
"""

import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

FAMILIES, INSTANCES = 900, 30
START_INDEX, SEED, WORKERS, GEN_MINUTES = 1000, 20261007, 128, 50  # scale-up batch: new ids and seeds, capped time
SYN_NAME = "synthetic_v3"
LABEL_FILES = [f"data/train/{SYN_NAME}.jsonl"]  # SNI was labeled in the pilot
GEN_MODEL, TEACHER = "openai/gpt-oss-120b", "google/gemma-4-26B-A4B-it"
W = Path("/content/s1")
LOG = Path("/content/data_session.log")
FLAG = Path("/content/downloaded.flag")
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def serve(model: str, extra: list[str], name: str) -> subprocess.Popen:
    out = open(f"/content/vllm_{name}.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", "8000",
                             "--gpu-memory-utilization", "0.92", "--enforce-eager", *extra], stdout=out, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if proc.poll() is not None:
            raise SystemExit(f"{name} server exited with code {proc.returncode}; see /content/vllm_{name}.log")
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
                if r.status == 200:
                    log(f"{name} server ready in {time.time() - t0:.0f}s")
                    return proc
        except Exception:
            pass
        time.sleep(10)
    proc.terminate()
    raise SystemExit(f"{name} server not healthy after 30 min")


def stop(proc: subprocess.Popen, model: str) -> None:
    proc.terminate()
    proc.wait(timeout=120)
    time.sleep(5)
    # This huggingface_hub keeps file contents in a shared hub/blobs/ directory, not in the per-model folder, so
    # deleting models--<name> frees nothing (that left gpt-oss's 61 GB behind). Clear the whole cache; the next
    # model is downloaded fresh either way.
    for d in [HF_CACHE, HF_CACHE.parent / "xet"]:
        shutil.rmtree(d, ignore_errors=True)
    log(f"stopped and removed {model}; {free_gb():.0f} GB free")


def free_gb() -> float:
    return shutil.disk_usage("/content").free / 1e9


def run(args: list[str], name: str) -> None:
    with open(f"/content/{name}.log", "w") as out:
        rc = subprocess.run([sys.executable, *args], cwd=W, stdout=out, stderr=subprocess.STDOUT).returncode
    tail = Path(f"/content/{name}.log").read_text().splitlines()[-2:]
    log(f"{name} exit {rc}: {' | '.join(tail)[:300]}")


FLAG.unlink(missing_ok=True)
LOG.write_text("")
if not (W / "pipeline").exists():
    subprocess.run(f"mkdir -p {W} && tar xzf /content/s1_data.tgz -C {W}", shell=True, check=True)
log(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip())
try:
    import vllm  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "httpx", "tqdm", "numpy"], check=True)
log("vllm ready to import")

SYN = W / "data" / "train" / f"{SYN_NAME}.jsonl"
done_families = len({l.split('"family": "')[1].split('"')[0] for l in SYN.open()}) if SYN.exists() else 0
try:
    if done_families >= 0.5 * FAMILIES:
        log(f"generation already done ({done_families} families); skipping")
    else:
        gen = serve(GEN_MODEL, ["--max-model-len", "16384"], "gptoss")
        run(["pipeline/generate_synthetic.py", "--model", GEN_MODEL, "--families", str(FAMILIES), "--instances", str(INSTANCES),
             "--workers", str(WORKERS), "--start-index", str(START_INDEX), "--seed", str(SEED),
             "--max-minutes", str(GEN_MINUTES), "--out", f"data/train/{SYN_NAME}.jsonl"], "generate")
        stop(gen, GEN_MODEL)

    log(f"{free_gb():.0f} GB free before loading the teacher")
    teacher = serve(TEACHER, ["--max-model-len", "8192", "--max-logprobs", "20"], "gemma")
    for rows in LABEL_FILES:
        run(["pipeline/label.py", rows, "--backend", "vllm", "--model", TEACHER, "--workers", "64", "--out", "data/labels"],
            "label_" + Path(rows).stem)
    stop(teacher, TEACHER)
    log("DATA SESSION DONE")
except SystemExit as exc:
    log(f"ABORTED: {exc}")

t0 = time.time()
while not FLAG.exists() and time.time() - t0 < 1800:
    time.sleep(15)
log("download confirmed" if FLAG.exists() else "no download confirmation after 30 min; releasing")
