"""Notebook cell: regenerate the soft targets the released model was trained on, for the open dataset.

The training session (session_bos_train.py) built data/train/targets_bos_core.jsonl on the VM and did not keep it
(only results were packed). This cell repeats exactly its relabel step with the same code bundle (s1_bostrain.tgz:
same rows, labeller, target builder and calibration defaults, byte for byte) and packs the result:

  1. vLLM 0.31.0 (separate venv) serves google/gemma-4-26B-A4B-it in bf16; prompts start with <bos>
  2. pipeline/label.py labels the 90,400 core rows (one rerun fills rows that errored); stop if under 98%
  3. pipeline/targets.py rebuilds the soft targets; the fixed rows are appended unchanged
  4. reproducibility check: per-family teacher agreement against the training session's targets.log
  5. pack labels, targets and sha256 sums into /content/relabel_out.tgz

Expects /content/s1_bostrain.tgz and /content/bostrain_targets.log. Holds the VM until /content/downloaded.flag.

    colab exec -s <session> -f colab/session_relabel.py

R_* environment variables shrink everything for a local dry run (R_SERVER_CMD: tests/mock_vllm.py as the server).
"""

import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = os.environ.get("R_MODEL", "google/gemma-4-26B-A4B-it")
BASE = Path(os.environ.get("R_BASE", "/content"))
SERVER_CMD = os.environ.get("R_SERVER_CMD")
LABEL_LIMIT = os.environ.get("R_LABEL_LIMIT")
VLLM_VERSION = "0.31.0"  # the version the training session labelled with
W = BASE / "s1"
VENV = BASE / "vllm_env"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
REPORT = BASE / "relabel.json"
CORE = "data/train/bos100k_core_rows.jsonl"
FIXED = "data/train/bos100k_fixed_rows.jsonl"
OUT_CORE = "data/train/targets_bos_core.jsonl"
OUT_ALL = "data/train/targets_bos100k.jsonl"
report: dict = {}


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    report[key] = value
    REPORT.write_text(json.dumps(report, indent=1))


def run(args: list[str], log: str, timeout_min: float) -> int:
    try:
        return subprocess.run(args, cwd=W, stdout=open(BASE / log, "w"), stderr=subprocess.STDOUT,
                              timeout=timeout_min * 60).returncode
    except subprocess.TimeoutExpired:
        return -9


def vllm_env() -> bool:
    if SERVER_CMD or (VENV / "bin" / "python").exists():
        return True
    log = open(BASE / "vllm_install.log", "w")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv", "httpx", "tqdm"], check=True, stdout=log,
                       stderr=subprocess.STDOUT)
        subprocess.run(["uv", "venv", str(VENV), "--python", sys.executable], check=True, stdout=log,
                       stderr=subprocess.STDOUT)
        subprocess.run(["uv", "pip", "install", "--python", str(VENV / "bin" / "python"), f"vllm=={VLLM_VERSION}"],
                       check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        return True
    except Exception as exc:
        status(f"vLLM environment failed ({exc!r})"[:300])
        return False


def start_vllm():
    if not vllm_env():
        return None
    cmd = shlex.split(SERVER_CMD) if SERVER_CMD else [
        str(VENV / "bin" / "python"), "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", "8000",
        "--max-model-len", "8192", "--max-logprobs", "20", "--gpu-memory-utilization", "0.90", "--enforce-eager"]
    server = subprocess.Popen(cmd, stdout=open(BASE / "vllm.log", "a"), stderr=subprocess.STDOUT, start_new_session=True,
                              env={**os.environ, "PATH": f"{VENV / 'bin'}:{os.environ.get('PATH', '')}"})
    t0 = time.time()
    while time.time() - t0 < 900 and server.poll() is None:  # includes the first model load from disk
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
                if r.status == 200:
                    status(f"vLLM ready in {time.time() - t0:.0f}s")
                    return server
        except Exception:
            pass
        time.sleep(10)
    status(f"vLLM did not become healthy (exit {server.poll()})")
    stop_vllm(server)
    return None


def stop_vllm(server) -> None:
    if server is None:
        return
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def agreement_table(path: Path) -> dict[str, float]:
    """family -> teacher agreement with gold, from a targets.py log ('family  rows  agreement  conflicts ...')."""
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^(\S+)\s+(\d+)\s+([01]\.\d+)\s+\d+", line)
        if m:
            out[m.group(1)] = float(m.group(3))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_bostrain.tgz -C {W}", shell=True, check=True)
    if not SERVER_CMD:
        status(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                              capture_output=True, text=True).stdout.strip())
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.18.0", "httpx", "tqdm"], check=True)
        t0 = time.time()
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "tokenizer*"])
        status(f"model downloaded in {time.time() - t0:.0f}s")

    server = start_vllm()
    if server is None:
        status("STOPPED: no vLLM server")
        return
    args = [sys.executable, "pipeline/label.py", CORE, "--backend", "vllm", "--model", MODEL, "--workers", "64",
            "--out", str(BASE / "labels")] + (["--limit", LABEL_LIMIT] if LABEL_LIMIT else [])
    t0 = time.time()
    rc = run(args, "label.log", 45)
    if rc not in (0, -9):  # rows that errored are not written; one rerun fills them in
        rc = run(args, "label_retry.log", 20)
    stop_vllm(server)
    labels = sorted((BASE / "labels").glob("*.labels.jsonl"))
    n_labels = sum(1 for p in labels for _ in p.open())
    want = int(LABEL_LIMIT) if LABEL_LIMIT else sum(1 for _ in open(W / CORE))
    record("label", {"exit": rc, "labelled": n_labels, "wanted": want, "minutes": round((time.time() - t0) / 60, 1)})
    status(f"label: {n_labels}/{want} core rows labelled with <bos> in {(time.time() - t0) / 60:.1f} min (exit {rc})")
    if n_labels < 0.98 * want:
        status("STOPPED: under 98% labelled")
        return

    rc = run([sys.executable, "pipeline/targets.py", "--rows", CORE, "--labels", *map(str, labels), "--out", OUT_CORE],
             "targets.log", 20)
    tail = [line.strip() for line in open(BASE / "targets.log") if line.startswith(("total", "synthetic"))]
    status(f"targets: exit {rc}; {' | '.join(tail)}")
    if rc != 0 or not (W / OUT_CORE).exists():
        status("STOPPED: targets not built")
        return
    with open(W / OUT_ALL, "w") as out:
        out.write((W / OUT_CORE).read_text())
        out.write((W / FIXED).read_text())

    new, old = agreement_table(BASE / "targets.log"), agreement_table(BASE / "bostrain_targets.log")
    common = sorted(set(new) & set(old))
    diffs = sorted(((abs(new[f] - old[f]), f) for f in common), reverse=True)
    rows = {"training": sum(1 for _ in open(W / OUT_ALL)), "core": sum(1 for _ in open(W / OUT_CORE))}
    record("reproducibility", {"families_compared": len(common), "families_only_new": len(set(new) - set(old)),
                               "families_only_training": len(set(old) - set(new)),
                               "max_agreement_diff": round(diffs[0][0], 4) if diffs else None,
                               "worst": [(f, old[f], new[f]) for _, f in diffs[:5]],
                               "summary_training": [t for t in (BASE / "bostrain_targets.log").read_text().splitlines()
                                                    if t.startswith(("total", "synthetic"))],
                               "summary_new": tail, "rows": rows})
    status(f"reproducibility: {len(common)} families compared, largest agreement difference "
           f"{diffs[0][0] if diffs else float('nan'):.4f}; rows {rows}")
    sums = {p: sha256(W / p) for p in (OUT_CORE, OUT_ALL, CORE, FIXED)}
    record("sha256", sums)
    status("RELABEL DONE")


def pack() -> None:
    files = " ".join(p for p in (OUT_CORE, OUT_ALL) if (W / p).exists())
    rc = subprocess.run(f"cd {W} && tar czf {BASE}/relabel_out.tgz {files} -C {BASE} relabel.json labels "
                        f"$(cd {BASE} && ls *.log)", shell=True).returncode
    status(f"packed relabel_out.tgz exit {rc}, {(BASE / 'relabel_out.tgz').stat().st_size / 1e6:.0f} MB")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
T0 = time.time()
try:
    main()
except Exception as exc:
    status(f"ABORTED: {exc!r}"[:300])
try:
    pack()
except Exception as exc:
    status(f"pack failed: {exc!r}"[:200])
status(f"SESSION DONE, elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("R_HOLD_MIN", 10)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
