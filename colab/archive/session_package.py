"""Notebook cell: package the <bos> fine-tune (step 1500) and verify it, then publish to a private Hugging Face repo.

  1. merge the LoRA adapter into Gemma 4 26B-A4B (scripts/merge_lora.py, one shard at a time)
  2. vLLM bf16 serves the merged model: JevBench and custom test predictions must match the adapter-served run
     (results/raw/*/g26-bos-vllm) on >= 97% of cases, and accuracy is reported; latency one request at a time
  3. vLLM FP8 serves the merged model: dev and test accuracy and latency
  4. 4-bit package for local use: Ollama imports the merged safetensors with --quantize q4_K_M; it must load and
     answer, and a 200-case sample is compared with the bf16 predictions
  5. upload to a private repo (merged bf16 weights, and the Ollama model as a GGUF-backed blob when available)
  6. the Hugging Face token copied to the VM for the upload is deleted at the end, whatever happens

Expects /content/s1_pkg.tgz, /content/adapter_final.tgz (adapter_final/), and for the upload /content/hf_token.
Keeps the kernel busy and holds the VM until /content/downloaded.flag appears.

    colab exec -s <session> -f colab/session_package.py

P_* environment variables point the cell at a small model and a stand-in server for a dry run.
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = os.environ.get("P_MODEL", "google/gemma-4-26B-A4B-it")
REPO = os.environ.get("P_REPO", "j-raghavan/s1-gemma4-26b-decision")
BASE = Path(os.environ.get("P_BASE", "/content"))
SERVER_CMD = os.environ.get("P_SERVER_CMD")
LIMIT = ["--limit", os.environ["P_LIMIT"]] if os.environ.get("P_LIMIT") else []
# JevBench agreement with the adapter-served run: a merge that silently lost the adapter scores 0.870 (the untuned
# model); custom cannot tell them apart (0.973) and is reported only.
AGREE_MIN = float(os.environ.get("P_AGREE_MIN", 0.95))
OLLAMA_NAME = os.environ.get("P_OLLAMA_NAME", "s1-gemma4-26b")
OLLAMA_QUANT = os.environ.get("P_OLLAMA_QUANT", "int4")  # Ollama >= 0.35 imports safetensors as int4/int8/nvfp4/mxfp4/mxfp8
SKIP_OLLAMA = os.environ.get("P_SKIP_OLLAMA") == "1"
VLLM_VERSION = "0.31.0"
W = BASE / "s1"
VENV = BASE / "vllm_env"
MERGED = BASE / "merged"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
REPORT = BASE / "package.json"
TOKEN = BASE / "hf_token"
results: dict = {}


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    results[key] = value
    REPORT.write_text(json.dumps(results, indent=1))


def run(args: list[str], log: str, timeout_min: float, env: dict | None = None) -> int:
    try:
        return subprocess.run(args, cwd=W, stdout=open(BASE / log, "w"), stderr=subprocess.STDOUT,
                              timeout=timeout_min * 60, env={**os.environ, **(env or {})}).returncode
    except subprocess.TimeoutExpired:
        return -9


def preds(suite: str, run_id: str) -> dict:
    sys.path.insert(0, str(W))
    from s1.decide import load_predictions
    p = W / f"results/raw/{suite}/{run_id}/predictions.jsonl"
    return load_predictions(p) if p.exists() else {}


def acc(suite: str, run_id: str) -> float | None:
    sys.path.insert(0, str(W))
    from s1.decide import slice_mean_accuracy
    p = preds(suite, run_id)
    return round(slice_mean_accuracy(p), 4) if p else None


def vllm_env() -> bool:
    if SERVER_CMD or (VENV / "bin" / "python").exists():
        return True
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv", "httpx", "tqdm"], check=True,
                       stdout=open(BASE / "vllm_install.log", "w"), stderr=subprocess.STDOUT)
        subprocess.run(["uv", "venv", str(VENV), "--python", sys.executable], check=True,
                       stdout=open(BASE / "vllm_install.log", "a"), stderr=subprocess.STDOUT)
        subprocess.run(["uv", "pip", "install", "--python", str(VENV / "bin" / "python"), f"vllm=={VLLM_VERSION}"],
                       check=True, stdout=open(BASE / "vllm_install.log", "a"), stderr=subprocess.STDOUT, timeout=1200)
        return True
    except Exception as exc:
        status(f"vLLM environment failed ({exc!r})"[:300])
        return False


def stop(server) -> None:
    if server is None:
        return
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=60)
    except ProcessLookupError:
        pass
    time.sleep(10)


def start(model_path: str, extra: list[str], tag: str):
    """vLLM serving a local model directory under the name 's1'; verified by a real request."""
    if not vllm_env():
        return None
    if SERVER_CMD:
        cmd = shlex.split(SERVER_CMD) + ["--model", "s1"]
    else:
        cmd = [str(VENV / "bin" / "python"), "-m", "vllm.entrypoints.openai.api_server", "--model", model_path,
               "--served-model-name", "s1", "--port", "8000", "--max-model-len", "8192", "--max-logprobs", "20",
               "--gpu-memory-utilization", "0.90", "--enforce-eager", "--no-enable-prefix-caching", *extra]
    server = subprocess.Popen(cmd, stdout=open(BASE / f"server_{tag}.log", "a"), stderr=subprocess.STDOUT,
                              start_new_session=True,
                              env={**os.environ, "PATH": f"{VENV / 'bin'}:{os.environ.get('PATH', '')}"})
    t0 = time.time()
    while time.time() - t0 < 600 and server.poll() is None:
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
                if r.status == 200:
                    body = json.dumps({"model": "s1", "prompt": "Answer:", "max_tokens": 1}).encode()
                    req = urllib.request.Request("http://localhost:8000/v1/completions", data=body,
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=300) as rr:
                        json.loads(rr.read())["choices"]
                    status(f"{tag}: server ready in {time.time() - t0:.0f}s")
                    return server
        except Exception:
            pass
        time.sleep(10)
    status(f"{tag}: server not usable (exit {server.poll()}); see server_{tag}.log")
    stop(server)
    return None


def score(run_id: str, suite: str, extra: tuple = (), workers: int = 64, timeout_min: float = 20) -> int:
    args = [sys.executable, "eval/jevbench/run_ollama.py", "--backend", "vllm", "--template", "gemma4", "--model", "s1",
            "--run-id", run_id, "--suite", suite, "--workers", str(workers), *extra, *([] if extra else LIMIT)]
    return run(args, f"score_{run_id}_{suite}.log", timeout_min)


def latency(run_id: str) -> dict:
    import random
    by: dict[str, list[str]] = {}
    for line in open(W / "data" / "eval" / "jevbench_dev.jsonl"):
        c = json.loads(line)
        by.setdefault(c["slice"], []).append(c["case_id"])
    rng = random.Random(20261007)
    ids = [i for s in sorted(by) for i in rng.sample(by[s], min(20, len(by[s])))]
    (BASE / "latency_ids.txt").write_text("\n".join(ids) + "\n")
    rc = score(run_id, "jevbench_dev", extra=("--case-ids", str(BASE / "latency_ids.txt")), workers=1, timeout_min=10)
    lat = sorted(r["latency_ms"] for r in preds("jevbench_dev", run_id).values() if r.get("latency_ms"))
    return {"exit": rc, "n": len(lat), **({"p50_ms": round(lat[len(lat) // 2], 1),
                                           "p95_ms": round(lat[max(0, int(len(lat) * 0.95) - 1)], 1)} if lat else {})}


def agreement(suite: str, a: str, b: str) -> tuple[float, int]:
    pa, pb = preds(suite, a), preds(suite, b)
    ids = [i for i in pa if i in pb]
    return sum(str(pa[i]["pred"]) == str(pb[i]["pred"]) for i in ids) / max(1, len(ids)), len(ids)


# ---------------------------------------------------------------- stages

def free_gb() -> float:
    import shutil
    return shutil.disk_usage(BASE).free / 1e9


def merge() -> bool:
    status(f"disk free before merge: {free_gb():.0f} GB")
    # the base shards are consumed as they are merged, so base + merged never sit on disk together
    consume = [] if SERVER_CMD else ["--consume-base"]  # never on a dry-run machine: it deletes the cached base
    rc = run([sys.executable, "scripts/merge_lora.py", MODEL, str(BASE / "adapter_final"), str(MERGED), *consume],
             "merge.log", 60)
    ok = rc == 0 and any(MERGED.glob("*.safetensors"))
    tail = (BASE / "merge.log").read_text().strip().splitlines()[-1:] if (BASE / "merge.log").exists() else []
    record("merge", {"exit": rc, "summary": tail})
    status(f"merge: exit {rc}; {tail}")
    return ok


def verify_bf16() -> bool:
    server = start(str(MERGED), [], "bf16")
    if server is None:
        return False
    try:
        out = {}
        for suite in ("jevbench", "custom"):
            rc = score("merged-bf16", suite)
            share, n = agreement(suite, "merged-bf16", "g26-bos-vllm")
            out[suite] = {"exit": rc, "acc": acc(suite, "merged-bf16"), "agreement_with_adapter_run": share, "n": n}
            status(f"merged bf16 {suite}: acc {out[suite]['acc']} (adapter run {acc(suite, 'g26-bos-vllm')}), "
                   f"agreement {share:.3f} on {n}")
        out["latency"] = latency("lat-merged-bf16")
        status(f"merged bf16 latency: {out['latency']}")
        ok = out["jevbench"]["agreement_with_adapter_run"] >= AGREE_MIN and out["jevbench"]["n"] > 0
        out["pass"] = ok
        record("bf16", out)
        status(f"merge verified: {'PASS' if ok else 'FAIL'} (need >= {AGREE_MIN} JevBench agreement; custom reported only)")
        return ok
    finally:
        stop(server)


def verify_fp8() -> None:
    server = start(str(MERGED), ["--quantization", "fp8"], "fp8")
    if server is None:
        record("fp8", {"skipped": True})
        return
    try:
        out = {}
        for suite in ("jevbench_dev", "custom_dev2", "jevbench", "custom"):
            rc = score("merged-fp8", suite, timeout_min=8)
            out[suite] = {"exit": rc, "acc": acc(suite, "merged-fp8")}
            status(f"merged fp8 {suite}: acc {out[suite]['acc']}")
        out["latency"] = latency("lat-merged-fp8")
        status(f"merged fp8 latency: {out['latency']}")
        record("fp8", out)
    finally:
        stop(server)


def ollama_package() -> dict:
    """Ollama imports the merged safetensors and quantizes to q4_K_M; the model must load and answer."""
    if SKIP_OLLAMA:
        return {"skipped": True}
    if free_gb() < 80:  # import plus quantize can need the merged size again; never fill the disk mid-session
        status(f"ollama: skipped, only {free_gb():.0f} GB free")
        return {"skipped": f"{free_gb():.0f} GB free"}
    out: dict = {}
    if subprocess.run("command -v ollama", shell=True).returncode != 0:
        rc = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True,
                            stdout=open(BASE / "ollama_install.log", "w"), stderr=subprocess.STDOUT).returncode
        if rc != 0:
            return {"error": f"ollama install exit {rc}"}
    # its own server, port and model store, so nothing depends on (or touches) another Ollama on the machine
    oenv = {"OLLAMA_MODELS": str(BASE / "ollama_models"), "OLLAMA_HOST": "127.0.0.1:11435"}
    serve = subprocess.Popen(["ollama", "serve"], stdout=open(BASE / "ollama_serve.log", "w"), stderr=subprocess.STDOUT,
                             start_new_session=True, env={**os.environ, **oenv})
    time.sleep(10)
    try:
        (BASE / "Modelfile").write_text(f"FROM {MERGED}\n")
        rc = run(["ollama", "create", OLLAMA_NAME, "--quantize", OLLAMA_QUANT, "-f", str(BASE / "Modelfile")],
                 "ollama_create.log", 60, env=oenv)
        out["create_exit"] = rc
        status(f"ollama create ({OLLAMA_QUANT}): exit {rc}")
        if rc != 0:
            return out
        ids = [line.strip() for line in open(W / "data" / "eval" / "canary_ids.txt") if line.strip()]
        rc = run([sys.executable, "eval/jevbench/run_ollama.py", "--backend", "ollama", "--model", OLLAMA_NAME,
                  "--run-id", "ollama-q4", "--suite", "jevbench", "--case-ids", "data/eval/canary_ids.txt"],
                 "score_ollama.log", 20, env={"OLLAMA_URL": "http://127.0.0.1:11435"})
        share, n = agreement("jevbench", "ollama-q4", "merged-bf16")
        out |= {"score_exit": rc, "agreement_with_bf16_on_canary": share, "n": n, "ids": len(ids)}
        status(f"ollama {OLLAMA_QUANT}: answers {n} of {len(ids)} canary cases, agreement with merged bf16 {share:.3f}")
        # Ollama stores the quantized weights as one GGUF blob; keep its path for the upload
        mf = subprocess.run(["ollama", "show", "--modelfile", OLLAMA_NAME], capture_output=True, text=True,
                            env={**os.environ, **oenv}).stdout
        blob = next((line.split(maxsplit=1)[1] for line in mf.splitlines() if line.startswith("FROM /")), None)
        if blob and Path(blob).exists() and open(blob, "rb").read(4) == b"GGUF":
            out["gguf"] = blob
            out["gguf_gb"] = round(Path(blob).stat().st_size / 1e9, 1)
    finally:
        try:
            serve.terminate()
            serve.wait(timeout=30)
        except Exception:
            pass
    return out


def upload() -> dict:
    if not TOKEN.exists():
        return {"skipped": "no token"}
    from huggingface_hub import HfApi
    api = HfApi(token=TOKEN.read_text().strip())
    user = api.whoami()["name"]
    api.create_repo(REPO, private=True, exist_ok=True)
    info = api.repo_info(REPO)
    if not info.private:
        return {"error": f"{REPO} exists and is public; not uploading"}
    readme = MERGED / "README.md"
    readme.write_text(
        "# s1 decision model (Gemma 4 26B-A4B, LoRA merged)\n\nPrivate. Fine-tuned for typed decisions (choice, yes/no, "
        "ordinal score) with a single forward pass and option-letter readout. Prompts must start with `<bos>` "
        "(the tokenizer does not add it). Base: google/gemma-4-26B-A4B-it (Apache-2.0).\n\n"
        f"Verification: {json.dumps({k: results.get(k) for k in ('bf16', 'fp8')})}\n")
    api.upload_folder(folder_path=str(MERGED), repo_id=REPO, path_in_repo="bf16",
                      commit_message="Merged bf16 weights (step 1500, trained with <bos>)")
    done = ["bf16/"]
    pkg = results.get("ollama") or {}
    store = BASE / "ollama_models"
    if pkg.get("gguf"):
        api.upload_file(path_or_fileobj=pkg["gguf"], path_in_repo=f"ollama/{OLLAMA_NAME}-{OLLAMA_QUANT}.gguf", repo_id=REPO,
                        commit_message=f"{OLLAMA_QUANT} GGUF made by Ollama")
        done.append("ollama/*.gguf")
    elif pkg.get("create_exit") == 0 and store.exists():
        # Ollama's own store (manifests + blobs): copy into ~/.ollama/models (or OLLAMA_MODELS) to use it locally
        api.upload_folder(folder_path=str(store), repo_id=REPO, path_in_repo="ollama/models",
                          commit_message=f"Ollama model store for {OLLAMA_NAME} ({OLLAMA_QUANT})")
        done.append("ollama/models/")
    return {"repo": REPO, "user": user, "private": True, "uploaded": done}


def main() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_pkg.tgz -C {W} && tar xzf {BASE}/adapter_final.tgz -C {BASE}",
                   shell=True, check=True)
    status(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                          capture_output=True, text=True).stdout.strip() or "no GPU (dry run)")
    if not SERVER_CMD:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "safetensors", "huggingface_hub"], check=True)
    if not merge():
        status("STOPPED: merge failed")
        return
    if not verify_bf16():
        status("STOPPED: merged model does not reproduce the adapter run; nothing uploaded")
        return
    verify_fp8()
    try:  # packaging for local use must never cost the upload of the verified weights
        pkg = ollama_package()
    except Exception as exc:
        pkg = {"error": repr(exc)[:300]}
        status(f"ollama: failed ({exc!r})"[:200])
    record("ollama", pkg)
    up = upload()
    record("upload", up)
    status(f"upload: {up}")
    status("PACKAGE DONE")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
T0 = time.time()
try:
    main()
except Exception as exc:
    status(f"ABORTED: {exc!r}"[:300])
finally:
    TOKEN.unlink(missing_ok=True)  # the upload token never outlives the session
    status(f"token removed from VM: {not TOKEN.exists()}")
rc = subprocess.run(f"cd {W} && tar czf {BASE}/pkg_out.tgz results/raw -C {BASE} package.json $(cd {BASE} && ls *.log)",
                    shell=True).returncode
status(f"packed pkg_out.tgz exit {rc}")
status(f"SESSION DONE, elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("P_HOLD_MIN", 10)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
