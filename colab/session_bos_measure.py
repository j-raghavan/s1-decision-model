"""Notebook cell: re-measure with <bos> (the HF/vLLM paths ran without it), plus latency and FP8 serving.

vLLM only (0.31.0 in its own environment, as in session s1-fullA); no training.

  1. bf16 server with the pilot-2 and option-A (step 500) adapters; every adapter answers a probe
  2. <bos> check: vLLM's own tokenizer must see exactly one <bos> at the start of a rendered prompt
  3. canary: pilot 2 scored WITHOUT <bos> on 200 fixed JevBench cases must reproduce its saved predictions
     (>= 95%), proving server and adapter are right before any new number counts
  4. untuned model with <bos> on jevbench_dev, custom_dev2, jevbench and custom (the true baseline)
  5. pilot 2 and option A with <bos> on the two dev splits (test sets stay untouched for adapters)
  6. latency: one request at a time on 200 jevbench_dev cases, untuned and option A
  7. FP8 server (untuned): dev splits and latency; skipped cleanly if vLLM rejects FP8 for this model
  8. results packed into /content/bos_out.tgz

Expects /content/s1_bos.tgz and /content/adapters.tgz (adapter_pilot2/, adapter_A500/). Keeps the kernel busy
and holds the VM until /content/downloaded.flag appears.

    colab exec -s <session> -f colab/session_bos_measure.py

B_* environment variables point the cell at a local stand-in server for a dry run (tests/mock_vllm.py).
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

MODEL = os.environ.get("B_MODEL", "google/gemma-4-26B-A4B-it")
BASE = Path(os.environ.get("B_BASE", "/content"))
SERVER_CMD = os.environ.get("B_SERVER_CMD")  # dry run: command for a stand-in server instead of vLLM
CANARY_MIN = float(os.environ.get("B_CANARY_MIN", 0.95))
LIMIT = ["--limit", os.environ["B_LIMIT"]] if os.environ.get("B_LIMIT") else []
VLLM_VERSION = "0.31.0"
W = BASE / "s1"
VENV = BASE / "vllm_env"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
REPORT = BASE / "measurements.json"
ADAPTERS = {"pilot2": BASE / "adapter_pilot2", "A500": BASE / "adapter_A500"}
results: dict = {}


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    results[key] = value
    REPORT.write_text(json.dumps(results, indent=1))


def run(args: list[str], log: str, timeout_min: float) -> int:
    try:
        return subprocess.run(args, cwd=W, stdout=open(BASE / log, "w"), stderr=subprocess.STDOUT,
                              timeout=timeout_min * 60).returncode
    except subprocess.TimeoutExpired:
        return -9


def preds(suite: str, run_id: str) -> dict:
    sys.path.insert(0, str(W))
    from s1.decide import load_predictions
    p = W / f"results/raw/{suite}/{run_id}/predictions.jsonl"
    return load_predictions(p) if p.exists() else {}


def accuracy(suite: str, run_id: str) -> float | None:
    sys.path.insert(0, str(W))
    from s1.decide import slice_mean_accuracy
    p = preds(suite, run_id)
    return round(slice_mean_accuracy(p), 4) if p else None


def post(path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(f"http://localhost:8000{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------- server

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


def probe(name: str) -> bool:
    try:
        r = post("/v1/completions", {"model": name, "prompt": "Answer:", "max_tokens": 1, "temperature": 0})
        return bool(r["choices"])
    except Exception:
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


def start(loras: dict[str, Path], extra: list[str], tag: str):
    if not vllm_env():
        return None
    if SERVER_CMD:
        cmd = shlex.split(SERVER_CMD) + ["--lora-names", ",".join(loras)]
    else:
        cmd = [str(VENV / "bin" / "python"), "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", "8000",
               "--max-model-len", "8192", "--max-logprobs", "20", "--gpu-memory-utilization", "0.90", "--enforce-eager",
               "--no-enable-prefix-caching",  # latency must not be served from prompts scored earlier
               *extra]
        if loras:
            cmd += ["--enable-lora", "--max-lora-rank", "64", "--max-loras", "1", "--max-cpu-loras", str(len(loras)),
                    "--lora-modules", *(f"{n}={p}" for n, p in loras.items())]
    server = subprocess.Popen(cmd, stdout=open(BASE / f"server_{tag}.log", "a"), stderr=subprocess.STDOUT,
                              start_new_session=True,
                                  env={**os.environ, "PATH": f"{VENV / 'bin'}:{os.environ.get('PATH', '')}"})
    t0 = time.time()
    while time.time() - t0 < 600 and server.poll() is None:
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
                if r.status == 200:
                    failed = [n for n in [MODEL, *loras] if not probe(n)]
                    if failed:
                        status(f"{tag}: up but not answering: {failed}")
                        break
                    status(f"{tag}: server ready in {time.time() - t0:.0f}s, {len(loras)} adapters answer")
                    return server
        except Exception:
            pass
        time.sleep(10)
    status(f"{tag}: server not usable (exit {server.poll()}); see server_{tag}.log")
    stop(server)
    return None


def score(model: str, run_id: str, suite: str, bos: str = "auto", extra: tuple = (), workers: int = 64,
          timeout_min: float = 25) -> int:
    """run_ollama.py through the server; error rows are removed and retried once, unless the run timed out."""
    args = [sys.executable, "eval/jevbench/run_ollama.py", "--backend", "vllm", "--template", "gemma4", "--model", model,
            "--run-id", run_id, "--suite", suite, "--workers", str(workers), "--bos", bos, *extra, *([] if extra else LIMIT)]
    rc = run(args, f"score_{run_id}_{suite}.log", timeout_min)
    p = W / f"results/raw/{suite}/{run_id}/predictions.jsonl"
    if rc != -9 and p.exists():  # a timeout means a slow server: retrying would only double the wait
        lines = p.read_text().splitlines(keepends=True)

        def answered(line: str) -> bool:
            try:
                return json.loads(line).get("error") is None and line.endswith("\n")
            except json.JSONDecodeError:
                return False
        good = [line for line in lines if answered(line)]
        if len(good) < len(lines):
            p.write_text("".join(good))
            rc = run(args, f"score_{run_id}_{suite}_retry.log", timeout_min)
    return rc


# ---------------------------------------------------------------- checks

def bos_check() -> bool:
    """vLLM's own tokenizer must see exactly one <bos> at the start of a rendered Gemma prompt."""
    sys.path[:0] = [str(W / "eval" / "jevbench")]
    from paths import SUITES, read_jsonl
    from run_ollama import labels_for, options_for, render
    case = read_jsonl(SUITES["jevbench_dev"]["cases"])[0]
    opts = options_for(case["question"])
    prompt = render(case, labels_for(len(opts)), opts, "gemma4")
    with_bos = post("/tokenize", {"model": MODEL, "prompt": "<bos>" + prompt})["tokens"]
    without = post("/tokenize", {"model": MODEL, "prompt": prompt})["tokens"]
    ok = with_bos[0] == 2 and with_bos.count(2) == 1 and without[0] != 2 and len(with_bos) == len(without) + 1
    record("bos_check", {"pass": ok, "with_bos_first": with_bos[:3], "without_first": without[:3]})
    status(f"<bos> check: with {with_bos[:3]} vs without {without[:3]} -> {'PASS' if ok else 'FAIL'}")
    return ok


def canary() -> bool:
    sys.path.insert(0, str(W))
    from s1.decide import agreement
    rc = score("pilot2", "canary-pilot2-nobos", "jevbench", bos="none", extra=("--case-ids", "data/eval/canary_ids.txt"))
    share, n = agreement(preds("jevbench", "canary-pilot2-nobos"), preds("jevbench", "g26-pilot2-vllm"))
    want = sum(1 for _ in open(W / "data" / "eval" / "canary_ids.txt"))
    ok = rc == 0 and n >= 0.98 * want and share >= CANARY_MIN
    record("canary", {"agreement": share, "cases": n, "expected": want, "required": CANARY_MIN, "pass": ok})
    status(f"canary (pilot 2 without <bos> vs its saved run): {share:.3f} on {n}/{want} -> {'PASS' if ok else 'FAIL'}")
    return ok


def latency(model: str, run_id: str, timeout_min: float = 10) -> dict:
    """One request at a time on a seeded sample of jevbench_dev (up to 20 cases per slice): per-decision latency as a
    client sees it (HTTP round trip, prefill, one decode step; two calls for two-digit option codes). Measured with
    --enforce-eager (no CUDA graphs), so a production server would be somewhat faster."""
    import random
    by: dict[str, list[str]] = {}
    for line in open(W / "data" / "eval" / "jevbench_dev.jsonl"):
        c = json.loads(line)
        by.setdefault(c["slice"], []).append(c["case_id"])
    rng = random.Random(20261007)
    ids = [i for s in sorted(by) for i in rng.sample(by[s], min(20, len(by[s])))]
    (BASE / "latency_ids.txt").write_text("\n".join(ids) + "\n")
    rc = score(model, run_id, "jevbench_dev", extra=("--case-ids", str(BASE / "latency_ids.txt")), workers=1,
               timeout_min=timeout_min)
    recs = [r for r in preds("jevbench_dev", run_id).values() if r.get("latency_ms")]

    def stats(xs: list[float]) -> dict:
        xs = sorted(xs)
        return {"n": len(xs), "p50_ms": round(xs[len(xs) // 2], 1), "p95_ms": round(xs[max(0, int(len(xs) * 0.95) - 1)], 1)}
    out = {"exit": rc, "engine_note": "vLLM --enforce-eager, no prefix caching, one request at a time"}
    if recs:
        out["all"] = stats([r["latency_ms"] for r in recs])
        out["by_slice"] = {s: stats([r["latency_ms"] for r in recs if r["slice"] == s])
                           for s in sorted({r["slice"] for r in recs})}
    return out


# ---------------------------------------------------------------- session

def main() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_bos.tgz -C {W} && tar xzf {BASE}/adapters.tgz -C {BASE}",
                   shell=True, check=True)
    status(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                          capture_output=True, text=True).stdout.strip() or "no GPU (dry run)")
    status(f"adapters present: {[n for n, p in ADAPTERS.items() if (p / 'adapter_model.safetensors').exists()]}")
    if not SERVER_CMD:  # a slow download inside the server's health wait would look like a failed server
        t0 = time.time()
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "tokenizer*"])
        status(f"model downloaded in {time.time() - t0:.0f}s")
    server = start(ADAPTERS, [], "bf16")
    if server is None:
        status("STOPPED: bf16 server unusable")
        return
    try:
        if not bos_check():
            status("STOPPED: <bos> check failed; nothing measured")
            return
        if not canary():
            status("STOPPED: canary failed; nothing measured")
            return
        for suite in ("jevbench_dev", "custom_dev2", "jevbench", "custom"):
            status(f"untuned+bos {suite}: exit {score(MODEL, 'base-bos-vllm', suite)}, acc {accuracy(suite, 'base-bos-vllm')}")
        # The adapters were trained WITHOUT <bos>; their +bos scores mix that train/inference mismatch with the fix and
        # are read against their own no-bos dev runs (F-step-500-vllm, pilot2-vllm), not as true scores.
        for name in ADAPTERS:
            for suite in ("jevbench_dev", "custom_dev2"):
                status(f"{name}+bos {suite}: exit {score(name, f'{name}-bos-vllm', suite)}, acc {accuracy(suite, f'{name}-bos-vllm')}")
        record("latency_bf16", {"untuned": latency(MODEL, "lat-base-bf16"), "optionA": latency("A500", "lat-A500-bf16")})
        status(f"latency bf16: {results['latency_bf16']}")
    finally:
        stop(server)
    fp8 = start({}, ["--quantization", "fp8"], "fp8")
    if fp8 is None:
        status("FP8: skipped (server not usable with --quantization fp8)")
        record("fp8", {"skipped": True})
    else:
        try:
            # bounded: on a slow FP8 fallback each step stops at its limit and is never retried (about 15 min in all)
            for suite in ("jevbench_dev", "custom_dev2"):
                rc = score(MODEL, "base-fp8-bos-vllm", suite, timeout_min=5)
                status(f"untuned fp8+bos {suite}: exit {rc}, acc {accuracy(suite, 'base-fp8-bos-vllm')}")
            record("fp8", {"latency": latency(MODEL, "lat-base-fp8", timeout_min=5)})
            status(f"latency fp8: {results['fp8']['latency']}")
        finally:
            stop(fp8)
    status("MEASUREMENTS DONE")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
T0 = time.time()
try:
    main()
except Exception as exc:  # logged so the watcher sees it; results so far are still packed
    status(f"ABORTED: {exc!r}"[:300])
rc = subprocess.run(f"cd {W} && tar czf {BASE}/bos_out.tgz results/raw -C {BASE} measurements.json $(cd {BASE} && ls *.log)",
                    shell=True).returncode
status(f"packed bos_out.tgz exit {rc}")
status(f"SESSION DONE, elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("B_HOLD_MIN", 20)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
