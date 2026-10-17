"""Notebook cell: check the GPU-server path in MODEL_CARD.md on the published model, in bf16 and FP8.

  1. vLLM 0.31.0 in its own environment; j-raghavan/s1-gemma4-26b-decision downloaded with no token
  2. for bf16, then FP8 (--quantization fp8; skipped cleanly if vLLM rejects it):
     - the API (S1_BACKEND=vllm) starts first, and /healthz must answer 503 while vLLM is not serving
     - vLLM starts with the model card's flags; /healthz must then answer 200
     - JevBench test (1,700) and dev (765) scored through vLLM (eval/jevbench/run_ollama.py, the API's own fetcher)
     - the 200 fixed canary cases sent one at a time through POST /v1/decisions: the API's top answer must match the
       direct run on at least 99% (same server and fetcher; the rest are batching numerics) and its round trip is timed
     - per-decision latency, one request at a time, on the 216-case dev sample of earlier sessions
     - vLLM's own memory figures (weights loaded, KV cache) read from its log
  3. results packed into /content/vllm_check_out.tgz; the VM is held until /content/downloaded.flag appears

Expects /content/s1_vllm_check.tgz (repository code, data/eval, the saved bf16 run results/raw/jevbench/g26-bos-vllm).

    colab exec -s <session> -f colab/session_vllm_check.py

V_* environment variables point the cell at tests/mock_vllm.py and a small case count for a local dry run.
"""

import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "j-raghavan/s1-gemma4-26b-decision"
BASE = Path(os.environ.get("V_BASE", "/content"))
SERVER_CMD = os.environ.get("V_SERVER_CMD")  # dry run: a stand-in for vLLM (gets --port appended)
LIMIT = ["--limit", os.environ["V_LIMIT"]] if os.environ.get("V_LIMIT") else []
CAP_MIN = float(os.environ.get("V_CAP_MIN", 40))  # stop starting new work after this many minutes
VLLM_VERSION = "0.31.0"
VLLM_PORT, API_PORT = 8001, 8000
API_MIN = 0.99
W = BASE / "s1"
VENV = BASE / "vllm_env"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
REPORT = BASE / "vllm_check.json"
T0 = time.time()
results: dict = {}


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    results[key] = value
    REPORT.write_text(json.dumps(results, indent=1))


def http(url: str, body: dict | None = None, timeout: float = 600) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def stop(proc) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=60)
    except ProcessLookupError:
        pass
    time.sleep(5)


def setup() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_vllm_check.tgz -C {W}", shell=True, check=True)
    status(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                          capture_output=True, text=True).stdout.strip() or "no GPU (dry run)")
    log = open(BASE / "install.log", "w")
    if not SERVER_CMD:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv", "httpx", "tqdm", "fastapi>=0.115", "uvicorn>=0.30"],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        subprocess.run(["uv", "venv", str(VENV), "--python", sys.executable], check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(["uv", "pip", "install", "--python", str(VENV / "bin" / "python"), f"vllm=={VLLM_VERSION}"],
                       check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        status(f"vLLM {VLLM_VERSION} installed in {time.time() - t0:.0f}s")
        t0 = time.time()
        os.environ.pop("HF_TOKEN", None)
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, token=False)
        status(f"downloaded {MODEL} anonymously in {time.time() - t0:.0f}s")


def start_api():
    env = {**os.environ, "S1_BACKEND": "vllm", "VLLM_URL": f"http://localhost:{VLLM_PORT}", "PYTHONPATH": str(W)}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.server:app", "--port", str(API_PORT), "--log-level", "warning"],
                            cwd=W, env=env, stdout=open(BASE / "api.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        try:
            code, body = http(f"http://localhost:{API_PORT}/healthz", timeout=10)
            return proc, code, body
        except Exception:
            time.sleep(1)
    return proc, None, {}


def start_vllm(tag: str, extra: list[str]):
    if SERVER_CMD:
        cmd = shlex.split(SERVER_CMD) + ["--port", str(VLLM_PORT), "--model", MODEL]
    else:  # the model card's command, plus no prefix caching so latency is not served from earlier prompts
        cmd = [str(VENV / "bin" / "vllm"), "serve", MODEL, "--port", str(VLLM_PORT), "--max-model-len", "8192",
               "--max-logprobs", "20", "--gpu-memory-utilization", "0.90", "--enforce-eager",
               "--no-enable-prefix-caching", *extra]
    log = BASE / f"vllm_{tag}.log"
    proc = subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PATH": f"{VENV / 'bin'}:{os.environ.get('PATH', '')}"})
    t0 = time.time()
    while time.time() - t0 < 900 and proc.poll() is None:
        try:
            code, body = http(f"http://localhost:{VLLM_PORT}/v1/models", timeout=5)
            if code == 200 and any(m.get("id") == MODEL for m in body.get("data", [])):
                status(f"{tag}: vLLM serving in {time.time() - t0:.0f}s")
                return proc
        except Exception:
            pass
        time.sleep(5)
    status(f"{tag}: vLLM not usable (exit {proc.poll()}); tail: {' | '.join(log.read_text().splitlines()[-3:])[:400]}")
    stop(proc)
    return None


def memory(tag: str) -> dict:
    text = (BASE / f"vllm_{tag}.log").read_text(errors="replace")
    grab = lambda pat: (m.group(1) if (m := re.search(pat, text)) else None)  # noqa: E731
    return {"weights_loaded_gib": grab(r"Model loading took ([\d.]+) GiB"),
            "kv_cache_gib": grab(r"Available KV cache memory: ([\d.]+) GiB"),
            "max_concurrency_8k": grab(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x")}


def score(run_id: str, suite: str, extra: tuple = (), workers: int = 64) -> int:
    args = [sys.executable, "eval/jevbench/run_ollama.py", "--backend", "vllm", "--base-url", f"http://localhost:{VLLM_PORT}",
            "--template", "gemma4", "--model", MODEL, "--run-id", run_id, "--suite", suite, "--workers", str(workers),
            *extra, *([] if extra else LIMIT)]
    try:
        return subprocess.run(args, cwd=W, stdout=open(BASE / f"score_{run_id}_{suite}.log", "w"),
                              stderr=subprocess.STDOUT, timeout=30 * 60).returncode
    except subprocess.TimeoutExpired:
        return -9


def preds(suite: str, run_id: str) -> dict:
    from s1.decide import load_predictions
    p = W / f"results/raw/{suite}/{run_id}/predictions.jsonl"
    return load_predictions(p) if p.exists() else {}


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"n": len(xs), "p50_ms": round(xs[len(xs) // 2], 1), "p95_ms": round(xs[max(0, int(len(xs) * 0.95) - 1)], 1)} if xs else {}


def api_check(tag: str, direct: dict) -> dict:
    """The canary cases one at a time through POST /v1/decisions; top answer compared with the direct run."""
    from paths import SUITES, read_jsonl
    ids = set((W / "data/eval/canary_ids.txt").read_text().split())
    cases = [c for c in read_jsonl(SUITES["jevbench"]["cases"]) if c["case_id"] in ids and c["case_id"] in direct]
    same, rows, lat = 0, [], []
    for c in cases:
        q = c["question"]
        t0 = time.perf_counter()
        code, body = http(f"http://localhost:{API_PORT}/v1/decisions",
                          {"state": c["state"], "questions": {"q": {k: q[k] for k in ("type", "instructions", "criteria")}}})
        lat.append((time.perf_counter() - t0) * 1000)
        if code != 200:
            rows.append({"case_id": c["case_id"], "http": code, "detail": str(body)[:200]})
            continue
        a = body["answers"]["q"]
        top = (a["noul_raw"] >= 0.5) if a["type"] == "noul" else max(a["probabilities"], key=a["probabilities"].get)
        ok = str(top).lower() == str(direct[c["case_id"]]["pred"]).lower()
        same += ok
        rows.append({"case_id": c["case_id"], "api": top, "direct": direct[c["case_id"]]["pred"], "same": ok})
    with (BASE / f"api_cases_{tag}.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    n = max(1, len(cases))
    return {"cases": len(cases), "same_top_as_direct": round(same / n, 4), "http_errors": sum("http" in r for r in rows),
            "round_trip": stats(lat), "pass": len(cases) > 0 and same / n >= API_MIN}


def measure(tag: str, extra: list[str]) -> None:
    from s1.decide import agreement, slice_mean_accuracy
    api, code, body = start_api()
    record(f"{tag}_healthz_before_vllm", {"http": code, "body": body})
    status(f"{tag}: /healthz before vLLM -> {code} ({body.get('vllm', '')[:80]})")
    server = start_vllm(tag, extra)
    try:
        if server is None:
            record(tag, {"skipped": True})
            return
        record(f"{tag}_memory", memory(tag))
        code, body = http(f"http://localhost:{API_PORT}/healthz", timeout=30)
        record(f"{tag}_healthz_ready", {"http": code, "body": body})
        status(f"{tag}: /healthz with vLLM -> {code}; memory {results[f'{tag}_memory']}")
        run_id = f"s1-vllm-{tag}"
        out = {}
        for suite in ("jevbench", "jevbench_dev"):
            t0 = time.time()
            rc = score(run_id, suite)
            p = preds(suite, run_id)
            out[suite] = {"exit": rc, "cases": len(p), "errors": sum(r.get("error") is not None for r in p.values()),
                          "accuracy": round(slice_mean_accuracy(p), 4) if p else None, "seconds": round(time.time() - t0)}
            status(f"{tag} {suite}: {out[suite]}")
        ref = preds("jevbench", "g26-bos-vllm")
        share, n = agreement(preds("jevbench", run_id), ref)
        out["same_top_as_saved_bf16_vllm"] = {"share": round(share, 4), "cases": n}
        status(f"{tag}: same top answer as the saved bf16 vLLM release run {share:.3f} on {n}")
        out["api"] = api_check(tag, preds("jevbench", run_id))
        status(f"{tag} API: {out['api']}")
        rng = random.Random(20261007)  # the dev sample of the earlier latency measurements
        by: dict[str, list[str]] = {}
        for line in open(W / "data/eval/jevbench_dev.jsonl"):
            c = json.loads(line)
            by.setdefault(c["slice"], []).append(c["case_id"])
        ids = [i for s in sorted(by) for i in rng.sample(by[s], min(20, len(by[s])))]
        (BASE / "latency_ids.txt").write_text("\n".join(ids) + "\n")
        lat_id = f"lat-{tag}"
        score(lat_id, "jevbench_dev", extra=("--case-ids", str(BASE / "latency_ids.txt")), workers=1)
        recs = [r for r in preds("jevbench_dev", lat_id).values() if r.get("latency_ms")]
        out["latency"] = {"all": stats([r["latency_ms"] for r in recs]),
                          "by_slice": {s: stats([r["latency_ms"] for r in recs if r["slice"] == s])
                                       for s in sorted({r["slice"] for r in recs})},
                          "note": "vLLM --enforce-eager, no prefix caching, one request at a time"}
        status(f"{tag} latency: {out['latency']['all']}")
        record(tag, out)
    finally:
        stop(server)
        stop(api)


def main() -> None:
    setup()
    sys.path[:0] = [str(W), str(W / "eval" / "jevbench")]
    measure("bf16", [])
    if (time.time() - T0) / 60 < CAP_MIN:
        measure("fp8", ["--quantization", "fp8"])
    else:
        status(f"fp8 skipped: past the {CAP_MIN:.0f}-minute cap")
    b, f = results.get("bf16", {}), results.get("fp8", {})
    ok = (results.get("bf16_healthz_before_vllm", {}).get("http") == 503
          and results.get("bf16_healthz_ready", {}).get("http") == 200 and b.get("api", {}).get("pass"))
    status(f"bf16 jevbench {b.get('jevbench', {}).get('accuracy')} | fp8 jevbench {f.get('jevbench', {}).get('accuracy')}")
    status("VLLM CHECK DONE" if ok else "VLLM CHECK FAILED")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
try:
    main()
except Exception as exc:  # logged so the watcher sees it; results so far are still packed
    status(f"ABORTED: {exc!r}"[:300])
rc = subprocess.run(f"cd {W} && tar czf {BASE}/vllm_check_out.tgz results/raw -C {BASE} vllm_check.json "
                    f"$(cd {BASE} && ls *.log api_cases_*.jsonl 2>/dev/null)", shell=True).returncode
status(f"packed vllm_check_out.tgz exit {rc}")
status(f"SESSION DONE, elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("V_HOLD_MIN", 10)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
