"""Notebook cell: does averaging over option orders reduce s1's option-order sensitivity, and what does it cost?

  1. vLLM 0.31.0 serving the published j-raghavan/s1-gemma4-26b-decision in bf16 (the model card's command)
  2. every JevBench test and dev case scored in up to three option orders: as given, reversed, and rotated by half
     (yes/no: as given and swapped; ordinal scales keep their order); each order's probabilities, mapped back to
     the option keys, are saved, so every variant (1, 2 or 3 orders; arithmetic or geometric mean) is built offline
  3. latency, one decision at a time on the 216-case dev sample of earlier sessions: one order, and three orders
     sent concurrently (vLLM batches them)
  4. predictions packed into /content/order_check_out.tgz; the VM is held until /content/downloaded.flag appears

Pre-registered before scoring: the variant is chosen on the dev split (highest mean accuracy; ties go to fewer
orders, then to the geometric mean), and the test set is scored once with that variant.

Expects /content/s1_order_check.tgz (repository code and data/eval jevbench_subset.jsonl, jevbench_dev.jsonl).

    colab exec -s <session> -f colab/session_order_check.py

O_SERVER_CMD runs a stand-in server (tests/mock_vllm.py) for a local dry run; O_LIMIT=<n> keeps n cases per slice.
"""

import json
import os
import random
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "j-raghavan/s1-gemma4-26b-decision"
BASE = Path(os.environ.get("O_BASE", "/content"))
SERVER_CMD = os.environ.get("O_SERVER_CMD")
LIMIT = int(os.environ.get("O_LIMIT", 0)) or None
VLLM_VERSION = "0.31.0"
PORT = 8001
W = BASE / "s1"
VENV = BASE / "vllm_env"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
REPORT = BASE / "order_check.json"
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


def setup() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_order_check.tgz -C {W}", shell=True, check=True)
    if SERVER_CMD:
        return
    log = open(BASE / "install.log", "w")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv", "httpx", "tqdm"], check=True, stdout=log,
                   stderr=subprocess.STDOUT)
    subprocess.run(["uv", "venv", str(VENV), "--python", sys.executable], check=True, stdout=log, stderr=subprocess.STDOUT)
    subprocess.run(["uv", "pip", "install", "--python", str(VENV / "bin" / "python"), f"vllm=={VLLM_VERSION}"],
                   check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    os.environ.pop("HF_TOKEN", None)
    t0 = time.time()
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, token=False)
    status(f"vLLM {VLLM_VERSION} installed; downloaded {MODEL} anonymously in {time.time() - t0:.0f}s")


def start_vllm():
    if SERVER_CMD:
        cmd = shlex.split(SERVER_CMD) + ["--port", str(PORT), "--model", MODEL]
    else:
        cmd = [str(VENV / "bin" / "vllm"), "serve", MODEL, "--port", str(PORT), "--max-model-len", "8192",
               "--max-logprobs", "20", "--gpu-memory-utilization", "0.90", "--enforce-eager", "--no-enable-prefix-caching"]
    proc = subprocess.Popen(cmd, stdout=open(BASE / "vllm.log", "a"), stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PATH": f"{VENV / 'bin'}:{os.environ.get('PATH', '')}"})
    t0 = time.time()
    while time.time() - t0 < 900 and proc.poll() is None:
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=5) as r:
                if any(m.get("id") == MODEL for m in json.loads(r.read()).get("data", [])):
                    status(f"vLLM serving in {time.time() - t0:.0f}s")
                    return proc
        except Exception:
            pass
        time.sleep(5)
    status(f"vLLM not usable (exit {proc.poll()})")
    return None


def stop(proc) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=120)
    except Exception:
        pass


def orders(question: dict) -> list[list[str]]:
    """Key orders to score: as given, reversed, rotated by half (deduplicated); ordinal scales keep their order."""
    keys = list(question["criteria"]) if isinstance(question["criteria"], dict) else None
    if question["type"] == "score" or not keys:
        return [keys or []]
    out = []
    for o in (keys, keys[::-1], keys[(len(keys) + 1) // 2:] + keys[:(len(keys) + 1) // 2]):
        if o not in out:
            out.append(o)
    return out


def score_orders(client, fetch, case: dict) -> dict:
    from paths import base_record
    from run_ollama import score_case
    q = case["question"]
    per_order, calls = [], 0
    for keys in orders(q):
        qq = q if q["type"] == "score" else {**q, "criteria": {k: q["criteria"][k] for k in keys}}
        scored = score_case(client, MODEL, {**case, "question": qq}, fetch=fetch)
        per_order.append({"order": keys, "by_key": scored["by_key"]})
        calls += scored["calls"]
    return base_record(case, "s1-orders") | {"per_order": per_order, "calls": calls, "error": None}


def run_suite(suite: str, fetch) -> None:
    import httpx

    from paths import read_jsonl
    cases = read_jsonl(W / f"data/eval/{'jevbench_subset' if suite == 'jevbench' else 'jevbench_dev'}.jsonl")
    if LIMIT:
        seen: dict[str, int] = {}
        kept = []
        for c in cases:
            seen[c["slice"]] = seen.get(c["slice"], 0) + 1
            if seen[c["slice"]] <= LIMIT:
                kept.append(c)
        cases = kept
    out_path = W / f"results/raw/orders/{suite}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=600, limits=httpx.Limits(max_connections=128))
    t0 = time.time()

    def one(c):
        try:
            return score_orders(client, fetch, c)
        except Exception as exc:
            return {"case_id": c["case_id"], "slice": c["slice"], "error": repr(exc)[:300]}
    with out_path.open("w") as out, ThreadPoolExecutor(64) as pool:
        for rec in pool.map(one, cases):
            out.write(json.dumps(rec) + "\n")
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    info = {"cases": len(rows), "errors": sum(r.get("error") is not None for r in rows), "seconds": round(time.time() - t0)}
    record(suite, info)
    status(f"{suite}: {info}")


def latency(fetch) -> None:
    import httpx

    from run_ollama import score_case
    by: dict[str, list[dict]] = {}
    for line in open(W / "data/eval/jevbench_dev.jsonl"):
        c = json.loads(line)
        by.setdefault(c["slice"], []).append(c)
    rng = random.Random(20261007)  # the dev sample of the earlier latency measurements
    sample = [c for s in sorted(by) for c in rng.sample(by[s], min(20, len(by[s])))]
    client = httpx.Client(timeout=600, limits=httpx.Limits(max_connections=8))
    pool = ThreadPoolExecutor(3)

    def timed(fn):
        t = time.perf_counter()
        fn()
        return (time.perf_counter() - t) * 1000
    one, three = [], []
    for c in sample:
        q = c["question"]
        variants = [q if q["type"] == "score" else {**q, "criteria": {k: q["criteria"][k] for k in o}} for o in orders(q)]
        one.append(timed(lambda c=c: score_case(client, MODEL, c, fetch=fetch)))
        three.append(timed(lambda c=c, vs=variants: list(pool.map(
            lambda qq: score_case(client, MODEL, {**c, "question": qq}, fetch=fetch), vs))))

    def stats(xs):
        xs = sorted(xs)
        return {"n": len(xs), "p50_ms": round(xs[len(xs) // 2], 1), "p95_ms": round(xs[max(0, int(len(xs) * 0.95) - 1)], 1)}
    record("latency", {"one_order": stats(one), "orders_concurrent": stats(three),
                       "note": "vLLM bf16 --enforce-eager, no prefix caching, one decision at a time"})
    status(f"latency: one order {stats(one)}, all orders concurrently {stats(three)}")


def main() -> None:
    setup()
    sys.path[:0] = [str(W), str(W / "eval" / "jevbench")]
    from run_ollama import vllm_fetcher
    fetch = vllm_fetcher(f"http://localhost:{PORT}", "<bos>")
    server = start_vllm()
    if server is None:
        return
    try:
        run_suite("jevbench_dev", fetch)
        run_suite("jevbench", fetch)
        latency(fetch)
    finally:
        stop(server)
    status("ORDER CHECK DONE")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
try:
    main()
except Exception as exc:  # logged so the watcher sees it; results so far are still packed
    status(f"ABORTED: {exc!r}"[:300])
rc = subprocess.run(f"cd {W} && tar czf {BASE}/order_check_out.tgz results/raw -C {BASE} "
                    f"$(cd {BASE} && ls order_check.json *.log 2>/dev/null)", shell=True).returncode
status(f"packed order_check_out.tgz exit {rc}")
status(f"SESSION DONE, elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("O_HOLD_MIN", 10)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
