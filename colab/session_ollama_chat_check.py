"""Notebook cell: score the published Ollama build on the full JevBench test set through the raw and the chat path.

  1. Ollama 0.35.0 installed, jrlabs01/s1 pulled from the registry as a user would get it; its config must have
     thinking off (scripts/ollama_configure.py)
  2. raw path: eval/jevbench/run_ollama.py with <bos>, the documented prompt (reproduces the published int4 run)
  3. chat path: every case as one /api/chat user message with Ollama's defaults (no think field, no prefill);
     probabilities from the reply's token logprobs, plus the reply text. Two-digit option codes (more than 26
     options) follow the generated first digit: the chat path cannot ask for the second digit after another one.
  4. predictions packed into /content/ollama_chat_out.tgz; the VM is held until /content/downloaded.flag appears

Expects /content/s1_ollama_chat.tgz (repository code and data/eval/jevbench_subset.jsonl).

    colab exec -s <session> -f colab/session_ollama_chat_check.py

O_LOCAL=1 uses an Ollama already running with the model (a local dry run); O_LIMIT=<n> scores n cases per slice.
"""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "jrlabs01/s1"
OLLAMA_VERSION = "0.35.0"
BASE = Path(os.environ.get("O_BASE", "/content"))
LOCAL = os.environ.get("O_LOCAL") == "1"
LIMIT = int(os.environ.get("O_LIMIT", 0)) or None
WORKERS = int(os.environ.get("O_WORKERS", 8))
W = BASE / "s1"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
REPORT = BASE / "ollama_chat_check.json"
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
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_ollama_chat.tgz -C {W}", shell=True, check=True)
    if LOCAL:
        return
    status(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                          capture_output=True, text=True).stdout.strip())
    log = open(BASE / "install.log", "w")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "httpx", "tqdm"], check=True, stdout=log, stderr=subprocess.STDOUT)
    subprocess.run("apt-get install -y -qq zstd >/dev/null 2>&1; curl -fsSL https://ollama.com/install.sh | sh", shell=True,
                   check=True, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "OLLAMA_VERSION": OLLAMA_VERSION})
    # jrlabs01/s1 is an MLX (safetensors) build: on Linux it needs the separate MLX runtime, which install.sh skips
    prefix = Path(subprocess.run(["which", "ollama"], capture_output=True, text=True).stdout.strip()).parents[1]
    subprocess.run(f"curl -fsSL https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-linux-amd64-mlx.tar.zst"
                   f" | zstd -d | tar -x -C {prefix}", shell=True, check=True, stdout=log, stderr=subprocess.STDOUT)
    libs = subprocess.run(f"find {prefix}/lib/ollama -name 'libmlxc*'", shell=True, capture_output=True, text=True).stdout.split()
    status(f"MLX runtime installed under {prefix}: {libs}")
    env = {**os.environ, "OLLAMA_NUM_PARALLEL": str(WORKERS), "OLLAMA_CONTEXT_LENGTH": "8192", "OLLAMA_KEEP_ALIVE": "-1"}
    subprocess.Popen(["ollama", "serve"], stdout=open(BASE / "ollama.log", "w"), stderr=subprocess.STDOUT, env=env,
                     start_new_session=True)
    time.sleep(5)
    version = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    subprocess.run(["ollama", "pull", MODEL], check=True, stdout=open(BASE / "pull.log", "w"), stderr=subprocess.STDOUT)
    status(f"{version}; pulled {MODEL} in {time.time() - t0:.0f}s")


def check_config() -> bool:
    import httpx
    show = httpx.post("http://localhost:11434/api/show", json={"model": MODEL}, timeout=60).json()
    mf = show.get("modelfile", "")
    cfg = {"capabilities": show.get("capabilities"), "renderer": re.search(r"^RENDERER (.+)$", mf, re.M).group(1),
           "parser": re.search(r"^PARSER (.+)$", mf, re.M).group(1)}
    ok = "thinking" not in (cfg["capabilities"] or []) and cfg["renderer"] == "gemma4-large"
    record("config", cfg | {"pass": ok})
    status(f"config {cfg} -> {'PASS' if ok else 'FAIL'}")
    return ok


def cases() -> list[dict]:
    from paths import read_jsonl
    rows = read_jsonl(W / "data/eval/jevbench_subset.jsonl")
    if not LIMIT:
        return rows
    kept, seen = [], {}
    for c in rows:  # the first LIMIT cases of each slice, as run_ollama.py --limit picks them
        seen[c["slice"]] = seen.get(c["slice"], 0) + 1
        if seen[c["slice"]] <= LIMIT:
            kept.append(c)
    return kept


def raw_path() -> None:
    t0 = time.time()
    args = [sys.executable, "eval/jevbench/run_ollama.py", "--backend", "ollama", "--model", MODEL, "--ollama-add-bos",
            "--run-id", "s1-int4-raw-colab", "--suite", "jevbench", "--workers", str(WORKERS)]
    if LIMIT:
        args += ["--limit", str(LIMIT)]
    rc = subprocess.run(args, cwd=W, stdout=open(BASE / "raw.log", "w"), stderr=subprocess.STDOUT, timeout=60 * 60).returncode
    p = W / "results/raw/jevbench/s1-int4-raw-colab/predictions.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    out = {"exit": rc, "cases": len(rows), "errors": sum(r.get("error") is not None for r in rows), "seconds": round(time.time() - t0)}
    record("raw", out)
    status(f"raw path: {out}")


def chat_one(client, case: dict) -> dict:
    from paths import base_record
    from run_ollama import GEMMA4_PROMPT, distribution, labels_for, options_for, render, to_record
    opts = options_for(case["question"])
    labels = labels_for(len(opts))
    user = render(case, labels, opts, "gemma4")[len(GEMMA4_PROMPT.split("{user}")[0]):].split("<turn|>")[0]
    t0 = time.perf_counter()
    try:
        r = client.post("http://localhost:11434/api/chat", json={
            "model": MODEL, "messages": [{"role": "user", "content": user}], "stream": False, "logprobs": True,
            "top_logprobs": 20, "options": {"num_predict": 8}})
        r.raise_for_status()
        body = r.json()
        steps = body.get("logprobs") or []

        def top(i: int) -> dict[str, float]:
            out: dict[str, float] = {}
            for item in (steps[i]["top_logprobs"] if i < len(steps) else []):
                t = item["token"].strip()
                if t and t not in out:
                    out[t] = item["logprob"]
            return out
        if labels[0].isalpha():
            probs, mass = distribution(top(0), labels)
        else:  # two-digit codes: P(first) x P(second | the generated first digit), uniform for the others
            firsts = sorted({lab[0] for lab in labels})
            p_first, mass = distribution(top(0), firsts)
            chosen = steps[0]["token"].strip() if steps else ""
            probs = {}
            for d in firsts:
                seconds = [lab[1] for lab in labels if lab[0] == d]
                p2 = distribution(top(1), seconds)[0] if d == chosen and len(steps) > 1 else {s: 1 / len(seconds) for s in seconds}
                probs.update({d + s: p_first[d] * p2[s] for s in seconds})
        by_key = {key: probs[lab] for lab, (key, _) in zip(labels, opts)}
        rec = to_record(case, {"by_key": by_key, "label_mass": mass, "calls": 1,
                               "input_tokens": body.get("prompt_eval_count", 0)}, "s1-int4-chat", MODEL,
                        (time.perf_counter() - t0) * 1000)
        text = body.get("message", {}).get("content", "")
        rec["reply_text"] = text[:80]
        rec["reply_tokens"] = body.get("eval_count")
        m = re.match(r"\s*\(?([A-Z]|\d{2})\b", text)
        rec["reply_label"] = m.group(1) if m and m.group(1) in labels else None
        rec["reply_key"] = dict(zip(labels, [k for k, _ in opts])).get(rec["reply_label"])
        rec["thinking_chars"] = len(body.get("message", {}).get("thinking", "") or "")
        return rec
    except Exception as exc:  # recorded; the scorer reports coverage
        return base_record(case, "s1-int4-chat") | {"error": repr(exc)[:300], "pred": None}


def chat_path() -> None:
    import httpx
    t0 = time.time()
    todo = cases()
    p = W / "results/raw/jevbench/s1-int4-chat/predictions.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.parent / "meta.json").write_text(json.dumps({"run_id": "s1-int4-chat", "system": f"ollama chat:{MODEL}",
                                                    "method": "/api/chat defaults, one user message, token logprobs"}))
    client = httpx.Client(timeout=600, limits=httpx.Limits(max_connections=WORKERS * 2))
    with p.open("w") as out, ThreadPoolExecutor(WORKERS) as pool:
        for rec in pool.map(lambda c: chat_one(client, c), todo):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    ok = [r for r in rows if r.get("error") is None]
    out = {"cases": len(rows), "errors": len(rows) - len(ok), "seconds": round(time.time() - t0),
           "thinking_replies": sum(r.get("thinking_chars", 0) > 0 for r in ok),
           "no_reply_label": sum(r.get("reply_label") is None for r in ok),
           "median_reply_tokens": sorted(r.get("reply_tokens") or 0 for r in ok)[len(ok) // 2] if ok else None}
    record("chat", out)
    status(f"chat path: {out}")


def main() -> None:
    setup()
    sys.path[:0] = [str(W), str(W / "eval" / "jevbench")]
    if not check_config():
        status("the pulled build still thinks: stopping before scoring")
        return
    raw_path()
    chat_path()
    status("OLLAMA CHAT CHECK DONE")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
try:
    main()
except Exception as exc:  # logged so the watcher sees it; results so far are still packed
    status(f"ABORTED: {exc!r}"[:300])
rc = subprocess.run(f"cd {W} && tar czf {BASE}/ollama_chat_out.tgz results/raw -C {BASE} "
                    f"$(cd {BASE} && ls ollama_chat_check.json *.log 2>/dev/null)", shell=True).returncode
status(f"packed ollama_chat_out.tgz exit {rc}")
status(f"SESSION DONE, elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("O_HOLD_MIN", 10)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
