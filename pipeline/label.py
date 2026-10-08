"""Label training rows with the teacher's option distribution (soft labels).

Uses exactly the prompt and probability extraction that scored the eval sets
(eval/jevbench/run_ollama.py), so teacher quality measured there carries over.

Backends:
  ollama  local Ollama server (default http://localhost:11434); sequential; for tests on the Mac
  vllm    a vLLM OpenAI-compatible server (/v1/completions with logprobs); many requests in parallel

    # local smoke test, 30 rows
    uv run pipeline/label.py data/train/structured_v1.jsonl --backend ollama --model gemma4:26b --limit 30

    # Colab, against a vLLM server started on the same VM
    python pipeline/label.py rows.jsonl --backend vllm --model google/gemma-4-26B-A4B-it \\
        --base-url http://localhost:8000 --workers 64 --shard 0/4

Output (one JSON line per row, appended, so runs resume where they stopped):
  <out>/<input stem>.<model>.labels.jsonl with row_id, family, teacher distribution over the
  question's option keys (`probs`), label mass, LLM calls, prompt tokens, latency, error.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "jevbench"))
from run_ollama import next_token_logprobs, score_case, vllm_fetcher  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "labels"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path)
    ap.add_argument("--backend", choices=["ollama", "vllm"], default="ollama")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--base-url", default="http://localhost:8000", help="vLLM server (vllm backend only)")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (keep 1 for Ollama)")
    ap.add_argument("--shard", default="0/1", help="i/n: label every n-th row starting at i")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    i, n = map(int, args.shard.split("/"))
    rows = [r for k, r in enumerate(read_rows(args.rows)) if k % n == i]
    if args.limit:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("/", "_").replace(":", "-")
    out_path = args.out / f"{args.rows.stem}.{tag}.shard{i}of{n}.labels.jsonl"
    done = {json.loads(l)["row_id"] for l in out_path.open(encoding="utf-8")} if out_path.exists() else set()
    todo = [r for r in rows if r["row_id"] not in done]
    print(f"{out_path.name}: {len(todo)} to label, {len(done)} already done", flush=True)

    # score_case uses the Gemma 4 template, whose tokenizer does not add <bos> (see run_ollama.vllm_fetcher)
    fetch = vllm_fetcher(args.base_url, bos="<bos>") if args.backend == "vllm" else next_token_logprobs
    lock = threading.Lock()
    client = httpx.Client(timeout=600, limits=httpx.Limits(max_connections=max(4, args.workers * 2)))

    def label(row: dict) -> dict:
        case = {"state": row["state"], "question": row["question"], "task_type": row["task_type"]}
        t0 = time.perf_counter()
        try:
            s = score_case(client, args.model, case, fetch=fetch)
            return {"row_id": row["row_id"], "family": row["family"], "teacher": args.model, "backend": args.backend,
                    "probs": s["by_key"], "label_mass": s["label_mass"], "llm_calls": s["calls"],
                    "input_tokens": s["input_tokens"], "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "error": None}
        except Exception as exc:  # recorded and retried on the next run, never silently dropped
            return {"row_id": row["row_id"], "family": row["family"], "teacher": args.model, "error": repr(exc)[:300]}

    t_start, tokens, errors = time.perf_counter(), 0, 0
    with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(label, r) for r in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="row", mininterval=5):
            rec = fut.result()
            if rec.get("error"):
                errors += 1
                continue  # not written, so a rerun picks it up
            tokens += rec["input_tokens"]
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
    elapsed = time.perf_counter() - t_start
    print(json.dumps({"labeled": len(todo) - errors, "errors": errors, "seconds": round(elapsed, 1),
                      "rows_per_s": round((len(todo) - errors) / elapsed, 3) if elapsed else None,
                      "prompt_tokens_per_s": round(tokens / elapsed, 1) if elapsed else None}), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
