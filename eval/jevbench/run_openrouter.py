"""Score an eval suite with a hosted model on OpenRouter, reading the model's chosen option (accuracy only).

The free OpenRouter endpoints return no logprobs, so this records the single answer the model gives at
temperature 0 as a one-hot distribution. Use it to compare accuracy (e.g. whether a candidate teacher
beats Gemma 4 26B); calibration metrics from this run are not meaningful.

The API key is read from OPENROUTER_API_KEY or the repo's .env (git-ignored) and is never printed.

    uv run eval/jevbench/run_openrouter.py --model thinkingmachines/inkling-small:free --sample 200
    uv run eval/jevbench/run_openrouter.py --model thinkingmachines/inkling-small:free   # all cases

Resumable: cases already answered are skipped, so a run can continue across days of free-tier quota.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime

import httpx

from paths import ROOT, SUITES, base_record, read_jsonl
from run_ollama import labels_for, options_for

API = "https://openrouter.ai/api/v1/chat/completions"


def api_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("OPENROUTER_API_KEY not found in the environment or .env")


def prompt_for(case: dict, fmt: str = "api") -> tuple[str, list[str], list[str]]:
    options = options_for(case["question"])
    labels = labels_for(len(options))
    kind = "letter" if labels[0].isalpha() else "two-digit code"
    lines = "\n".join(f"{lab}) {text}" for lab, (_, text) in zip(labels, options))
    text = ("You are a decision model. Read the state and answer the question by choosing one option.\n\n"
            f"STATE:\n{json.dumps(case['state'], ensure_ascii=False, indent=1)}\n\n"
            f"QUESTION: {case['question']['instructions']}\n\nOPTIONS:\n{lines}\n\n"
            + (f"Answer with the option {kind} only." if fmt.startswith("ours") else f"Reply with only the option {kind}, nothing else."))
    return text, labels, [k for k, _ in options]


def parse(reply: str, labels: list[str]) -> str | None:
    reply = (reply or "").strip()
    pattern = r"\b([A-Z])\b" if labels[0].isalpha() else r"\b(\d{2})\b"
    for m in re.finditer(pattern, reply.upper() if labels[0].isalpha() else reply):
        if m.group(1) in labels:
            return m.group(1)
    return None


def stratified(cases: list[dict], n: int, seed: int) -> list[dict]:
    """n cases spread evenly across slices (whole families, so reordered twins stay together)."""
    rng = random.Random(seed)
    by_slice: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in cases:
        by_slice[c["slice"]][c["family_id"]].append(c)
    per = max(1, n // len(by_slice))
    out = []
    for fams in by_slice.values():
        keys = sorted(fams)
        rng.shuffle(keys)
        picked = 0
        for k in keys:
            if picked >= per:
                break
            out += fams[k]
            picked += len(fams[k])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="thinkingmachines/inkling-small:free")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--suite", default="jevbench", choices=sorted(SUITES))
    ap.add_argument("--sample", type=int, default=None, help="stratified sample size (default: all cases)")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--min-interval", type=float, default=3.5, help="seconds between requests (free tier ~20/min)")
    ap.add_argument("--provider", default=None, help="pin one OpenRouter provider (e.g. Together), no fallbacks")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="seconds per request; providers can hang for minutes on some safety-benchmark prompts")
    ap.add_argument("--slices", nargs="+", default=None, help="only these slices")
    ap.add_argument("--format", choices=["api", "ours", "ours-noprefill"], default="api",
                    help="ours = the wording and 'Answer:' prefill the vLLM/HF readout uses; for format comparisons")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="output budget; thinking needs thousands (default: 16 with reasoning off, 600 otherwise)")
    ap.add_argument("--reasoning", choices=["off", "on", "low", "default"], default="off",
                    help="reasoning models: 'off' answers directly (System One style, matches how Gemma was measured)")
    args = ap.parse_args()

    key = api_key()
    run_id = args.run_id or args.model.split("/")[-1].replace(":", "-")
    suite = SUITES[args.suite]
    cases = read_jsonl(suite["cases"])
    if args.slices:
        cases = [c for c in cases if c["slice"] in args.slices]
    if args.sample:
        cases = stratified(cases, args.sample, args.seed)
    run_dir = suite["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "predictions.jsonl"
    done = {r["case_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [c for c in cases if c["case_id"] not in done]
    (run_dir / "meta.json").write_text(json.dumps({"run_id": run_id, "system": f"openrouter:{args.model}",
                                                   "method": f"single answer at temperature 0, reasoning {args.reasoning} (no logprobs; accuracy only)",
                                                   "n_cases_total": len(cases)}, indent=2))
    print(f"{run_id}: {len(todo)} to score, {len(done)} already done", flush=True)

    headers = {"Authorization": f"Bearer {key}", "X-Title": "s1-model eval"}
    pool = ThreadPoolExecutor(max_workers=8)  # abandoned requests finish in the background without blocking the run
    with httpx.Client(timeout=args.timeout * 6) as client, out_path.open("a", encoding="utf-8") as out:
        last = 0.0
        for i, case in enumerate(todo):
            text, labels, keys = prompt_for(case, args.format)
            wait = args.min_interval - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            t0 = time.perf_counter()
            try:
                body = {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens or (16 if args.reasoning == "off" else 600),
                        "messages": [{"role": "user", "content": text}]
                        + ([{"role": "assistant", "content": "Answer:"}] if args.format == "ours" else [])}
                if args.provider:
                    body["provider"] = {"order": [args.provider], "allow_fallbacks": False}
                if args.reasoning == "off":
                    body["reasoning"] = {"enabled": False}
                elif args.reasoning == "on":  # explicit: 'default' does not switch Gemma 4's thinking on
                    body["reasoning"] = {"enabled": True}
                elif args.reasoning == "low":
                    body["reasoning"] = {"effort": "low"}
                # A hard total deadline: OpenRouter sends keep-alive bytes while a provider is slow, so the HTTP
                # read timeout never fires on its own. A hung request is abandoned and recorded as unanswered.
                fut = pool.submit(client.post, API, headers=headers, json=body)
                try:
                    r = fut.result(timeout=args.timeout)
                except FutureTimeout:
                    raise RuntimeError(f"no answer within {args.timeout:.0f}s") from None
                retries = 0
                while r.status_code == 429 and retries < 10:  # per-minute limits: wait and retry the same case
                    retries += 1
                    time.sleep(20)
                    r = pool.submit(client.post, API, headers=headers, json=body).result(timeout=args.timeout)
                if r.status_code == 429:
                    print(f"still rate limited after {i} cases this run ({r.text[:160]}); stopping, rerun later to resume", flush=True)
                    break
                body_json = r.json()
                if r.status_code != 200 or "choices" not in body_json:
                    raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(body_json.get('error', body_json))[:160]}")
                reply = body_json["choices"][0]["message"].get("content") or ""
            except Exception as exc:
                # Recorded as unanswered (not retried), so the scorer reports reduced coverage instead of the run stalling.
                out.write(json.dumps(base_record(case, run_id) | {"error": repr(exc)[:200], "pred": None}) + "\n")
                out.flush()
                print(f"unanswered {case['case_id']}: {repr(exc)[:120]}", flush=True)
                continue
            lab = parse(reply, labels)
            rec = base_record(case, run_id) | {"latency_ms": (time.perf_counter() - t0) * 1000, "attempts": 1,
                                               "error": None, "answer_type": case["task_type"], "raw_reply": reply[:40],
                                               "model_reported": args.model, "ts": datetime.now(UTC).isoformat()}
            if lab is None:
                rec |= {"error": "unparseable reply", "pred": None}
            else:
                choice = keys[labels.index(lab)]
                probs = {k: (1.0 if k == choice else 0.0) for k in keys}
                if case["task_type"] == "noul":
                    rec |= {"p_true": probs["true"], "pred": choice == "true", "confidence": 1.0}
                elif case["task_type"] == "choice":
                    rec |= {"probabilities": probs, "pred": choice, "confidence": 1.0}
                else:
                    rec |= {"probabilities": probs, "pred": int(choice), "confidence": 1.0, "expected_level": float(choice)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if (i + 1) % 25 == 0:
                print(f"{i + 1} scored this run", flush=True)
    n = sum(1 for _ in out_path.open()) if out_path.exists() else 0
    print(f"{out_path}: {n} records", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
