"""Generate new decision task families with an LLM generator served by vLLM (gpt-oss-120b on Colab).

v1 generalized poorly because it had 25 task families with fixed phrasing. This
asks a generator to invent many new families, each a (domain x decision pattern)
seed: a written policy or rubric, a JSON state shape, a question with options,
and instances whose answers follow from applying the policy to the state.

The generator's answers are not trusted on their own. targets.py keeps a
synthetic row only when an independent teacher from a different model family
(Gemma 4) agrees with it, and the row's family is recorded so validation can
hold whole families out.

    python pipeline/generate_synthetic.py --base-url http://localhost:8000 --model openai/gpt-oss-120b \\
        --families 300 --instances 25 --workers 32 --out data/train/synthetic_v2.jsonl

Resumable: families already in the output file are skipped.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm

DOMAINS = [
    "hospital admissions", "pharmacy dispensing", "veterinary clinic triage", "health insurance claims", "clinical trial screening",
    "freight logistics", "warehouse inventory", "last-mile delivery", "maritime shipping", "aviation maintenance",
    "card payments", "loan underwriting", "anti-money-laundering review", "expense reimbursement", "accounts payable",
    "payroll", "employee onboarding", "recruiting pipeline", "workplace safety", "benefits enrollment",
    "contract review", "legal discovery", "regulatory compliance", "privacy requests", "export controls",
    "online marketplace listings", "subscription billing", "customer support tickets", "returns and refunds", "loyalty programs",
    "cloud cost management", "incident response", "CI/CD pipelines", "database operations", "network operations",
    "identity and access management", "vulnerability management", "fraud detection", "content moderation", "child safety review",
    "university admissions", "course scheduling", "library services", "grant applications", "scientific lab operations",
    "energy grid operations", "utilities billing", "smart building management", "agriculture irrigation", "food safety inspection",
    "manufacturing quality control", "automotive service", "construction site safety", "real estate leasing", "property maintenance",
    "hotel reservations", "travel booking", "event ticketing", "restaurant operations", "sports league scheduling",
    "news publishing", "advertising review", "localization QA", "accessibility audits", "public benefits eligibility",
    "coding agent", "web browsing agent", "email assistant agent", "calendar assistant agent", "spreadsheet agent",
    "slide deck design", "document formatting", "data pipeline monitoring", "ML model deployment", "API gateway policy",
]
PATTERNS = [
    ("noul", "eligibility or policy check: does the case satisfy a written policy with several conditions"),
    ("choice", "routing or triage: which team, queue or handler should take the case"),
    ("score", "severity or priority rating on a 3-5 level ordinal scale defined by explicit thresholds"),
    ("choice", "next best action for an automated agent given its goal and history"),
    ("noul", "anomaly or violation detection in records, logs or measurements"),
    ("choice", "root-cause or category classification from evidence in the state"),
    ("noul", "completion check: has a multi-step task been fully and correctly finished"),
    ("score", "risk rating combining several factors with explicit rules"),
    ("noul", "duplicate or same-entity detection between two records"),
    ("choice", "which of several candidate items best satisfies stated constraints"),
    ("noul", "consistency check between two documents or fields"),
    ("score", "quality rating against a rubric with defined levels"),
    # v2 scale-up: the pilot's weakest patterns appear twice so they are seeded twice as often
    ("score", "severity or priority rating on a 3-5 level ordinal scale defined by explicit thresholds"),
    ("score", "risk rating combining several factors with explicit rules"),
    ("noul", "duplicate or same-entity detection between two records"),
    ("choice", "next best action for an automated agent given its goal and history"),
    ("noul", "safety review of a long agent trajectory: did the agent violate a stated rule at any step (states hold 15-40 steps of tool calls and observations)"),
]

PROMPT = """Design one decision task for training a small decision model.

Domain: {domain}
Decision pattern: {pattern}
Question type: {qtype}

Requirements:
- Write a concrete policy, rubric or set of rules (2-6 conditions, with exact thresholds where numbers matter). The answer for every instance must follow unambiguously from applying it to that instance's state.
- Each instance has a realistic JSON state (nested objects and arrays are welcome; 3-15 fields, including some fields that do not affect the answer). Precompute any arithmetic the decision needs (durations, percentages, totals) as fields, so no instance requires date or multi-step arithmetic.
- Vary the states widely and include edge cases near thresholds. Balance the answers: every option should be the correct answer for roughly the same number of instances.
- Use fictional names and identifiers only. No real people, no real personal data.
- Question type rules: "noul" means a yes/no question with option keys "true" and "false"; "choice" means 3-8 options with short snake_case keys; "score" means an ordered list of 3-5 levels from lowest to highest, and answers are the 0-based level index as a string.
- Put the policy either in the instructions or inside each state under the key "policy" (choose one for the whole task).
- {polarity}

Return only JSON, no prose, in exactly this shape:
{{"family_name": "snake_case_name", "instructions": "question text (including the policy if it is not in the state)",
  "policy_in_state": false,
  "options": {{"key": "description", ...}} or ["level0", "level1", ...] for score,
  "instances": [{{"state": {{...}}, "answer": "key or level index"}}, ... {n} instances]}}"""


POLARITY = {
    "positive": "Phrase the question so that answering yes/true or a higher level means the favourable or permitted outcome.",
    "negative": "Phrase the question so that answering yes/true or a higher level means the unfavourable, blocked or risky outcome "
                "(for example 'Should this be rejected?', 'Did the agent break a rule?').",
}


def seeds(n: int, rng: random.Random) -> list[tuple[str, str, str]]:
    combos = [(d, p, t) for d, (t, p) in itertools.product(DOMAINS, PATTERNS)]
    rng.shuffle(combos)
    return combos[:n]


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def to_rows(spec: dict, qtype: str, domain: str, pattern: str, idx: int) -> list[dict]:
    """Validate a generated family and convert it to training rows; returns [] if anything is malformed."""
    name = re.sub(r"[^a-z0-9_]", "_", str(spec.get("family_name", "")).lower())[:50] or f"family{idx}"
    family = f"syn_{idx:04d}_{name}"
    instr, opts, insts = spec.get("instructions"), spec.get("options"), spec.get("instances")
    if not isinstance(instr, str) or not isinstance(insts, list) or len(insts) < 8:
        return []
    if qtype == "score":
        if not isinstance(opts, list) or not 3 <= len(opts) <= 5:
            return []
        criteria, keys = [str(o) for o in opts], {str(i) for i in range(len(opts))}
    elif qtype == "noul":
        if not isinstance(opts, dict) or set(opts) != {"true", "false"}:
            return []
        criteria, keys = {"true": str(opts["true"]), "false": str(opts["false"])}, {"true", "false"}
    else:
        if not isinstance(opts, dict) or not 3 <= len(opts) <= 8:
            return []
        criteria, keys = {str(k): str(v) for k, v in opts.items()}, {str(k) for k in opts}
    rows, seen = [], set()
    for inst in insts:
        if not isinstance(inst, dict) or not isinstance(inst.get("state"), dict):
            continue
        ans = str(inst.get("answer", "")).strip().lower() if qtype == "noul" else str(inst.get("answer", "")).strip()
        if ans not in keys:
            continue
        sig = json.dumps(inst["state"], sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        gold = (ans == "true") if qtype == "noul" else (int(ans) if qtype == "score" else ans)
        rows.append({"row_id": f"{family}-{len(rows)}", "family": family, "task_type": qtype, "state": inst["state"],
                     "question": {"type": qtype, "instructions": instr, "criteria": criteria}, "gold": gold,
                     "n_options": 2 if qtype == "noul" else len(criteria),
                     "provenance": {"source": "synthetic", "generator": "openai/gpt-oss-120b", "domain": domain,
                                    "pattern": pattern, "license": "Apache-2.0", "origin": "OpenAI gpt-oss-120b (US); this repo's prompt"}})
    answers = {str(r["gold"]) for r in rows}
    return rows if len(rows) >= 8 and len(answers) >= 2 else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--families", type=int, default=300)
    ap.add_argument("--instances", type=int, default=25)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--start-index", type=int, default=0, help="family index offset, so a new batch never reuses pilot ids")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop submitting new families after this long")
    ap.add_argument("--out", type=Path, default=Path("data/train/synthetic_v2.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    todo = [(args.start_index + i, sd) for i, sd in enumerate(seeds(args.families, rng))]
    done = set()
    if args.out.exists():
        done = {int(json.loads(l)["family"].split("_")[1]) for l in args.out.open(encoding="utf-8")}
    todo = [t for t in todo if t[0] not in done]
    print(f"{len(todo)} families to generate, {len(done)} already done", flush=True)
    client = httpx.Client(timeout=900, limits=httpx.Limits(max_connections=args.workers * 2))
    lock, stats = threading.Lock(), {"ok": 0, "rejected": 0, "rows": 0}

    def make(item):
        idx, (domain, pattern, qtype) = item
        polarity = POLARITY["negative" if idx % 2 else "positive"]  # half the families phrase the question each way
        long_trace = "agent trajectory" in pattern
        body = {"model": args.model, "temperature": 1.0, "max_tokens": 16000 if long_trace else 12000, "reasoning_effort": "medium",
                "messages": [{"role": "user", "content": PROMPT.format(domain=domain, pattern=pattern, qtype=qtype,
                                                                      n=12 if long_trace else args.instances, polarity=polarity)}]}
        try:
            r = client.post(args.base_url.rstrip("/") + "/v1/chat/completions", json=body)
            r.raise_for_status()
            spec = parse(r.json()["choices"][0]["message"]["content"] or "")
        except Exception:
            return []
        return to_rows(spec, qtype, domain, pattern, idx) if spec else []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import time
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None

    def make_unless_late(item):
        if deadline and time.time() > deadline:
            return []  # time cap reached: skip without calling the model
        return make(item)

    with args.out.open("a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        for fut in tqdm(as_completed([pool.submit(make_unless_late, t) for t in todo]), total=len(todo), unit="family", mininterval=10):
            rows = fut.result()
            with lock:
                stats["ok" if rows else "rejected"] += 1
                stats["rows"] += len(rows)
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
    print("SYNTH " + json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
