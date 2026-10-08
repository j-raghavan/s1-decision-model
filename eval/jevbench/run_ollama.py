"""Score JevBench cases with a local LLM served by Ollama, reading probabilities from logits.

Each question is rendered as a lettered (K <= 26) or two-digit-coded (K > 26)
option list. The option distribution is read from the next-token log
probabilities: one call for letters, and for codes one call for the first digit
plus one call per plausible first digit for the second digit. Options that fall
outside the returned top-k get a floor of half the smallest returned
probability, so no option is exactly zero.

    uv run eval/jevbench/run_ollama.py --model gemma4:12b
    uv run eval/jevbench/run_ollama.py --model gemma4:12b --limit 5   # smoke test

Runs resume: cases already in the run's predictions.jsonl are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import string
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tqdm import tqdm

from paths import SUITES, base_record, read_jsonl

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
TOP_K = 20
NUM_CTX = 8192
LETTERS = string.ascii_uppercase
SECOND_DIGIT_MIN_P = 1e-3

# Gemma 4 turn format, rendered from google/gemma-4-12B-it chat_template.jinja
# with add_generation_prompt=True (thinking disabled), minus the leading <bos>: Ollama adds <bos> itself, and the
# vLLM and HF paths prepend it (Gemma 4's tokenizer does not).
GEMMA4_PROMPT = "<|turn>user\n{user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
# gpt-oss "harmony" format, opened directly in the final channel so the model answers without an analysis
# (reasoning) message first, System One style, as Gemma does with its empty thought channel.
HARMONY_PROMPT = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
                  "Reasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every "
                  "message.<|end|><|start|>user<|message|>{user}<|end|><|start|>assistant<|channel|>final<|message|>")
TEMPLATES = {"gemma4": GEMMA4_PROMPT, "harmony": HARMONY_PROMPT}


def options_for(question: dict) -> list[tuple[str, str]]:
    """(key, text) per option, in presented order. Score keys are level indices."""
    criteria = question["criteria"]
    if question["type"] == "score":
        return [(str(i), text) for i, text in enumerate(criteria)]
    out = []
    for key, desc in criteria.items():
        if not desc or desc == key:
            out.append((key, key))
        elif len(key) == 1:  # bare ids like a/b/c add nothing beside the letter label
            out.append((key, desc))
        else:
            out.append((key, f"{key}: {desc}"))
    return out


def labels_for(n: int) -> list[str]:
    if n <= len(LETTERS):
        return list(LETTERS[:n])
    if n > 90:
        raise ValueError(f"{n} options exceeds the two-digit code range")
    return [str(10 + i) for i in range(n)]


def render(case: dict, labels: list[str], options: list[tuple[str, str]], template: str = "gemma4") -> str:
    q = case["question"]
    state = json.dumps(case["state"], ensure_ascii=False, indent=1)
    lines = "\n".join(f"{lab}) {text}" for lab, (_, text) in zip(labels, options))
    kind = "letter" if labels[0].isalpha() else "two-digit code"
    user = (
        "You are a decision model. Read the state and answer the question by choosing one option.\n\n"
        f"STATE:\n{state}\n\nQUESTION: {q['instructions']}\n\nOPTIONS:\n{lines}\n\n"
        f"Answer with the option {kind} only."
    )
    # Prefilling the reply stops the model from opening with reasoning. Gemma
    # tokenizes " A" as one token but " 7" as a space then a digit, so codes
    # need the trailing space.
    prefix = "Answer:" if labels[0].isalpha() else "Answer: "
    return TEMPLATES[template].format(user=user) + prefix


def next_token_logprobs(client: httpx.Client, model: str, prompt: str) -> tuple[dict[str, float], int]:
    resp = client.post(
        OLLAMA,
        json={
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": TOP_K,
            "options": {"temperature": 0, "num_predict": 1, "num_ctx": NUM_CTX},
        },
    )
    resp.raise_for_status()
    body = resp.json()
    out: dict[str, float] = {}
    for item in body["logprobs"][0]["top_logprobs"]:
        tok = item["token"].strip()
        if tok and tok not in out:  # keep the most likely spelling of each stripped token
            out[tok] = item["logprob"]
    return out, body.get("prompt_eval_count", 0)


def ollama_fetcher(bos: str = ""):
    """fetch() for Ollama that prepends <bos> once. Ollama's library Gemma models add <bos> themselves; a model
    imported from safetensors (our s1-gemma4-26b) does not, and a literal "<bos>" in a raw prompt is read as the
    single special token (checked: prompt token count +1)."""
    def fetch(client: httpx.Client, model: str, prompt: str) -> tuple[dict[str, float], int]:
        if bos and not prompt.startswith(bos):
            prompt = bos + prompt
        return next_token_logprobs(client, model, prompt)

    return fetch


def distribution(lps: dict[str, float], allowed: list[str]) -> tuple[dict[str, float], float]:
    """Normalised distribution over allowed tokens, and the raw mass they held."""
    floor = 0.5 * math.exp(min(lps.values())) if lps else 1e-6
    raw = {a: math.exp(lps[a]) if a in lps else 0.0 for a in allowed}
    mass = sum(raw.values())
    filled = {a: (p if p > 0 else floor) for a, p in raw.items()}
    total = sum(filled.values())
    return {a: p / total for a, p in filled.items()}, mass


def vllm_fetcher(base_url: str, bos: str = ""):
    """fetch() for a vLLM OpenAI-compatible server: next-token top logprobs from /v1/completions.

    vLLM tokenizes the prompt with the model's own settings, and Gemma 4's tokenizer does not add <bos> (its chat
    template writes it out); without it the model loses 4-13 points on knowledge questions. Ollama adds <bos>
    itself, so the templates leave it out and Gemma callers pass bos="<bos>" here."""
    url = base_url.rstrip("/") + "/v1/completions"

    def fetch(client: httpx.Client, model: str, prompt: str) -> tuple[dict[str, float], int]:
        if bos and not prompt.startswith(bos):
            prompt = bos + prompt
        resp = client.post(url, json={"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": TOP_K})
        resp.raise_for_status()
        body = resp.json()
        top = body["choices"][0]["logprobs"]["top_logprobs"][0] or {}
        out: dict[str, float] = {}
        for tok, lp in sorted(top.items(), key=lambda kv: -kv[1]):
            t = tok.strip()
            if t and t not in out:
                out[t] = lp
        return out, body.get("usage", {}).get("prompt_tokens", 0)

    return fetch


def score_case(client: httpx.Client, model: str, case: dict, fetch=None, template: str = "gemma4") -> dict:
    """Option distribution for one case. `fetch(client, model, prompt)` returns (top logprobs, prompt tokens);
    it defaults to Ollama, and the labeling pipeline passes a vLLM fetcher instead."""
    fetch = fetch or next_token_logprobs
    options = options_for(case["question"])
    labels = labels_for(len(options))
    prompt = render(case, labels, options, template)
    calls = 1
    lps, n_in = fetch(client, model, prompt)

    whole = sum(math.exp(lps[lab]) for lab in labels if lab in lps)
    if labels[0].isalpha() or whole > 0.5:
        # Letters, or a tokenizer that keeps each two-digit code as one token (gpt-oss): read the labels directly.
        probs, mass = distribution(lps, labels)
    else:
        firsts = sorted({lab[0] for lab in labels})
        p_first, mass = distribution(lps, firsts)
        probs = {}
        for d in firsts:
            seconds = [lab[1] for lab in labels if lab[0] == d]
            if p_first[d] >= SECOND_DIGIT_MIN_P:
                lps2, _ = fetch(client, model, prompt + d)
                calls += 1
                p_second, _ = distribution(lps2, seconds)
            else:
                p_second = {s: 1 / len(seconds) for s in seconds}
            for s in seconds:
                probs[d + s] = p_first[d] * p_second[s]

    by_key = {key: probs[lab] for lab, (key, _) in zip(labels, options)}
    return {"by_key": by_key, "label_mass": mass, "calls": calls, "input_tokens": n_in}


def to_record(case: dict, scored: dict, run_id: str, model: str, latency_ms: float) -> dict:
    rec = base_record(case, run_id)
    probs = scored["by_key"]
    t = case["task_type"]
    rec.update({"latency_ms": latency_ms, "attempts": 1, "error": None, "answer_type": t})
    if t == "noul":
        p_true = probs["true"] / (probs["true"] + probs["false"])
        rec.update({"p_true": p_true, "pred": p_true >= 0.5, "confidence": max(p_true, 1 - p_true)})
    elif t == "choice":
        pred = max(probs, key=probs.get)
        rec.update({"probabilities": probs, "pred": pred, "confidence": probs[pred]})
    else:
        expected = sum(int(k) * p for k, p in probs.items())
        pred = max(probs, key=probs.get)
        rec.update({"probabilities": probs, "pred": int(pred), "confidence": probs[pred], "expected_level": expected})
    rec.update({
        "label_mass": scored["label_mass"],
        "llm_calls": scored["calls"],
        "input_tokens": scored["input_tokens"],
        "output_tokens": 0,
        "model_reported": model,
        "ts": datetime.now(UTC).isoformat(),
    })
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma4:12b")
    ap.add_argument("--run-id", default=None, help="defaults to the model name")
    ap.add_argument("--limit", type=int, default=None, help="cases per slice, for smoke tests")
    ap.add_argument("--case-ids", type=Path, default=None, help="file with one case_id per line: only these cases")
    ap.add_argument("--ollama-add-bos", action="store_true",
                    help="ollama only: prepend <bos> (models imported from safetensors, e.g. s1-gemma4-26b, do not add it)")
    ap.add_argument("--bos", choices=["auto", "none"], default="auto",
                    help="vLLM only: auto prepends <bos> for the gemma4 template; none reproduces runs made before that fix")
    ap.add_argument("--suite", default="jevbench", choices=sorted(SUITES))
    ap.add_argument("--backend", choices=["ollama", "vllm"], default="ollama")
    ap.add_argument("--base-url", default="http://localhost:8000", help="vLLM server (vllm backend only)")
    ap.add_argument("--template", choices=sorted(TEMPLATES), default="gemma4")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (vLLM batches them; keep 1 for Ollama)")
    args = ap.parse_args()
    suite = SUITES[args.suite]
    fetch = (vllm_fetcher(args.base_url, bos="<bos>" if args.template == "gemma4" and args.bos == "auto" else "")
             if args.backend == "vllm" else ollama_fetcher("<bos>" if args.ollama_add_bos else ""))

    run_id = args.run_id or args.model.replace(":", "-")
    run_dir = suite["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "predictions.jsonl"
    done = {r["case_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    cases = read_jsonl(suite["cases"])
    if args.limit:
        seen: dict[str, int] = {}
        kept = []
        for c in cases:
            seen[c["slice"]] = seen.get(c["slice"], 0) + 1
            if seen[c["slice"]] <= args.limit:
                kept.append(c)
        cases = kept
    if args.case_ids:
        wanted = set(args.case_ids.read_text().split())
        cases = [c for c in cases if c["case_id"] in wanted]
    todo = [c for c in cases if c["case_id"] not in done]
    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": run_id, "system": f"{args.backend}:{args.model}", "template": args.template,
        "method": "next-token logprobs over option labels",
        "top_k": TOP_K, "num_ctx": NUM_CTX, "n_cases_total": len(cases), "one_question_per_request": True,
    }, indent=2))
    print(f"{run_id}: {len(todo)} to score, {len(done)} already done")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(case: dict) -> dict:
        t0 = time.perf_counter()
        try:
            scored = score_case(client, args.model, case, fetch=fetch, template=args.template)
            return to_record(case, scored, run_id, args.model, (time.perf_counter() - t0) * 1000)
        except Exception as exc:  # recorded, not fatal: the scorer reports coverage
            return base_record(case, run_id) | {"error": repr(exc)[:300], "pred": None}

    client = httpx.Client(timeout=600, limits=httpx.Limits(max_connections=max(4, args.workers * 2)))
    with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in tqdm(as_completed([pool.submit(one, c) for c in todo]), total=len(todo), unit="case", mininterval=10):
            out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
