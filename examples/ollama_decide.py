"""Ask s1 one question through a local Ollama and print each option's probability and the timings.

Standard library only. Builds the prompt exactly as training did (raw, starting with <bos>), so no quoting by hand:

    ollama pull jrlabs01/s1
    python3 examples/ollama_decide.py \\
        --state '{"ticket": "I was charged twice for order 4471."}' \\
        --question "Which team should handle this ticket?" \\
        --option billing="Payments and refunds" --option shipping=Deliveries --option tech="Bugs and outages"

Yes/no: --yesno instead of --option. Ordered scale: --levels low medium high. Probabilities are the model's raw ones;
the repository's API (api/server.py) adds the calibration fitted on dev data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEMPLATE = "<|turn>user\n{user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
NOUL = {"true": "Yes, the condition holds.", "false": "No, the condition does not hold."}


def options_for(question: dict) -> list[tuple[str, str]]:
    """(key, text) per option, in presented order (same layout as examples/quickstart.py)."""
    if question["type"] == "score":
        return [(str(i), text) for i, text in enumerate(question["criteria"])]
    out = []
    for key, desc in question["criteria"].items():
        if not desc or desc == key:
            out.append((key, key))
        elif len(key) == 1:
            out.append((key, desc))
        else:
            out.append((key, f"{key}: {desc}"))
    return out


def prompt_for(state, question: dict, options: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"{LETTERS[i]}) {text}" for i, (_, text) in enumerate(options))
    user = ("You are a decision model. Read the state and answer the question by choosing one option.\n\n"
            f"STATE:\n{json.dumps(state, ensure_ascii=False, indent=1)}\n\nQUESTION: {question['instructions']}\n\n"
            f"OPTIONS:\n{lines}\n\nAnswer with the option letter only.")
    return "<bos>" + TEMPLATE.format(user=user) + "Answer:"


def option_probs(top_logprobs: list[dict], options: list[tuple[str, str]]) -> tuple[dict[str, float], float]:
    """Probability of each option key from Ollama's top_logprobs, renormalised over the option letters, and the
    probability mass the letters had. Ollama lists several spellings of a letter (" A" and "A"); entries come most
    likely first, and only the first spelling of each letter is kept, as the repository's scorer does."""
    top: dict[str, float] = {}
    for t in top_logprobs:
        tok = t["token"].strip()
        if tok and tok not in top:
            top[tok] = t["logprob"]
    raw = {key: math.exp(top.get(LETTERS[i], -100.0)) for i, (key, _) in enumerate(options)}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}, z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, help="JSON (or plain text) the decision is about")
    ap.add_argument("--question", required=True, help="what to decide")
    kind = ap.add_mutually_exclusive_group(required=True)
    kind.add_argument("--option", action="append", metavar="KEY=DESCRIPTION", help="a choice option (repeat, 2-26)")
    kind.add_argument("--yesno", action="store_true", help="a yes/no question")
    kind.add_argument("--levels", nargs="+", metavar="LEVEL", help="an ordered scale, lowest first")
    ap.add_argument("--model", default="jrlabs01/s1")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    args = ap.parse_args()

    try:
        state = json.loads(args.state)
    except json.JSONDecodeError:
        state = args.state
    if args.yesno:
        question = {"type": "noul", "instructions": args.question, "criteria": NOUL}
    elif args.levels:
        question = {"type": "score", "instructions": args.question, "criteria": args.levels}
    else:
        pairs = [o.split("=", 1) if "=" in o else (o, "") for o in args.option]
        question = {"type": "choice", "instructions": args.question, "criteria": dict(pairs)}
    options = options_for(question)
    if not 2 <= len(options) <= 26:
        sys.exit("give 2 to 26 options")

    body = {"model": args.model, "raw": True, "stream": False, "logprobs": True, "top_logprobs": 20,
            "prompt": prompt_for(state, question, options), "options": {"temperature": 0, "num_predict": 1}}
    host = args.host if args.host.startswith("http") else f"http://{args.host}"
    req = urllib.request.Request(f"{host}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            r = json.load(resp)
    except urllib.error.HTTPError as exc:
        sys.exit(f"Ollama error {exc.code}: {exc.read().decode(errors='replace')}")
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach Ollama at {host} ({exc.reason}); is it running? Try: ollama pull {args.model}")
    if "error" in r or not r.get("logprobs"):
        sys.exit(f"Ollama did not return probabilities: {r.get('error', r)}")

    probs, z = option_probs(r["logprobs"][0]["top_logprobs"], options)
    best = max(probs, key=probs.get)
    width = max(len(k) for k in probs)
    for key, p in sorted(probs.items(), key=lambda kv: -kv[1]) if question["type"] != "score" else probs.items():
        label = key if question["type"] != "score" else f"{key} {question['criteria'][int(key)]}"
        print(f"{label:<{width + 12}} {p:6.3f}  {'#' * round(p * 30)}")
    if question["type"] == "noul":
        print(f"\nanswer: {'yes' if best == 'true' else 'no'} (P(yes) = {probs['true']:.3f})")
    elif question["type"] == "score":
        print(f"\nanswer: {question['criteria'][int(best)]} (expected level {sum(int(k) * p for k, p in probs.items()):.2f})")
    else:
        print(f"\nanswer: {best} ({probs[best]:.3f})")
    print(f"probability mass on the option letters: {z:.3f}")
    ms = lambda k: r.get(k, 0) / 1e6  # noqa: E731
    print(f"time: total {ms('total_duration'):.0f} ms (load {ms('load_duration'):.0f} ms, "
          f"prompt {r.get('prompt_eval_count', 0)} tokens in {ms('prompt_eval_duration'):.0f} ms, "
          f"answer {ms('eval_duration'):.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
