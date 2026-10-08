"""Minimal, self-contained s1 inference with Hugging Face transformers: one forward pass per question.

    pip install "transformers>=5.18" torch accelerate
    python examples/quickstart.py

Needs about 52 GB of accelerator memory in bf16 (one 80-96 GB GPU, or device_map="auto" across several). With less
memory, device_map="auto" offloads layers and the forward pass fails unless you also pass offload_folder="..."
(slow); use a quantised build instead. The
model returns a probability for every option; these raw probabilities are already reasonably calibrated, and the
repo's API applies per-type calibrators fitted on dev data (api/calibration_s1.json).
"""

from __future__ import annotations

import json

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "j-raghavan/s1-gemma4-26b-decision"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEMPLATE = "<|turn>user\n{user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
NOUL_DEFAULT = {"true": "Yes, the condition holds.", "false": "No, the condition does not hold."}


def options_for(question: dict) -> list[tuple[str, str]]:
    """(key, text) per option, in presented order. Score keys are level indices."""
    if question["type"] == "score":
        return [(str(i), text) for i, text in enumerate(question["criteria"])]
    criteria = question.get("criteria") or (NOUL_DEFAULT if question["type"] == "noul" else {})
    out = []
    for key, desc in criteria.items():
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
    # <bos> is required: Gemma 4's tokenizer does not add it, and accuracy drops without it.
    return "<bos>" + TEMPLATE.format(user=user) + "Answer:"


@torch.no_grad()
def decide(model, tok, state, question: dict) -> dict[str, float]:
    """Probability of each option key (2-26 options; the repo's runners handle more with two-digit codes)."""
    options = options_for(question)
    if not 2 <= len(options) <= 26:
        raise ValueError("this example handles 2-26 options")
    enc = tok(prompt_for(state, question, options), return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(**enc, logits_to_keep=1).logits[0, -1]
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS[: len(options)]]
    probs = torch.softmax(logits[letter_ids].float(), -1).tolist()
    return {key: p for (key, _), p in zip(options, probs)}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    # Gemma4ForConditionalGeneration, not a causal-LM class: the checkpoint stores the language model under the
    # multimodal wrapper, and loading it as a plain causal LM silently skips those weights.
    model = Gemma4ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()

    state = {"ticket": "I was charged twice for order 4471 and need the duplicate refunded."}
    questions = {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments and refunds", "shipping": "Deliveries", "tech": "Bugs and outages"}},
        "urgent": {"type": "noul", "instructions": "Does the customer report losing money?"},
        "severity": {"type": "score", "instructions": "How severe is the issue for the customer?",
                     "criteria": ["minor", "moderate", "serious", "critical"]},
    }
    for name, q in questions.items():
        probs = decide(model, tok, state, q)
        best = max(probs, key=probs.get)
        print(f"{name}: {best} ({probs[best]:.2f})  {json.dumps({k: round(v, 3) for k, v in probs.items()})}")


if __name__ == "__main__":
    main()
