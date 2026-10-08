"""Decision model on a Gemma 4 decoder: one forward pass, read the option-letter logits.

The prompt is exactly the one the teachers were scored with (eval/jevbench/run_ollama.py: Gemma 4 turn
format, lettered options, reply prefilled with "Answer:"), so a fine-tuned model is measured the same way
as the zero-shot teachers. The decision distribution is a softmax over the option-letter tokens at the
last position: no generation, no parsing.

Questions with more than 26 options use two-digit codes, read in two passes (first digit, then second
digit given the first), as the teachers were.

Training uses LoRA on the attention and dense-MLP projections. Gemma 4 26B-A4B stores its 128 experts
as fused 3-D tensors, which standard LoRA cannot adapt, so the experts and router stay frozen.
"""

from __future__ import annotations

import math
import random
import string
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "jevbench"))
from run_ollama import labels_for, options_for, render  # noqa: E402

LETTERS = list(string.ascii_uppercase)
# Language-model projections only; the vision and audio towers use the same module names.
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load(model_id: str, device: str, dtype: torch.dtype, lora_rank: int = 0, adapter: str | None = None,
         grad_checkpoint: bool = False, experts_impl: str | None = None, attn_impl: str | None = None,
         lora_dropout: float = 0.05):
    """Gemma 4 run on text only, optionally with a new or saved LoRA adapter.

    The released checkpoints store the language model under model.language_model.*, which the text-only
    Gemma4ForCausalLM does not map (it silently initialises random weights), so the full
    Gemma4ForConditionalGeneration is loaded and the vision and audio towers simply go unused.
    """
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(model_id)
    kernels = {k: v for k, v in (("experts_implementation", experts_impl), ("attn_implementation", attn_impl)) if v}
    model, info = Gemma4ForConditionalGeneration.from_pretrained(model_id, dtype=dtype, output_loading_info=True, **kernels)
    missing = [k for k in info["missing_keys"] if "language_model" in k or k.startswith("lm_head")]
    if missing:
        raise RuntimeError(f"{len(missing)} language-model weights missing from {model_id}, e.g. {missing[:3]}")
    if grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=lora_rank > 0)
    elif lora_rank:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=lora_dropout,
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    return tok, model.to(device)


class Readout:
    """Token ids for option labels: ' A'..' Z' after 'Answer:', and digits after 'Answer: '."""

    def __init__(self, tok):
        self.letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]
        self.digit_ids = {d: tok.encode(d, add_special_tokens=False)[-1] for d in "0123456789"}
        assert all(len(tok.encode(" " + L, add_special_tokens=False)) == 1 for L in LETTERS)


def shuffled_question(question: dict, rng: random.Random) -> dict:
    """Same question with its options in a new order (ordinal scales keep their order)."""
    if question["type"] == "score":
        return question
    items = list(question["criteria"].items())
    rng.shuffle(items)
    return {**question, "criteria": dict(items)}


def prompt_and_keys(case: dict) -> tuple[str, list[str], list[str]]:
    options = options_for(case["question"])
    labels = labels_for(len(options))
    return render(case, labels, options, "gemma4"), labels, [k for k, _ in options]


def last_logits(model, tok, prompts: list[str], device: str, add_bos: bool = True) -> torch.Tensor:
    """Logits at each prompt's final position (right padding, so the index is length - 1).

    Only the final positions are projected onto the 262K-token vocabulary (logits_to_keep), which keeps
    the model's own final-logit softcapping and saves gigabytes per batch over full-sequence logits.
    """
    # Gemma 4's tokenizer does not add <bos> by itself (its chat template writes it out), and the model degrades
    # without it, so it is prepended here when missing.
    bos = (tok.bos_token or "") if add_bos else ""
    prompts = [p if p.startswith(bos) else bos + p for p in prompts]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    last = enc["attention_mask"].sum(dim=1) - 1
    keep = torch.unique(last)
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], logits_to_keep=keep)
    col = torch.searchsorted(keep, last)
    return out.logits[torch.arange(len(prompts), device=device), col]


def option_log_probs(model, tok, readout: Readout, cases: list[dict], device: str,
                     add_bos: bool = True) -> list[dict[str, float]]:
    """Log-probabilities over each case's option keys. Letter questions take one batched pass."""
    tok.padding_side = "right"
    built = [prompt_and_keys(c) for c in cases]
    out: list[dict[str, float] | None] = [None] * len(cases)
    letter_idx = [i for i, (_, labels, _) in enumerate(built) if labels[0].isalpha()]
    if letter_idx:
        logits = last_logits(model, tok, [built[i][0] for i in letter_idx], device, add_bos)
        for row, i in zip(logits, letter_idx):
            _, labels, keys = built[i]
            lp = torch.log_softmax(row[readout.letter_ids[: len(labels)]].float(), -1)
            out[i] = dict(zip(keys, lp.tolist()))
    for i, (prompt, labels, keys) in enumerate(built):
        if out[i] is not None:
            continue
        firsts = sorted({lab[0] for lab in labels})
        row = last_logits(model, tok, [prompt], device, add_bos)[0]
        lp_first = torch.log_softmax(row[[readout.digit_ids[d] for d in firsts]].float(), -1)
        lp: dict[str, float] = {}
        for d, lf in zip(firsts, lp_first.tolist()):
            seconds = [lab[1] for lab in labels if lab[0] == d]
            if lf < math.log(1e-3):
                for s in seconds:
                    lp[d + s] = lf - math.log(len(seconds))
                continue
            row2 = last_logits(model, tok, [prompt + d], device, add_bos)[0]
            lp2 = torch.log_softmax(row2[[readout.digit_ids[s] for s in seconds]].float(), -1)
            for s, l2 in zip(seconds, lp2.tolist()):
                lp[d + s] = lf + l2
        out[i] = {k: lp[lab] for lab, k in zip(labels, keys)}
    return out


def decision_loss(logits_rows: list[torch.Tensor], targets: list[torch.Tensor], types: list[str]) -> torch.Tensor:
    """Soft-target cross-entropy + Brier, plus ranked probability score for ordinal questions (all proper)."""
    losses = []
    for z, t, typ in zip(logits_rows, targets, types):
        logp = torch.log_softmax(z.float(), -1)
        p = logp.exp()
        loss = -(t * logp).sum() + 0.5 * ((p - t) ** 2).sum()
        if typ == "score" and len(t) > 1:
            loss = loss + 0.5 * ((p.cumsum(0) - t.cumsum(0)) ** 2).sum() / (len(t) - 1)
        losses.append(loss)
    return torch.stack(losses).mean()
