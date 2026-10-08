"""The Tier S decision model: ModernBERT-large with order-invariant option-marker scoring.

One forward pass answers a question over any number of options:

    [CLS] question [SEP] state [SEP] [MASK] option_1 [MASK] option_2 ... [MASK] option_K

Each option's logit is read from the encoder output at its [MASK] marker. Two
constructions make that logit a function of (question, state, that option) only,
so reordering the options cannot change any probability:

  - attention: prefix tokens attend only to the prefix; option tokens attend to
    the prefix and to their own option, never to another option
  - positions: every option's position ids restart at the prefix length, so each
    option sits at the same positions whatever its place in the sequence

Because the prefix never attends to options, its encoding is also shared by
every question over the same state. The design follows Von's independent-options
mode (github.com/wfzyx/von, Apache-2.0); this is an independent implementation,
and no Von weights or data are used.

Question types map onto option sets: choice uses its options, yes/no uses its
two criteria (false, true), score uses its ordered levels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

BASE_MODEL = "answerdotai/ModernBERT-large"


@dataclass
class Encoded:
    input_ids: torch.Tensor  # (B, T)
    position_ids: torch.Tensor  # (B, T)
    masks: dict[str, torch.Tensor]  # layer type -> additive mask (B, 1, T, T)
    marker_positions: list[list[int]]  # per sample, index of each option's [MASK]


def option_texts(question: dict) -> tuple[list[str], list[str]]:
    """(option keys, option texts) in the question's own order. Yes/no keys are ["false", "true"]."""
    crit = question.get("criteria")
    if question["type"] == "noul":
        crit = crit if isinstance(crit, dict) else {}
        return ["false", "true"], [crit.get("false") or "No, the condition does not hold.",
                                   crit.get("true") or "Yes, the condition holds."]
    if question["type"] == "score":
        return [str(i) for i in range(len(crit))], [str(level) for level in crit]
    keys = list(crit)
    texts = [desc if desc and desc != key else key.replace("_", " ") for key, desc in crit.items()]
    return keys, texts


def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(", ", ": "))


class OptionScorer(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class S1Model(nn.Module):
    def __init__(self, base: str = BASE_MODEL, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base, attn_implementation="sdpa")
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        cfg = self.encoder.config
        self.window = cfg.sliding_window
        self.layer_types = set(cfg.layer_types)
        self.max_len = max_len
        self.scorer = OptionScorer(cfg.hidden_size, dropout)
        t = self.tokenizer
        self.cls_id, self.sep_id, self.mask_id, self.pad_id = t.cls_token_id, t.sep_token_id, t.mask_token_id, t.pad_token_id
        self.special = {self.cls_id, self.sep_id, self.mask_id, self.pad_id}

    # ---------------------------------------------------------------- packing

    def _ids(self, text: str) -> list[int]:
        # User text must not forge structure: any special id inside it is dropped.
        return [i for i in self.tokenizer(text, add_special_tokens=False)["input_ids"] if i not in self.special]

    def pack(self, question: str, state: str, options: list[str]) -> tuple[list[int], list[int], list[int], list[int]]:
        """Token ids, position ids, option id per token (-1 prefix), marker indices. The state is truncated, never the options."""
        q_ids, s_ids = self._ids(question), self._ids(state)
        opt_ids = [[self.mask_id] + self._ids(o) for o in options]
        budget = self.max_len - 3 - len(q_ids) - sum(len(o) for o in opt_ids)
        if budget < 16:
            raise ValueError(f"question and options alone exceed max_len={self.max_len}")
        if len(s_ids) > budget:
            s_ids = s_ids[: budget // 2] + s_ids[-(budget - budget // 2):]  # keep the head and tail of long states
        prefix = [self.cls_id] + q_ids + [self.sep_id] + s_ids + [self.sep_id]
        ids, pos, owner, markers = list(prefix), list(range(len(prefix))), [-1] * len(prefix), []
        for k, o in enumerate(opt_ids):
            markers.append(len(ids))
            ids += o
            pos += list(range(len(prefix), len(prefix) + len(o)))
            owner += [k] * len(o)
        return ids, pos, owner, markers

    def encode(self, items: list[tuple[str, str, list[str]]], device: torch.device | str) -> Encoded:
        packed = [self.pack(q, s, o) for q, s, o in items]
        T = max(len(p[0]) for p in packed)
        B = len(packed)
        ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        pos = torch.zeros((B, T), dtype=torch.long)
        owner = torch.full((B, T), -2, dtype=torch.long)  # -2 marks padding
        for b, (i, p, o, _) in enumerate(packed):
            ids[b, : len(i)] = torch.tensor(i)
            pos[b, : len(p)] = torch.tensor(p)
            owner[b, : len(o)] = torch.tensor(o)
        oi, oj = owner.unsqueeze(2), owner.unsqueeze(1)
        allowed = ((oj == -1) | (oi == oj)) & (oj != -2) & (oi != -2)
        allowed |= torch.eye(T, dtype=torch.bool).unsqueeze(0)  # padding rows attend to themselves (avoids NaN)
        local = (pos.unsqueeze(2) - pos.unsqueeze(1)).abs() <= self.window
        dtype = next(self.encoder.parameters()).dtype
        neg = torch.finfo(dtype).min

        def additive(m: torch.Tensor) -> torch.Tensor:
            return torch.zeros(m.shape, dtype=dtype).masked_fill(~m, neg).unsqueeze(1).to(device)

        masks = {"full_attention": additive(allowed),
                 "sliding_attention": additive((allowed & local) | torch.eye(T, dtype=torch.bool).unsqueeze(0))}
        return Encoded(ids.to(device), pos.to(device), {k: v for k, v in masks.items() if k in self.layer_types},
                       [p[3] for p in packed])

    # ---------------------------------------------------------------- forward

    def forward(self, enc: Encoded) -> list[torch.Tensor]:
        hidden = self.encoder(input_ids=enc.input_ids, attention_mask=enc.masks, position_ids=enc.position_ids).last_hidden_state
        return [self.scorer(hidden[b, m]) for b, m in enumerate(enc.marker_positions)]

    @torch.no_grad()
    def predict(self, rows: list[dict], device: torch.device | str) -> list[dict[str, float]]:
        """Probabilities over each row's option keys; rows carry `state` and `question`."""
        self.eval()
        items, keys = [], []
        for r in rows:
            k, texts = option_texts(r["question"])
            items.append((r["question"]["instructions"], state_text(r["state"]), texts))
            keys.append(k)
        logits = self(self.encode(items, device))
        return [dict(zip(k, torch.softmax(l.float(), -1).tolist())) for k, l in zip(keys, logits)]


def decision_loss(logits: list[torch.Tensor], targets: list[torch.Tensor], types: list[str],
                  brier_weight: float = 0.5, rps_weight: float = 0.5) -> torch.Tensor:
    """Soft-target cross-entropy + Brier, plus ranked probability score for ordinal (score) questions.

    All three are strictly proper scoring rules, so the loss is minimised by honest probabilities."""
    total = []
    for z, t, typ in zip(logits, targets, types):
        logp = torch.log_softmax(z.float(), -1)
        p = logp.exp()
        loss = -(t * logp).sum() + brier_weight * ((p - t) ** 2).sum()
        if typ == "score" and len(t) > 1:
            loss = loss + rps_weight * ((p.cumsum(0) - t.cumsum(0)) ** 2).sum() / (len(t) - 1)
        total.append(loss)
    return torch.stack(total).mean()
