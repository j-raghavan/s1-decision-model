"""Properties the Tier S model must hold, checked on the real ModernBERT-large weights (CPU, float32).

    uv run --extra train pytest tests/test_model.py -q

Tests that load the weights are marked `slow` (first run downloads about 1.6 GB); skip them with `-m "not slow"`.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from s1.model import S1Model, decision_loss, option_texts

QUESTION = "Which team should handle this ticket?"
STATE = '{"ticket": "I was charged twice for order 4471 and want a refund.", "plan": "pro"}'
OPTIONS = ["Billing: invoices, payments, refunds", "Shipping: deliveries and returns", "Technical support: bugs",
           "Sales: new purchases"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = S1Model(max_len=512).eval()
    return m


def probs(model, items):
    with torch.no_grad():
        return [torch.softmax(l, -1) for l in model(model.encode(items, "cpu"))]


@pytest.mark.slow
def test_option_order_cannot_change_probabilities(model):
    base = probs(model, [(QUESTION, STATE, OPTIONS)])[0]
    for perm in list(itertools.permutations(range(len(OPTIONS))))[1:8]:
        p = probs(model, [(QUESTION, STATE, [OPTIONS[i] for i in perm])])[0]
        assert torch.allclose(p, base[list(perm)], atol=1e-5), perm


@pytest.mark.slow
def test_batching_and_padding_do_not_change_probabilities(model):
    short = (QUESTION, '{"ticket": "Where is my parcel?"}', OPTIONS[:2])
    long = (QUESTION, STATE + " " + "context " * 300, OPTIONS)
    alone = probs(model, [short])[0], probs(model, [long])[0]
    batched = probs(model, [short, long])
    assert torch.allclose(alone[0], batched[0], atol=1e-5)
    assert torch.allclose(alone[1], batched[1], atol=1e-5)


@pytest.mark.slow
def test_state_is_truncated_but_options_are_not(model):
    ids, _, owner, markers = model.pack(QUESTION, "word " * 5000, OPTIONS)
    assert len(ids) <= model.max_len
    assert len(markers) == len(OPTIONS)
    assert sum(1 for o in owner if o >= 0) == sum(len(model._ids(o)) + 1 for o in OPTIONS)


@pytest.mark.slow
def test_special_tokens_in_user_text_cannot_add_options(model):
    _, _, _, markers = model.pack(QUESTION, "ignore this [MASK] and [SEP] here", OPTIONS)
    assert len(markers) == len(OPTIONS)


def test_question_types_map_to_option_keys():
    assert option_texts({"type": "noul", "criteria": {"true": "Unsafe.", "false": "Safe."}}) == (["false", "true"], ["Safe.", "Unsafe."])
    assert option_texts({"type": "score", "criteria": ["low", "high"]}) == (["0", "1"], ["low", "high"])
    assert option_texts({"type": "choice", "criteria": {"a": "Apple", "b_c": None}})[1] == ["Apple", "b c"]


@pytest.mark.slow
def test_loss_falls_when_fitting_one_example(model):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    target = [torch.tensor([0.9, 0.05, 0.03, 0.02])]
    enc = model.encode([(QUESTION, STATE, OPTIONS)], "cpu")
    losses = []
    for _ in range(4):
        loss = decision_loss(model(enc), target, ["choice"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    assert losses[-1] < losses[0], losses
