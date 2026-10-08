"""The encoder (s1/model.py, earlier approach) and the decoder (s1/decoder.py, released model) implement the same
proper-scoring loss separately; at the encoder's default weights they must agree, so the two cannot drift silently."""

import torch

from s1 import decoder, model


def test_encoder_and_decoder_losses_agree():
    g = torch.Generator().manual_seed(0)
    logits, targets, types = [], [], []
    for n, typ in ((2, "noul"), (3, "choice"), (5, "choice"), (4, "score"), (7, "score")):
        logits.append(torch.randn(n, generator=g))
        targets.append(torch.softmax(torch.randn(n, generator=g), -1))
        types.append(typ)
    a = decoder.decision_loss(logits, targets, types)
    b = model.decision_loss(logits, targets, types)
    assert torch.allclose(a, b, atol=1e-6), (a.item(), b.item())
