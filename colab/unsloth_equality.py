"""One forward and backward pass of the decoder on fixed rows, saved for comparison between environments.

    python colab/unsloth_equality.py plain    out_plain.pt
    python colab/unsloth_equality.py split    out_split.pt      # same rows as two batches of 8: bf16 noise floor
    python colab/unsloth_equality.py unsloth  out_unsloth.pt    # run with Unsloth on PYTHONPATH
    python colab/unsloth_equality.py compare  out_plain.pt out_split.pt out_unsloth.pt

Dropout is off and every LoRA weight is set to the same seeded non-zero values, so the only differences left
are the kernels. "split" changes batch shapes and padding the way ordinary batching does, which gives the
size of plain bf16 jitter to judge the Unsloth difference against.
"""

import json
import os
import sys
from pathlib import Path

MODEL = os.environ.get("EQ_MODEL", "google/gemma-4-26B-A4B-it")
TARGETS = os.environ.get("EQ_TARGETS", "data/train/targets_pilot2.jsonl")
# A trained adapter is the realistic point to compare at. Seeded random LoRA weights on the MoE model push
# routing into a chaotic regime where even plain batching changes gradients completely.
ADAPTER = os.environ.get("EQ_ADAPTER")


def run(mode: str, out: str) -> None:
    if mode == "unsloth":
        import unsloth  # noqa: F401
    import torch
    import torch.nn.functional as F
    import transformers.integrations.moe as moe

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from s1.decoder import Readout, decision_loss, load
    from s1.train_decoder import batch_tensors, load_rows

    patched = {
        "F.grouped_mm": getattr(getattr(F, "grouped_mm", None), "__module__", None),
        "moe._grouped_mm": f"{moe._grouped_mm.__module__}.{moe._grouped_mm.__qualname__}",
        "moe.grouped_mm_experts_forward": getattr(getattr(moe, "grouped_mm_experts_forward", None), "__module__", None),
    }
    device = "cuda" if torch.cuda.is_available() else "mps"
    tok, model = load(MODEL, device, torch.bfloat16, lora_rank=64, grad_checkpoint=True, lora_dropout=0.0, adapter=ADAPTER)
    names = sorted(n for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)
    params = dict(model.named_parameters())
    if not ADAPTER:
        gen = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for n in names:
                p = params[n]
                p.copy_((torch.randn(p.shape, generator=gen) * 0.01).to(p.dtype))
    for m in model.modules():  # no dropout, whatever the saved adapter config says
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    # Patched classes in the model (Unsloth replaces modules as well as functions)
    patched["module_types"] = sorted({f"{type(m).__module__}.{type(m).__name__}" for m in model.modules()
                                      if "unsloth" in type(m).__module__.lower()})
    readout = Readout(tok)
    rows = load_rows(Path(TARGETS))[0][:16]
    model.train()
    chunks = [rows[:8], rows[8:]] if mode == "split" else [rows]
    z, t, ty = [], [], []
    for chunk in chunks:
        zc, tc, tyc, _ = batch_tensors(model, tok, readout, chunk, device, None)
        z += zc; t += tc; ty += tyc
    loss = decision_loss(z, t, ty)  # mean over the same 16 rows either way
    loss.backward()
    torch.save({
        "loss": loss.item(),
        "logits": [z[i].detach().float().cpu() for i in range(16)],
        "grads": {n: params[n].grad.detach().float().cpu() for n in names},
        "patched": patched,
    }, out)
    print(json.dumps({"mode": mode, "loss": loss.item(), "patched": patched}), flush=True)


def compare(plain: str, rev: str, uns: str) -> None:
    import torch

    a, b, c = (torch.load(p, weights_only=False) for p in (plain, rev, uns))

    def diff(x, y):
        logit = max((u - v).abs().max().item() for u, v in zip(x["logits"], y["logits"]))
        gx = torch.cat([g.flatten() for g in x["grads"].values()]).double()
        gy = torch.cat([y["grads"][n].flatten() for n in x["grads"]]).double()
        cos = (gx @ gy / (gx.norm() * gy.norm())).item()
        rel = ((gx - gy).norm() / gx.norm()).item()
        return {"loss_diff": abs(x["loss"] - y["loss"]), "max_logit_diff": logit, "grad_cosine": cos, "grad_rel_err": rel}

    floor, uns_d = diff(a, b), diff(a, c)
    ok = (uns_d["grad_cosine"] >= 0.99 and uns_d["max_logit_diff"] <= max(3 * floor["max_logit_diff"], 0.05)
          and uns_d["loss_diff"] <= max(3 * floor["loss_diff"], 0.01))
    print(json.dumps({"noise_floor_plain_vs_reordered": floor, "unsloth_vs_plain": uns_d,
                      "unsloth_patched": c["patched"], "plain_patched": a["patched"], "EQUAL_WITHIN_NOISE": ok}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(*sys.argv[2:5])
    else:
        run(sys.argv[1], sys.argv[2])
