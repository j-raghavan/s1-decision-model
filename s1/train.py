"""Train the Tier S model on soft decision targets.

Runs the same way on a Mac (MPS, float32, for trial runs on a subset) and on a
CUDA GPU (bfloat16 autocast, for the full run). Rows are bucketed by length so
batches carry little padding; a fixed slice of every family is held out for
validation. Checkpoints are written every --save-every steps and on exit, and a
rerun with the same --out resumes from the latest one (Colab VMs can be
reclaimed without warning).

    # trial run on the Mac
    uv run --extra train python -m s1.train --targets data/train/targets_trial.jsonl --out checkpoints/trial \\
        --max-steps 200 --batch-tokens 4096

    # full run on a GPU
    python -m s1.train --targets data/train/targets_v1.jsonl --out /content/ckpt --epochs 3 --batch-tokens 32768
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from s1.model import S1Model, decision_loss, option_texts, state_text


def device_and_dtype() -> tuple[str, torch.dtype | None]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", None
    return "cpu", None


def is_val(row_id: str, frac: float) -> bool:
    return int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def load(path: Path, model: S1Model, val_frac: float) -> tuple[list[dict], list[dict], list[dict]]:
    """Train rows, seen-family validation rows (a hashed slice), and unseen-family rows (whole held-out families)."""
    train, val, unseen = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            keys, texts = option_texts(r["question"])
            target = [r["target"][k] for k in keys]
            ex = {"row_id": r["row_id"], "family": r["family"], "type": r["task_type"],
                  "item": (r["question"]["instructions"], state_text(r["state"]), texts), "target": target,
                  "gold_index": keys.index("true" if r["gold"] is True else "false" if r["gold"] is False else str(r["gold"]))}
            ex["length"] = len(model.pack(*ex["item"])[0])
            if r.get("holdout_family"):
                unseen.append(ex)
            else:
                (val if is_val(r["row_id"], val_frac) else train).append(ex)
    return train, val, unseen


def batches(examples: list[dict], batch_tokens: int, rng: random.Random) -> list[list[dict]]:
    """Length-bucketed batches whose padded size (rows x longest) stays under batch_tokens; shuffled order."""
    pool = sorted(examples, key=lambda e: e["length"] + rng.random())
    out, cur, longest = [], [], 0
    for e in pool:
        if cur and max(longest, e["length"]) * (len(cur) + 1) > batch_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(e)
        longest = max(longest, e["length"])
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def evaluate(model: S1Model, val: list[dict], device: str, dtype, batch_tokens: int) -> dict:
    model.eval()
    by_family: dict[str, list[float]] = {}
    losses = []
    with torch.no_grad():
        for b in batches(val, batch_tokens, random.Random(0)):
            with torch.autocast(device, dtype=dtype, enabled=dtype is not None):
                logits = model(model.encode([e["item"] for e in b], device))
            targets = [torch.tensor(e["target"], device=device) for e in b]
            losses.append(decision_loss(logits, targets, [e["type"] for e in b]).item() * len(b))
            for e, z in zip(b, logits):
                by_family.setdefault(e["family"], []).append(float(z.argmax().item() == e["gold_index"]))
    model.train()
    accs = {f: sum(v) / len(v) for f, v in by_family.items()}
    return {"val_loss": sum(losses) / max(1, len(val)), "val_acc_macro": sum(accs.values()) / max(1, len(accs)),
            "val_acc_by_family": accs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute encoder activations in backward; needed for multi-thousand-token rows")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start from this checkpoint's weights (fresh optimizer and schedule); ignored when resuming")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device, dtype = device_and_dtype()
    model = S1Model(max_len=args.max_len).to(device)
    if args.grad_checkpoint:
        model.encoder.gradient_checkpointing_enable()
    train, val, unseen = load(args.targets, model, args.val_frac)
    rng = random.Random(args.seed)
    steps_per_epoch = math.ceil(len(batches(train, args.batch_tokens, random.Random(0))) / args.accum)
    total = args.max_steps or int(steps_per_epoch * args.epochs)
    print(json.dumps({"device": device, "dtype": str(dtype), "train": len(train), "val": len(val), "unseen": len(unseen),
                      "steps_per_epoch": steps_per_epoch, "total_steps": total}), flush=True)

    enc_params = list(model.encoder.parameters())
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": model.scorer.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    warm = max(1, int(total * args.warmup))
    # Linear warm-up, then linear decay to zero.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "last.pt"
    step, best = 0, -1.0
    if args.init_from and not ckpt.exists():
        model.load_state_dict(torch.load(args.init_from, map_location=device)["model"])
        print(f"initialised from {args.init_from}", flush=True)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
        best = state.get("best", -1.0)
        rng.setstate(state["rng"])
        print(f"resumed from step {step}", flush=True)

    def save(tag: str = "last") -> None:
        """last.pt holds everything needed to resume; best.pt holds model weights only (about a third of the size)."""
        # Settings are stored as plain strings so the checkpoint loads under torch.load(weights_only=True).
        meta = {"step": step, "best": best, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
        payload = {"model": model.state_dict(), **meta}
        if tag == "last":
            payload |= {"opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.getstate()}
        tmp = args.out / f"{tag}.pt.tmp"
        torch.save(payload, tmp)
        tmp.replace(args.out / f"{tag}.pt")

    log = (args.out / "log.jsonl").open("a", encoding="utf-8")
    model.train()
    t0, running = time.time(), []
    while step < total:
        for i, b in enumerate(batches(train, args.batch_tokens, rng)):
            with torch.autocast(device, dtype=dtype, enabled=dtype is not None):
                logits = model(model.encode([e["item"] for e in b], device))
            targets = [torch.tensor(e["target"], device=device) for e in b]
            loss = decision_loss(logits, targets, [e["type"] for e in b]) / args.accum
            loss.backward()
            running.append(loss.item() * args.accum)
            if (i + 1) % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                rec = {"step": step, "loss": sum(running) / len(running), "lr": sched.get_last_lr()[0],
                       "elapsed_s": round(time.time() - t0)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                running = []
            if step % args.eval_every == 0 or step == total:
                ev = evaluate(model, val, device, dtype, args.batch_tokens)
                ev_u = evaluate(model, unseen, device, dtype, args.batch_tokens) if unseen else None
                rec = {"step": step, "val_loss": ev["val_loss"], "val_acc_macro": ev["val_acc_macro"]}
                if ev_u:
                    rec |= {"unseen_loss": ev_u["val_loss"], "unseen_acc_macro": ev_u["val_acc_macro"]}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps({"step": step} | ev | ({"unseen": ev_u} if ev_u else {})) + "\n")
                log.flush()
                # Select on loss over held-out families when there are any (generalization to unseen instructions),
                # else on seen-family loss. Loss, not accuracy: as a proper scoring rule it also catches the creeping
                # overconfidence of later epochs that accuracy cannot see.
                score = ev_u["val_loss"] if ev_u else ev["val_loss"]
                if best < 0 or score < best:
                    best = score
                    save("best")
            if step % args.save_every == 0:
                save()
            if step >= total:
                break
    save()
    log.flush()
    print(json.dumps({"done": True, "steps": step, "best_selection_loss": best, "elapsed_s": round(time.time() - t0)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
