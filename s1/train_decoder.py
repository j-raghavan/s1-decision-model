"""LoRA-train a Gemma 4 decoder on soft decision targets, validating on held-out task families.

Every example is rendered with the teachers' prompt; options are reshuffled per example (except ordinal
scales) so the model cannot learn positional preferences. Only letter-labelled questions (2-26 options)
are trained; that covers all training rows. Validation reports seen-family and unseen-family loss and
accuracy, and the adapter with the lowest unseen-family loss is kept. Only the LoRA adapter is saved.

    python -m s1.train_decoder --model google/gemma-4-26B-A4B-it --targets data/train/targets_pilot.jsonl \\
        --out /content/adapter --max-steps 300 --batch 8 --accum 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

if os.environ.get("S1_UNSLOTH") == "1":  # opt-in: Unsloth's Triton grouped GEMM replaces the per-expert loop on sm_120
    import unsloth  # noqa: F401  (must be imported before transformers to patch it)
    # isort: split
    import transformers.integrations.moe as _moe
    print(json.dumps({"unsloth": getattr(unsloth, "__version__", "?"),
                      "grouped_mm_impl": f"{_moe._grouped_mm.__module__}.{_moe._grouped_mm.__qualname__}"}), flush=True)

from s1.decoder import Readout, decision_loss, last_logits, load, prompt_and_keys, shuffled_question


def device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def load_rows(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    train, val, unseen = [], [], []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        n = len(r["question"]["criteria"]) if r["task_type"] != "noul" else 2
        if n > 26:
            continue
        if r.get("holdout_family"):
            unseen.append(r)
        elif int(hashlib.sha256(r["row_id"].encode()).hexdigest()[:8], 16) % 50 == 0:  # stable 2% seen-family slice
            val.append(r)
        else:
            train.append(r)
    return train, val, unseen


def make_batches(rows: list[dict], batch: int, rng: random.Random, bucket: bool = True) -> list[list[int]]:
    """One epoch of batches, in random order. With bucketing, rows are sorted by length inside random windows of
    64 batches, so each batch pads to a similar length while the order across batches stays random."""
    order = list(range(len(rows)))
    rng.shuffle(order)
    if bucket:
        size = lambda i: len(json.dumps(rows[i]["state"])) + len(json.dumps(rows[i]["question"]))  # noqa: E731
        window = 64 * batch
        order = [i for w in range(0, len(order), window) for i in sorted(order[w:w + window], key=size)]
    out = [order[i:i + batch] for i in range(0, len(order) - batch + 1, batch)]
    rng.shuffle(out)
    return out


def token_lengths(tok, rows: list[dict]) -> list[int]:
    """Prompt length in tokens of every row, as rendered for training.

    One token short of the forward pass: Gemma 4's tokenizer adds nothing with add_special_tokens=True, while
    last_logits prepends <bos>. So --max-row-tokens admits rows one token longer, and a batch can exceed
    --max-batch-tokens by at most one token per row. Kept as is because the released model was trained with it;
    changing it would change which rows are dropped and how batches split."""
    prompts = [prompt_and_keys({"state": r["state"], "question": r["question"], "task_type": r["task_type"]})[0] for r in rows]
    lengths: list[int] = []
    for i in range(0, len(prompts), 2048):
        lengths += [len(ids) for ids in tok(prompts[i:i + 2048], add_special_tokens=True)["input_ids"]]
    return lengths


def split_by_tokens(idx: list[int], lengths: list[int], max_tokens: int) -> list[list[int]]:
    """Consecutive pieces of one batch whose padded size (rows x longest row) stays within max_tokens.

    A full epoch eventually puts the longest rows into one length-bucketed batch; on the 26B model a 47K-token
    batch ran out of GPU memory, while 32K-token batches fit. Pieces keep the rows' order, so the per-row option
    shuffles draw the same random numbers as an unsplit batch."""
    pieces: list[list[int]] = [[]]
    for i in idx:
        cand = pieces[-1] + [i]
        if pieces[-1] and len(cand) * max(lengths[j] for j in cand) > max_tokens:
            pieces.append([i])
        else:
            pieces[-1] = cand
    return pieces


def batch_tensors(model, tok, readout: Readout, rows: list[dict], device: str, rng: random.Random | None):
    cases, targets, types = [], [], []
    for r in rows:
        q = shuffled_question(r["question"], rng) if rng else r["question"]
        case = {"state": r["state"], "question": q, "task_type": r["task_type"]}
        _, labels, keys = prompt_and_keys(case)
        cases.append(case)
        targets.append(torch.tensor([r["target"][k] for k in keys], device=device))
        types.append(r["task_type"])
    tok.padding_side = "right"
    logits = last_logits(model, tok, [prompt_and_keys(c)[0] for c in cases], device)
    rows_logits = [logits[i, readout.letter_ids[: len(t)]] for i, t in enumerate(targets)]
    gold = []
    for r, c in zip(rows, cases):
        _, _, keys = prompt_and_keys(c)
        g = r["gold"]
        gold.append(keys.index("true" if g is True else "false" if g is False else str(g)))
    return rows_logits, targets, types, gold


@torch.no_grad()
def evaluate(model, tok, readout, rows, device, batch) -> dict:
    model.eval()
    losses, by_fam = [], {}
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        z, t, ty, gold = batch_tensors(model, tok, readout, chunk, device, None)
        losses.append(decision_loss(z, t, ty).item() * len(chunk))
        for r, zz, g in zip(chunk, z, gold):
            by_fam.setdefault(r["family"], []).append(float(int(zz.argmax()) == g))
    model.train()
    accs = {f: sum(v) / len(v) for f, v in by_fam.items()}
    return {"loss": sum(losses) / max(1, len(rows)), "acc_macro": sum(accs.values()) / max(1, len(accs))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-limit", type=int, default=800, help="cap rows per validation set to keep evals short")
    ap.add_argument("--no-grad-checkpoint", action="store_true", help="faster when activations fit in memory")
    ap.add_argument("--no-bucket", action="store_true", help="random batches instead of length-bucketed ones")
    ap.add_argument("--max-row-tokens", type=int, default=2048, help="training rows longer than this are dropped")
    ap.add_argument("--max-batch-tokens", type=int, default=24576,
                    help="a batch whose padded size exceeds this is run in pieces (same gradient, less memory)")
    ap.add_argument("--skip-initial-eval", action="store_true", help="no step-0 baseline (speed tests)")
    ap.add_argument("--no-save", action="store_true", help="save no adapters (speed tests)")
    ap.add_argument("--save-every", type=int, default=0, help="also save the adapter as step-N every N steps, for dev-set selection")
    ap.add_argument("--init-adapter", default=None, help="continue from a saved adapter (fresh optimizer), e.g. after a lost VM")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="training time budget: after 10 steps the step count shrinks to fit, so the schedule still completes")
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device, dtype = device_and_dtype()
    tok, model = load(args.model, device, dtype, lora_rank=args.rank, grad_checkpoint=not args.no_grad_checkpoint,
                      lora_dropout=args.lora_dropout, adapter=args.init_adapter)
    model.print_trainable_parameters()
    readout = Readout(tok)
    train, val, unseen = load_rows(args.targets)
    rng.shuffle(val), rng.shuffle(unseen)
    val, unseen = val[: args.val_limit], unseen[: args.val_limit]
    print(json.dumps({"device": device, "train": len(train), "val": len(val), "unseen": len(unseen),
                      "steps": args.max_steps, "batch": args.batch, "accum": args.accum}), flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    # Warm-up is never under 10 steps: a probe on Gemma 4 E2B lost 12 points of accuracy in 20 steps with a
    # 1-step warm-up and gained 19 with a 10-step one (same LR). The plan is mutable: --max-minutes may shrink it.
    plan = {"steps": args.max_steps, "warm": max(10, args.max_steps // 20)}
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / plan["warm"] if s < plan["warm"] else
                                              max(0.0, (plan["steps"] - s) / max(1, plan["steps"] - plan["warm"])))
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.jsonl").open("a", encoding="utf-8")

    best = float("inf")
    if not args.skip_initial_eval:
        ev0 = evaluate(model, tok, readout, unseen, device, args.batch)  # zero-shot baseline on held-out families
        rec = {"step": 0, "unseen_loss": ev0["loss"], "unseen_acc_macro": ev0["acc_macro"]}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        best = ev0["loss"]

    model.train()
    lengths = token_lengths(tok, train)
    keep = [i for i, n in enumerate(lengths) if n <= args.max_row_tokens]
    rec = {"rows_over_max_row_tokens_dropped": len(train) - len(keep), "max_row_tokens": args.max_row_tokens,
           "max_batch_tokens": args.max_batch_tokens}
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    train, lengths = [train[i] for i in keep], [lengths[i] for i in keep]
    t0, step, running = time.time(), 0, []
    batches = make_batches(train, args.batch, rng, bucket=not args.no_bucket)
    while step < plan["steps"]:
        for _ in range(args.accum):
            if not batches:
                batches = make_batches(train, args.batch, rng, bucket=not args.no_bucket)
            idx = batches.pop()
            batch_loss = 0.0
            for piece in split_by_tokens(idx, lengths, args.max_batch_tokens):
                z, t, ty, _ = batch_tensors(model, tok, readout, [train[i] for i in piece], device, rng)
                # each piece's mean loss weighted by its share of the batch: the sum equals the batch mean
                loss = decision_loss(z, t, ty) * len(piece) / len(idx)
                (loss / args.accum).backward()
                batch_loss += loss.item()
            running.append(batch_loss)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step == 10 and args.max_minutes:
            per_step = (time.time() - t0) / 10
            fit = int(args.max_minutes * 60 * 0.85 / per_step)  # 15% held back for the periodic evaluations
            plan["steps"] = min(plan["steps"], max(20, fit))  # never more than asked for
            plan["warm"] = max(10, plan["steps"] // 20)  # warm-up stays 5% of the steps actually run
            rec = {"step": step, "sec_per_step": round(per_step, 2), "planned_steps": plan["steps"]}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 10 == 0:
            rec = {"step": step, "loss": sum(running) / len(running), "elapsed_s": round(time.time() - t0)}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            running = []
        if args.save_every and step % args.save_every == 0 and not args.no_save:
            model.save_pretrained(args.out / f"step-{step}")
        if step % args.eval_every == 0 or step == plan["steps"]:
            ev_s = evaluate(model, tok, readout, val, device, args.batch)
            ev_u = evaluate(model, tok, readout, unseen, device, args.batch)
            rec = {"step": step, "val_loss": ev_s["loss"], "val_acc_macro": ev_s["acc_macro"],
                   "unseen_loss": ev_u["loss"], "unseen_acc_macro": ev_u["acc_macro"]}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            if ev_u["loss"] < best:
                best = ev_u["loss"]
                if not args.no_save:
                    model.save_pretrained(args.out / "best")
    if not args.no_save:
        model.save_pretrained(args.out / "last")
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(json.dumps({"done": True, "steps": step, "best_unseen_loss": best, "elapsed_s": round(time.time() - t0),
                      "peak_mem_gb": round(peak, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
