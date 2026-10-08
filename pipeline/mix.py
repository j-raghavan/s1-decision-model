"""Rebalance training targets so the skill being taught is not drowned out by large families.

    uv run pipeline/mix.py --inputs data/train/targets_v2.jsonl data/train/targets_v3_syn.jsonl \\
        --out data/train/targets_v3.jsonl --cap-v1 3000 --cap-sni 400 --synthetic-repeat 2

Caps rows per family (v1 families and SNI tasks separately), keeps every synthetic row and repeats it
--synthetic-repeat times (each repeat gets its own row_id), and leaves held-out families untouched so
unseen-family validation is unchanged.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap-v1", type=int, default=3000)
    ap.add_argument("--cap-sni", type=int, default=400)
    ap.add_argument("--synthetic-repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261007)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_family: dict[str, list[str]] = collections.defaultdict(list)
    for path in args.inputs:
        for line in path.open(encoding="utf-8"):
            by_family[json.loads(line)["family"]].append(line)
    out, share = [], collections.Counter()
    for fam, lines in by_family.items():
        rows = [json.loads(l) for l in lines]
        if rows[0].get("holdout_family"):
            out += rows
            share["held-out"] += len(rows)
            continue
        rng.shuffle(rows)
        if fam.startswith("syn_"):
            for k in range(args.synthetic_repeat):
                out += [r if k == 0 else r | {"row_id": f"{r['row_id']}~{k}"} for r in rows]
            share["synthetic"] += len(rows) * args.synthetic_repeat
        elif fam.startswith("sni_"):
            out += rows[: args.cap_sni]
            share["sni"] += min(len(rows), args.cap_sni)
        else:
            out += rows[: args.cap_v1]
            share["v1"] += min(len(rows), args.cap_v1)
    rng.shuffle(out)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    trainable = sum(v for k, v in share.items() if k != "held-out")
    print(f"{len(out):,} rows -> {args.out}")
    for k, v in share.items():
        print(f"  {k:9s} {v:7,d}" + (f"  ({v / trainable:.0%} of training)" if k != "held-out" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
