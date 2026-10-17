"""Build prediction runs that average each case over option orders, from colab/session_order_check.py's output.

Each input row holds one case's option probabilities under up to three orders (as given, reversed, rotated by
half). A variant keeps the first n orders and combines them with the arithmetic mean of the probabilities or the
geometric mean (mean log-probability, renormalised); n = 1 is the ordinary single-order run.

    uv run eval/jevbench/order_average.py results/raw/orders/jevbench.jsonl results/raw/jevbench

writes <out>/orders-n{1,2,3}-{arith,geo}/predictions.jsonl in the shared prediction format, for score.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_ollama import to_record

VARIANTS = [(1, "arith"), (2, "arith"), (2, "geo"), (3, "arith"), (3, "geo")]


def combine(dists: list[dict[str, float]], how: str) -> dict[str, float]:
    keys = list(dists[0])
    if how == "arith":
        avg = {k: sum(d[k] for d in dists) / len(dists) for k in keys}
    else:
        avg = {k: math.exp(sum(math.log(max(d[k], 1e-15)) for d in dists) / len(dists)) for k in keys}
    z = sum(avg.values())
    return {k: v / z for k, v in avg.items()}


def run_id(n: int, how: str) -> str:
    return f"orders-n{n}" if n == 1 else f"orders-n{n}-{how}"


def build(rows: list[dict], n: int, how: str) -> list[dict]:
    out = []
    for r in rows:
        if r.get("error") is not None:
            out.append({k: r.get(k) for k in ("case_id", "slice")} | {"run_id": run_id(n, how), "error": r["error"], "pred": None})
            continue
        probs = combine([o["by_key"] for o in r["per_order"][:n]], how)
        case = {k: r[k] for k in ("case_id", "family_id", "slice", "source", "task_type", "perturbation", "n_options", "gold")}
        case["truncated"] = r["truncated_state"]
        out.append(to_record(case, {"by_key": probs, "label_mass": 1.0, "calls": r["calls"], "input_tokens": 0},
                             run_id(n, how), "j-raghavan/s1-gemma4-26b-decision", 0.0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("orders", type=Path, help="session output: one row per case with per_order probabilities")
    ap.add_argument("out", type=Path, help="runs directory to write the variants into")
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.orders.read_text().splitlines() if line.strip()]
    seen = set()
    for n, how in VARIANTS:
        rid = run_id(n, how)
        if rid in seen:
            continue
        seen.add(rid)
        d = args.out / rid
        d.mkdir(parents=True, exist_ok=True)
        with (d / "predictions.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in build(rows, n, how))
        (d / "meta.json").write_text(json.dumps({"run_id": rid, "orders": n, "mean": how,
                                                 "system": "vllm bf16: j-raghavan/s1-gemma4-26b-decision"}))
        print(f"wrote {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
