"""Score an eval suite with a (fine-tuned) Gemma 4 decoder decision model, in the shared prediction format.

Uses the teachers' prompt and reads option-letter logits in one pass (two-digit codes in two passes),
so zero-shot and fine-tuned results compare directly with the teacher runs.

    python eval/jevbench/run_decoder.py --model google/gemma-4-26B-A4B-it --adapter /content/adapter/best \\
        --run-id s1-gemma26b-pilot --suite jevbench --batch 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import SUITES, base_record, read_jsonl  # noqa: E402
from run_ollama import to_record  # noqa: E402
from s1.decoder import Readout, load, option_log_probs  # noqa: E402
from s1.train_decoder import device_and_dtype  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--suite", default="jevbench", choices=sorted(SUITES))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N cases, for smoke tests")
    ap.add_argument("--slices", nargs="+", default=None, help="only these slices")
    ap.add_argument("--case-ids", type=Path, default=None, help="file with one case_id per line: only these cases")
    ap.add_argument("--experts-impl", default=None, help="MoE kernel: eager, grouped_mm or batched_mm (default: library choice)")
    ap.add_argument("--attn-impl", default=None, help="attention kernel: eager or sdpa (default: library choice)")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float32"])
    ap.add_argument("--bos", choices=["auto", "none"], default="auto",
                    help="none reproduces runs made before <bos> was prepended (canaries against saved predictions)")
    args = ap.parse_args()

    device, dtype = device_and_dtype()
    dtype = getattr(torch, args.dtype) if args.dtype else dtype
    tok, model = load(args.model, device, dtype, adapter=args.adapter, experts_impl=args.experts_impl, attn_impl=args.attn_impl)
    model.eval()
    readout = Readout(tok)
    suite = SUITES[args.suite]
    cases = read_jsonl(suite["cases"])
    if args.slices:
        cases = [c for c in cases if c["slice"] in args.slices]
    if args.case_ids:
        wanted = set(args.case_ids.read_text().split())
        cases = [c for c in cases if c["case_id"] in wanted]
    cases = cases[: args.limit] if args.limit else cases
    cases.sort(key=lambda c: len(json.dumps(c["state"])))  # similar lengths per batch, less padding
    run_dir = suite["runs"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps({"run_id": args.run_id, "system": f"hf:{args.model}",
                                                   "adapter": args.adapter, "method": "option-letter logits, one pass",
                                                   "experts_impl": args.experts_impl, "attn_impl": args.attn_impl,
                                                   "dtype": str(dtype), "batch": args.batch}, indent=2))
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as out:
        for i in range(0, len(cases), args.batch):
            chunk = cases[i:i + args.batch]
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    lps = option_log_probs(model, tok, readout, chunk, device, add_bos=args.bos == "auto")
                ms = (time.perf_counter() - t0) * 1000 / len(chunk)
                for c, lp in zip(chunk, lps):
                    probs = {k: math.exp(v) for k, v in lp.items()}
                    scored = {"by_key": probs, "label_mass": 1.0, "calls": 1, "input_tokens": 0}
                    out.write(json.dumps(to_record(c, scored, args.run_id, args.model, ms), ensure_ascii=False) + "\n")
            except Exception as exc:  # recorded; the scorer reports coverage
                for c in chunk:
                    out.write(json.dumps(base_record(c, args.run_id) | {"error": repr(exc)[:300], "pred": None}) + "\n")
            out.flush()
            if (i // args.batch) % 25 == 0:
                print(f"{i + len(chunk)}/{len(cases)}", flush=True)
    print(f"wrote {run_dir / 'predictions.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
