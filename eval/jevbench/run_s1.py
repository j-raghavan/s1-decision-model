"""Score an eval suite with a trained Tier S checkpoint, in the same prediction format as every other system.

    uv run --extra train python eval/jevbench/run_s1.py --checkpoint checkpoints/trial20k/best.pt --run-id s1-trial20k
    uv run --extra train python eval/jevbench/run_s1.py --checkpoint ... --run-id ... --suite custom

Then score as usual: uv run eval/jevbench/score.py [--suite custom], and calibrate.py for post-hoc calibration.
Latency is measured per batch of one question, so it reflects single-request serving on this machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import SUITES, base_record, read_jsonl  # noqa: E402
from s1.model import S1Model  # noqa: E402
from s1.train import device_and_dtype  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--suite", default="jevbench", choices=sorted(SUITES))
    ap.add_argument("--max-len", type=int, default=None, help="defaults to the checkpoint's training max_len")
    args = ap.parse_args()

    suite = SUITES[args.suite]
    # weights_only=False: these checkpoints come from our own training runs, and older ones store run settings
    # (including pathlib paths) that PyTorch's restricted unpickler rejects. Never point this at untrusted files.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    max_len = args.max_len or state.get("args", {}).get("max_len", 2048)
    device, dtype = device_and_dtype()
    model = S1Model(max_len=max_len)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    run_dir = suite["runs"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": args.run_id, "system": "s1 Tier S (ModernBERT-large, order-invariant option markers)",
        "checkpoint": str(args.checkpoint), "step": state.get("step"), "max_len": max_len, "device": device,
    }, indent=2))
    out = (run_dir / "predictions.jsonl").open("w", encoding="utf-8")
    for case in tqdm(read_jsonl(suite["cases"]), unit="case", mininterval=10):
        t0 = time.perf_counter()
        try:
            with torch.autocast(device, dtype=dtype, enabled=dtype is not None):
                probs = model.predict([case], device)[0]
            ms = (time.perf_counter() - t0) * 1000
            rec = base_record(case, args.run_id) | {"latency_ms": ms, "attempts": 1, "error": None,
                                                    "answer_type": case["task_type"], "output_tokens": 0,
                                                    "model_reported": args.run_id,
                                                    "ts": datetime.now(UTC).isoformat()}
            if case["task_type"] == "noul":
                p = probs["true"]
                rec |= {"p_true": p, "pred": p >= 0.5, "confidence": max(p, 1 - p)}
            elif case["task_type"] == "choice":
                pred = max(probs, key=probs.get)
                rec |= {"probabilities": probs, "pred": pred, "confidence": probs[pred]}
            else:
                pred = max(probs, key=probs.get)
                rec |= {"probabilities": probs, "pred": int(pred), "confidence": probs[pred],
                        "expected_level": sum(int(k) * v for k, v in probs.items())}
        except Exception as exc:  # recorded; the scorer reports coverage
            rec = base_record(case, args.run_id) | {"error": repr(exc)[:300], "pred": None}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()
    print(f"wrote {run_dir / 'predictions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
