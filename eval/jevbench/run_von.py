"""Score JevBench cases with Von, in-process via von-sdk.

Von consumes the same /v1/systemone question objects the cases carry, so each
case's question is passed through unchanged. For noul, p_true is Von's
calibrated posterior (`noul_raw`), not the banded decision value, so ECE is
measured on the probability Von actually estimates.

    uv run --extra von eval/jevbench/run_von.py
    uv run --extra von eval/jevbench/run_von.py --limit 3   # smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

from tqdm import tqdm

from paths import SUITES, base_record, read_jsonl


def to_record(case: dict, answer, run_id: str, latency_ms: float, model: str) -> dict:
    rec = base_record(case, run_id)
    t = case["task_type"]
    rec.update({"latency_ms": latency_ms, "attempts": 1, "error": None, "answer_type": t})
    if t == "noul":
        p_true = answer.noul_raw if answer.noul_raw is not None else answer.noul
        rec.update({"p_true": p_true, "p_true_banded": answer.noul, "pred": p_true >= 0.5,
                    "confidence": max(p_true, 1 - p_true)})
    elif t == "choice":
        rec.update({"probabilities": answer.probabilities, "pred": answer.choice, "confidence": answer.confidence})
    else:
        # Von's legend keys are level indices as strings, matching the case's 0-based gold.
        probs = {str(int(k)): v for k, v in answer.probabilities.items()}
        expected = sum(int(k) * p for k, p in probs.items())
        pred = max(probs, key=probs.get)
        rec.update({"probabilities": probs, "pred": int(pred), "confidence": answer.confidence,
                    "expected_level": expected, "von_score": answer.score})
    rec.update({"output_tokens": 0, "model_reported": model, "ts": datetime.now(UTC).isoformat()})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="von-1.3")
    ap.add_argument("--model", default="von-latest")
    ap.add_argument("--limit", type=int, default=None, help="cases per slice, for smoke tests")
    ap.add_argument("--suite", default="jevbench", choices=sorted(SUITES))
    args = ap.parse_args()
    suite = SUITES[args.suite]

    from von.engine import VonEngine

    engine = VonEngine.get_instance()
    run_dir = suite["runs"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "predictions.jsonl"
    done = {r["case_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    cases = read_jsonl(suite["cases"])
    if args.limit:
        seen: dict[str, int] = {}
        kept = []
        for c in cases:
            seen[c["slice"]] = seen.get(c["slice"], 0) + 1
            if seen[c["slice"]] <= args.limit:
                kept.append(c)
        cases = kept
    todo = [c for c in cases if c["case_id"] not in done]
    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": args.run_id, "system": "von-sdk (local)", "model_requested": args.model,
        "n_cases_total": len(cases), "one_question_per_request": True,
        "noul_probability": "noul_raw (calibrated posterior, before the band rule)",
    }, indent=2))
    print(f"{args.run_id}: {len(todo)} to score, {len(done)} already done")

    with out_path.open("a", encoding="utf-8") as out:
        for case in tqdm(todo, unit="case"):
            t0 = time.perf_counter()
            try:
                resp = engine.evaluate(state=case["state"], questions={"decision": case["question"]}, model=args.model)
                rec = to_record(case, resp.answers["decision"], args.run_id, (time.perf_counter() - t0) * 1000, args.model)
            except Exception as exc:  # recorded, not fatal: the scorer reports coverage
                rec = base_record(case, args.run_id) | {"error": repr(exc)[:300], "pred": None}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
