"""Average the probabilities of several runs into a new run, case by case.

    uv run eval/jevbench/ensemble.py gemma4-12b gemma4-26b --run-id gemma4-12b+26b

Noul averages p_true, choice and score average the option distributions. The
result is written as an ordinary run, so score.py and calibrate.py pick it up.
"""

from __future__ import annotations

import argparse
import json

from paths import SUITES, read_jsonl, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--suite", default="jevbench", choices=sorted(SUITES))
    args = ap.parse_args()
    RUNS = SUITES[args.suite]["runs"]

    members = [{r["case_id"]: r for r in read_jsonl(RUNS / run / "predictions.jsonl")} for run in args.runs]
    shared = set.intersection(*(set(m) for m in members))
    out = []
    for case_id in sorted(shared):
        recs = [m[case_id] for m in members]
        if any(r.get("error") or r.get("pred") is None for r in recs):
            continue
        rec = {k: recs[0][k] for k in ("case_id", "family_id", "slice", "source", "task_type", "perturbation",
                                         "n_options", "gold", "truncated_state")}
        rec.update({"run_id": args.run_id, "error": None, "answer_type": rec["task_type"],
                    "latency_ms": sum(r["latency_ms"] for r in recs),
                    "input_tokens": recs[0].get("input_tokens"), "model_reported": "+".join(args.runs)})
        if rec["task_type"] == "noul":
            p = sum(float(r["p_true"]) for r in recs) / len(recs)
            rec.update({"p_true": p, "pred": p >= 0.5, "confidence": max(p, 1 - p)})
        else:
            keys = list(recs[0]["probabilities"])
            probs = {k: sum(float(r["probabilities"][k]) for r in recs) / len(recs) for k in keys}
            pred = max(probs, key=probs.get)
            rec.update({"probabilities": probs, "confidence": probs[pred]})
            if rec["task_type"] == "score":
                rec.update({"pred": int(pred), "expected_level": sum(int(k) * p for k, p in probs.items())})
            else:
                rec["pred"] = pred
        out.append(rec)

    write_jsonl(RUNS / args.run_id / "predictions.jsonl", out)
    (RUNS / args.run_id / "meta.json").write_text(json.dumps(
        {"run_id": args.run_id, "system": "probability average of " + ", ".join(args.runs), "n_cases_total": len(out)},
        indent=2))
    print(f"{args.run_id}: {len(out)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
