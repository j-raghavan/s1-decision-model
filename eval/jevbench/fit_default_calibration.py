"""Fit the per-type default calibration the v0 API applies to task families it has no calibrator for.

Pools every slice of the given runs: one temperature for choice, one for score,
and Platt (a, b) for noul. Per-family calibrators fitted on gold labels are
better (see calibrate.py); these defaults only cover unseen families.

    uv run eval/jevbench/fit_default_calibration.py jevbench:gemma4-26b custom:gemma4-26b

Writes api/calibration.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate import TEMPS, as_logits, fit_platt, softmax
from paths import ROOT, SUITES, read_jsonl

OUT = ROOT / "api" / "calibration.json"


def fit_t_value(items: list[tuple[np.ndarray, int]]) -> float:
    def nll(t: float) -> float:
        return -sum(np.log(softmax(z / t)[g] + 1e-300) for z, g in items) / len(items)

    return float(min(TEMPS, key=nll))


def platt_params(items: list[tuple[np.ndarray, int]]) -> tuple[float, float]:
    apply = fit_platt(items)
    # Recover (a, b) from two points of the fitted map: logit(p) = a*x + b.
    def logit_at(x: float) -> float:
        p = apply(np.array([0.0, x]))[1]
        p = min(max(p, 1e-12), 1 - 1e-12)
        return float(np.log(p / (1 - p)))

    b = logit_at(0.0)
    a = logit_at(1.0) - b
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="suite:run_id pairs")
    ap.add_argument("--out", type=Path, default=OUT, help="where to write the calibration (default: the training defaults)")
    args = ap.parse_args()

    by_type: dict[str, list] = {"choice": [], "score": [], "noul": []}
    sources = []
    for spec in args.runs:
        suite, run_id = spec.split(":", 1)
        path = SUITES[suite]["runs"] / run_id / "predictions.jsonl"
        for rec in read_jsonl(path):
            item = as_logits(rec)
            if item is not None:
                by_type[rec["task_type"]].append(item)
        sources.append(spec)

    a, b = platt_params(by_type["noul"])
    cal = {
        "fitted_on": sources,
        "n": {k: len(v) for k, v in by_type.items()},
        "choice": {"temperature": fit_t_value(by_type["choice"])},
        "score": {"temperature": fit_t_value(by_type["score"]) if by_type["score"] else 1.0},
        "noul": {"a": a, "b": b},
    }
    args.out.write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(cal, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
