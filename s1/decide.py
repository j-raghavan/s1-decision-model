"""Decision rules for the full-run session, as plain functions over prediction files.

Everything the session decides on its own goes through here, so the rules can be tested against runs whose
outcome is already known (tests/test_decide.py) before any GPU time is spent.

Accuracy is computed on yes/no and choice cases only. Score-type cases are excluded: their headline metric
is an error (MAE of the expected level, lower is better), and mixing it with accuracies would invert its
meaning.

A candidate is judged only on full evidence: it must have answers for at least MIN_COVERAGE of the base
model's cases in every slice, so a failed or partial scoring run can never make it look safe.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

MIN_COVERAGE = 0.98
# The custom slices where pilot 2 regressed: policy rules applied to structured state. Watched on their own
# because an overall test on ~480 dev cases catches a pilot-2-sized regression only about half the time.
RULE_SLICES = ("state_api_retry", "state_refund_eligibility", "state_ticket_routing", "ppt_split_slide")
RULE_NET_LOSS_SHARE = 0.02  # more net breaks than 2% of the rule-slice cases counts as a regression


def load_predictions(path: Path) -> dict[str, dict]:
    """case_id -> record, for answered yes/no and choice cases."""
    out = {}
    for line in Path(path).open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("task_type") == "score" or r.get("error") or r.get("pred") is None:
            continue
        out[r["case_id"]] = r
    return out


def is_correct(r: dict) -> bool:
    return str(r["pred"]).lower() == str(r["gold"]).lower()


def slice_mean_accuracy(preds: dict[str, dict]) -> float:
    """Mean over slices of per-slice accuracy (the headline used throughout the project)."""
    by: dict[str, list[bool]] = {}
    for r in preds.values():
        by.setdefault(r["slice"], []).append(is_correct(r))
    return sum(sum(v) / len(v) for v in by.values()) / max(1, len(by))


def coverage(base: dict[str, dict], cand: dict[str, dict]) -> float:
    """Lowest per-slice share of the base model's cases that the candidate answered."""
    by: dict[str, list[bool]] = {}
    for i, r in base.items():
        by.setdefault(r["slice"], []).append(i in cand)
    return min((sum(v) / len(v) for v in by.values()), default=0.0)


def mcnemar_p(newly_wrong: int, newly_right: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs."""
    n = newly_wrong + newly_right
    if n == 0:
        return 1.0
    k = min(newly_wrong, newly_right)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(base: dict[str, dict], cand: dict[str, dict], slices: tuple[str, ...] | None = None) -> dict:
    """Paired comparison of a candidate against the base on the cases both answered (optionally some slices)."""
    ids = [i for i in base if i in cand and (slices is None or base[i]["slice"] in slices)]
    wrong = sum(1 for i in ids if is_correct(base[i]) and not is_correct(cand[i]))
    right = sum(1 for i in ids if not is_correct(base[i]) and is_correct(cand[i]))
    return {"n": len(ids), "newly_wrong": wrong, "newly_right": right, "p": mcnemar_p(wrong, right)}


def regression(base: dict[str, dict], cand: dict[str, dict], alpha: float = 0.05) -> dict:
    """Regression check on the custom set: overall McNemar, or net losses on the rule slices above the cap."""
    overall = paired(base, cand)
    rule = paired(base, cand, RULE_SLICES)
    cap = RULE_NET_LOSS_SHARE * rule["n"]
    net = rule["newly_wrong"] - rule["newly_right"]
    flagged = (overall["newly_wrong"] > overall["newly_right"] and overall["p"] < alpha) or net > cap
    return {"overall": overall, "rule": rule, "rule_net_loss": net, "rule_cap": cap, "regresses": flagged}


def significant_regression(base: dict[str, dict], cand: dict[str, dict]) -> bool:
    return regression(base, cand)["regresses"]


def assess(name: str, base_custom: dict, base_jev: dict, cust: dict, jev: dict) -> dict:
    """One report row: accuracy, coverage and the regression check; eligible only with full coverage."""
    reg = regression(base_custom, cust)
    cov = min(coverage(base_custom, cust), coverage(base_jev, jev))
    return {"name": name, "jev_acc": slice_mean_accuracy(jev), "coverage": cov,
            "custom_newly_wrong": reg["overall"]["newly_wrong"], "custom_newly_right": reg["overall"]["newly_right"],
            "custom_p": reg["overall"]["p"], "rule_net_loss": reg["rule_net_loss"], "rule_cap": reg["rule_cap"],
            "regresses": reg["regresses"], "eligible": cov >= MIN_COVERAGE and not reg["regresses"]}


def choose_snapshot(base_custom: dict, base_jev: dict, candidates: dict[str, tuple[dict, dict]]) -> tuple[str | None, list[dict]]:
    """Best candidate by JevBench-dev accuracy among the eligible ones.

    candidates: name -> (custom_dev predictions, jevbench_dev predictions). Returns (name or None, report rows).
    """
    report = [assess(n, base_custom, base_jev, c, j) for n, (c, j) in candidates.items()]
    eligible = [r for r in report if r["eligible"]]
    best = max(eligible, key=lambda r: r["jev_acc"])["name"] if eligible else None
    return best, report


NON_INFERIORITY_MARGIN = 0.015  # a candidate may lose at most 1.5 points of custom accuracy against the untuned model


def accuracy_change(base: dict[str, dict], cand: dict[str, dict]) -> float:
    """Paired change in accuracy on the cases both answered: (newly right - newly wrong) / n."""
    c = paired(base, cand)
    return (c["newly_right"] - c["newly_wrong"]) / max(1, c["n"])


def non_inferior(base: dict[str, dict], cand: dict[str, dict], margin: float = NON_INFERIORITY_MARGIN) -> bool:
    return accuracy_change(base, cand) >= -margin


def choose_snapshot_ni(base_custom: dict, base_jev: dict, candidates: dict[str, tuple[dict, dict]]) -> tuple[str | None, list[dict]]:
    """Best candidate by JevBench-dev accuracy among those with full coverage and custom-dev accuracy no more than
    NON_INFERIORITY_MARGIN below the untuned model (paired). Report rows also carry the regression check."""
    report = []
    for name, (cust, jev) in candidates.items():
        row = assess(name, base_custom, base_jev, cust, jev)
        row["custom_change"] = accuracy_change(base_custom, cust)
        row["eligible"] = row["coverage"] >= MIN_COVERAGE and non_inferior(base_custom, cust)
        report.append(row)
    eligible = [r for r in report if r["eligible"]]
    best = max(eligible, key=lambda r: r["jev_acc"])["name"] if eligible else None
    return best, report


def base_complete(base_custom: dict, base_jev: dict, n_custom: int, n_jev: int) -> bool:
    """The untuned model's own dev scoring must be essentially complete before anything is compared to it."""
    return len(base_custom) >= MIN_COVERAGE * n_custom and len(base_jev) >= MIN_COVERAGE * n_jev


def gate_informative(base_custom: dict, control_custom: dict) -> bool:
    """Positive control: the regression check must flag pilot 2, whose regression is known. If it does not, the
    dev set cannot see the failure being fixed and selecting on it would be blind."""
    return coverage(base_custom, control_custom) >= MIN_COVERAGE and significant_regression(base_custom, control_custom)


def agreement(a: dict[str, dict], b: dict[str, dict]) -> tuple[float, int]:
    """Share of shared cases with the same prediction (pipeline canary)."""
    ids = [i for i in a if i in b]
    return sum(str(a[i]["pred"]) == str(b[i]["pred"]) for i in ids) / max(1, len(ids)), len(ids)
