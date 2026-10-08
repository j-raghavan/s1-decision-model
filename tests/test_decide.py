"""The full-run decision rules must reproduce verdicts already known from saved runs.

Uses local prediction files (results/raw is not in git); skipped when they are absent.
"""

from pathlib import Path

import pytest

from s1.decide import (
    agreement,
    assess,
    base_complete,
    choose_snapshot,
    coverage,
    gate_informative,
    load_predictions,
    mcnemar_p,
    paired,
    regression,
    slice_mean_accuracy,
)

RAW = Path(__file__).resolve().parents[1] / "results" / "raw"
BASE, P1, P2 = "gemma4-26b-bf16-vllm", "g26-pilot-vllm", "g26-pilot2-vllm"
need = pytest.mark.skipif(not (RAW / "custom" / P2 / "predictions.jsonl").exists(), reason="local predictions absent")


def preds(suite, run):
    return load_predictions(RAW / suite / run / "predictions.jsonl")


def test_mcnemar_exact_values():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) == 1.0
    assert abs(mcnemar_p(22, 7) - 0.0081) < 0.0005  # hand-checked binomial tail, doubled


@need
def test_known_custom_comparisons():
    base = preds("custom", BASE)
    assert paired(base, preds("custom", P2)) == {"n": 960, "newly_wrong": 22, "newly_right": 7, "p": pytest.approx(0.0081, abs=5e-4)}
    p1 = paired(base, preds("custom", P1))
    assert (p1["newly_wrong"], p1["newly_right"]) == (8, 14)


@need
def test_regression_rule_on_known_runs():
    base = preds("custom", BASE)
    r2 = regression(base, preds("custom", P2))
    assert r2["rule_net_loss"] == 16 and r2["regresses"] is True             # the regression we found
    assert regression(base, preds("custom", P1))["regresses"] is False        # pilot 1 had none
    assert regression(base, preds("custom", "g26-hf-zeroshot"))["regresses"] is False  # same model, other engine
    assert regression(base, preds("custom", "g26-pilot"))["regresses"] is False        # pilot 1 through HF


@need
def test_accuracy_matches_scorer_and_excludes_score_slices():
    jev = preds("jevbench", P2)
    assert all(r["task_type"] != "score" for r in jev.values())
    assert slice_mean_accuracy(jev) == pytest.approx(0.795, abs=0.0006)       # score.py: MEAN ACC 0.795
    assert slice_mean_accuracy(preds("jevbench", BASE)) == pytest.approx(0.742, abs=0.0006)
    assert slice_mean_accuracy(preds("custom", BASE)) == pytest.approx(0.940, abs=0.0006)


@need
def test_missing_or_partial_scores_are_never_eligible():
    base_c, base_j = preds("custom", BASE), preds("jevbench", BASE)
    # the reviewer's reproduction: a failed custom run used to make a candidate eligible
    row = assess("failed", base_c, base_j, {}, base_j)
    assert row["coverage"] == 0.0 and row["eligible"] is False
    # a partial run (only the first 90% of one slice) is also ineligible
    first = next(iter(base_c.values()))["slice"]
    ids = [i for i, r in base_c.items() if r["slice"] == first]
    partial = {i: r for i, r in base_c.items() if r["slice"] != first or i in ids[: int(0.9 * len(ids))]}
    assert coverage(base_c, partial) < 0.98 and assess("partial", base_c, base_j, partial, base_j)["eligible"] is False
    # everything failed: nothing is chosen
    best, _ = choose_snapshot(base_c, base_j, {"a": ({}, {}), "b": ({}, {})})
    assert best is None
    assert base_complete(base_c, base_j, len(base_c), len(base_j)) is True
    assert base_complete({}, base_j, len(base_c), len(base_j)) is False


@need
def test_selection_and_positive_control_on_known_runs():
    base_c, base_j = preds("custom", BASE), preds("jevbench", BASE)
    cands = {"pilot1": (preds("custom", P1), preds("jevbench", P1)), "pilot2": (preds("custom", P2), preds("jevbench", P2))}
    best, report = choose_snapshot(base_c, base_j, cands)
    rows = {r["name"]: r for r in report}
    assert rows["pilot2"]["eligible"] is False and rows["pilot1"]["eligible"] is True
    assert best == "pilot1"  # pilot 2 is better on JevBench but is excluded by its custom regression
    assert gate_informative(base_c, preds("custom", P2)) is True    # the control is flagged: the gate can see it
    assert gate_informative(base_c, preds("custom", P1)) is False   # a clean model would not serve as a control
    assert gate_informative(base_c, {}) is False                    # a failed control run never counts


@need
def test_canary_hf_vs_vllm_agreement_on_pilot1():
    a, n = agreement(preds("jevbench", "g26-pilot"), preds("jevbench", P1))  # same adapter, HF vs vLLM
    assert n == 1600 and a >= 0.97


@need
def test_non_inferiority_rule_on_known_runs():
    from s1.decide import accuracy_change, choose_snapshot_ni, non_inferior
    base_c, base_j = preds("custom", BASE), preds("jevbench", BASE)
    assert accuracy_change(base_c, preds("custom", P2)) == pytest.approx(-15 / 960)
    assert non_inferior(base_c, preds("custom", P2)) is False      # -1.56 points: just outside the 1.5 margin
    assert non_inferior(base_c, preds("custom", P1)) is True       # +0.63 points
    assert non_inferior(base_c, preds("custom", "g26-hf-zeroshot")) is True  # engine noise only
    best, report = choose_snapshot_ni(base_c, base_j, {"pilot1": (preds("custom", P1), preds("jevbench", P1)),
                                                       "pilot2": (preds("custom", P2), preds("jevbench", P2)),
                                                       "failed": ({}, preds("jevbench", P2))})
    rows = {r["name"]: r for r in report}
    assert best == "pilot1" and rows["failed"]["eligible"] is False and rows["pilot2"]["eligible"] is False
