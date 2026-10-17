"""eval/jevbench/order_average.py: averaging over option orders keeps a distribution over the same keys."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "jevbench"))
import order_average as oa  # noqa: E402

DISTS = [{"a": 0.6, "b": 0.3, "c": 0.1}, {"a": 0.2, "b": 0.7, "c": 0.1}]


@pytest.mark.parametrize("how", ["arith", "geo"])
def test_combine_is_a_distribution_over_the_same_keys(how):
    out = oa.combine(DISTS, how)
    assert set(out) == {"a", "b", "c"} and sum(out.values()) == pytest.approx(1.0)


def test_arithmetic_and_geometric_means():
    assert oa.combine(DISTS, "arith") == pytest.approx({"a": 0.4, "b": 0.5, "c": 0.1})
    geo = oa.combine(DISTS, "geo")  # sqrt(.12), sqrt(.21), .1, renormalised
    assert geo["b"] > geo["a"] > geo["c"]


def test_one_order_is_unchanged():
    assert oa.combine(DISTS[:1], "geo") == pytest.approx(DISTS[0])
