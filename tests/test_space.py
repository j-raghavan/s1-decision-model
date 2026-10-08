"""The demo Space (space/) must answer exactly like the repository API, validate its input, and ship valid examples.
Tests the model-free part (space/s1_space.py); the model call is the tested examples/quickstart.py."""

import filecmp
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "space"), str(ROOT / "examples")]
import s1_space  # noqa: E402

fastapi = pytest.importorskip("fastapi")
import api.server as server  # noqa: E402

EXAMPLES = json.loads((ROOT / "space" / "examples.json").read_text(encoding="utf-8"))
CASES = [("choice", {"a": "A", "b": "B", "c": "C"}, {"a": 0.7, "b": 0.2, "c": 0.1}),
         ("noul", None, {"true": 0.83, "false": 0.17}),
         ("score", ["low", "mid", "high", "max"], {"0": 0.1, "1": 0.5, "2": 0.3, "3": 0.1})]


@pytest.mark.parametrize("qtype,criteria,by_key", CASES, ids=[c[0] for c in CASES])
def test_answers_match_the_api(qtype, criteria, by_key):
    q = server.Question(type=qtype, instructions="?", criteria=criteria)
    expected = server.answer(q, dict(by_key), s1_space.CAL)
    got = s1_space.answer(qtype, criteria, dict(by_key))
    assert got.keys() == expected.keys()
    for k, v in expected.items():
        if isinstance(v, float):
            assert math.isclose(got[k], v, rel_tol=1e-12), k
        else:
            assert got[k] == v, k


def test_calibration_is_valid():
    cal = s1_space.CAL
    assert all(math.isfinite(cal[t]["temperature"]) and cal[t]["temperature"] > 0 for t in ("choice", "score"))
    assert all(math.isfinite(cal["noul"][x]) for x in ("a", "b"))


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["name"] for e in EXAMPLES])
def test_examples_parse(example):
    state, questions = s1_space.parse(example["state"], example["questions"])
    assert isinstance(state, dict) and 1 <= len(questions) <= s1_space.MAX_QUESTIONS


@pytest.mark.parametrize("state,questions,message", [
    ("x" * (s1_space.MAX_STATE_CHARS + 1), '{"q": {"type": "noul", "instructions": "?"}}', "longer than"),
    ("{}", "not json", "JSON object"),
    ("{}", "[]", "JSON object"),
    ("{}", json.dumps({f"q{i}": {"type": "noul", "instructions": "?"} for i in range(7)}), "at most"),
    ("{}", '{"q": {"type": "maybe", "instructions": "?"}}', "needs a type"),
    ("{}", '{"q": {"type": "choice", "instructions": "?", "criteria": {"only": "one"}}}', "at least two"),
    ("{}", '{"q": {"type": "score", "instructions": "?", "criteria": ["one"]}}', "at least two"),
    ("{}", json.dumps({"q": {"type": "choice", "instructions": "?", "criteria": {f"k{i}": "x" for i in range(27)}}}), "up to 26"),
])
def test_bad_requests_are_rejected(state, questions, message):
    with pytest.raises(s1_space.BadRequest, match=message):
        s1_space.parse(state, questions)


def test_plain_text_state_is_accepted():
    state, _ = s1_space.parse("just some text", '{"q": {"type": "noul", "instructions": "?"}}')
    assert state == "just some text"


def test_space_copy_of_quickstart_is_current():
    copy = ROOT / "space" / "quickstart.py"
    if copy.exists():  # written by scripts/deploy_space.py; must never drift from the tested original
        assert filecmp.cmp(copy, ROOT / "examples" / "quickstart.py", shallow=False)
