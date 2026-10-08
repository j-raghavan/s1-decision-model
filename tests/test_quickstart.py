"""examples/quickstart.py is self-contained on purpose, so it copies the prompt and option layout. These tests pin the
copy to the repository's renderer (eval/jevbench/run_ollama.py), which produced the training prompts."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "jevbench"))
import run_ollama  # noqa: E402

spec = importlib.util.spec_from_file_location("quickstart", ROOT / "examples" / "quickstart.py")
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)

STATE = {"ticket": "I was charged twice for order 4471.", "nested": {"amount": 42.5, "tags": ["refund", "ünïcode"]}}
QUESTIONS = [
    {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "shipping": "Deliveries"}},
    {"type": "choice", "instructions": "Pick one", "criteria": {"a": "first option", "b": "second option", "c": None}},
    {"type": "choice", "instructions": "Same key and text", "criteria": {"yes": "yes", "no": ""}},
    {"type": "noul", "instructions": "Is it urgent?",
     "criteria": {"true": "Yes, the condition holds.", "false": "No, the condition does not hold."}},
    {"type": "score", "instructions": "How severe?", "criteria": ["minor", "moderate", "serious", "critical"]},
    {"type": "choice", "instructions": "Many options", "criteria": {f"opt{i}": f"option {i}" for i in range(26)}},
]


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q["instructions"])
def test_prompt_matches_repository_renderer(question):
    options = run_ollama.options_for(question)
    assert qs.options_for(question) == options
    expected = "<bos>" + run_ollama.render({"state": STATE, "question": question}, run_ollama.labels_for(len(options)),
                                           options, "gemma4")
    assert qs.prompt_for(STATE, question, options) == expected


def test_noul_default_criteria_match_the_api():
    import api.server as server
    q = {"type": "noul", "instructions": "Is it urgent?"}
    api_q = server.normalise(server.Question(**q))
    assert qs.options_for(q) == run_ollama.options_for(api_q)


def test_more_than_26_options_is_refused_not_misread():
    q = {"type": "choice", "instructions": "?", "criteria": {f"k{i}": str(i) for i in range(27)}}
    with pytest.raises(ValueError):
        qs.decide(None, None, {}, q)
