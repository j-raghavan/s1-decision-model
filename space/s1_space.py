"""Request parsing and calibrated answers for the s1 demo Space, free of any model, so they can be tested in CI.

answer() mirrors the repository API (api/server.py); tests/test_space.py checks they agree.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import quickstart as qs

HERE = Path(__file__).resolve().parent
MAX_STATE_CHARS = 12_000
MAX_QUESTIONS = 6
CAL = json.loads((HERE / "calibration.json").read_text(encoding="utf-8"))


class BadRequest(ValueError):
    pass


def softmax_t(probs: dict[str, float], t: float) -> dict[str, float]:
    logits = {k: math.log(max(p, 1e-15)) / t for k, p in probs.items()}
    m = max(logits.values())
    exp = {k: math.exp(v - m) for k, v in logits.items()}
    z = sum(exp.values())
    return {k: v / z for k, v in exp.items()}


def answer(qtype: str, criteria, by_key: dict[str, float]) -> dict:
    """Calibrated answer, in the shape of the repository API (api/server.py answer())."""
    if qtype == "noul":
        p = by_key["true"] / (by_key["true"] + by_key["false"])
        logit = math.log(max(p, 1e-15)) - math.log(max(1 - p, 1e-15))
        z = CAL["noul"]["a"] * logit + CAL["noul"]["b"]
        p_cal = 1 / (1 + math.exp(-max(min(z, 50), -50)))
        return {"type": "noul", "noul": p_cal, "noul_raw": p}
    probs = softmax_t(by_key, CAL[qtype]["temperature"])
    best = max(probs, key=probs.get)
    if qtype == "choice":
        return {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}
    return {"type": "score", "score": sum(int(k) * p for k, p in probs.items()), "confidence": probs[best],
            "legend": {str(i): text for i, text in enumerate(criteria)}, "probabilities": probs}


def parse(state_text: str, questions_text: str) -> tuple[object, dict]:
    if len(state_text) > MAX_STATE_CHARS:
        raise BadRequest(f"state is longer than {MAX_STATE_CHARS:,} characters")
    try:
        state = json.loads(state_text)
    except json.JSONDecodeError:
        state = state_text  # plain text is a valid state
    try:
        questions = json.loads(questions_text)
    except json.JSONDecodeError as exc:
        raise BadRequest(f"questions must be a JSON object: {exc}") from exc
    if not isinstance(questions, dict) or not questions:
        raise BadRequest("questions must be a JSON object of {name: question}")
    if len(questions) > MAX_QUESTIONS:
        raise BadRequest(f"at most {MAX_QUESTIONS} questions per request")
    for name, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "noul", "score") or not q.get("instructions"):
            raise BadRequest(f"question {name!r} needs a type (choice, noul or score) and instructions")
        n = len(qs.options_for(q))
        if q["type"] == "choice" and not (isinstance(q.get("criteria"), dict) and n >= 2):
            raise BadRequest(f"choice question {name!r} needs criteria: at least two option keys with descriptions")
        if q["type"] == "score" and not (isinstance(q.get("criteria"), list) and n >= 2):
            raise BadRequest(f"score question {name!r} needs criteria: an ordered list of at least two levels")
        if n > 26:
            raise BadRequest(f"question {name!r} has {n} options; this demo handles up to 26")
    return state, questions
