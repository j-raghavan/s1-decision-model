"""v1 /v1/decisions API: the s1 decision model behind a typed-decision endpoint.

Serves the fine-tuned Gemma 4 26B-A4B decision model (LoRA trained with <bos>, merged; Hugging Face
j-raghavan/s1-gemma4-26b-decision) through a local Ollama, int4, imported as `s1-gemma4-26b`. One forward pass per
question reads the option-letter probabilities (two passes for more than 26 options). Ollama does not add <bos> to
an imported model, so the API prepends it (S1_ADD_BOS).
Probabilities are calibrated per answer type with api/calibration_s1.json, fitted on the dev splits scored through
this same Ollama model (the test sets were not used).

    uv run --extra api uvicorn api.server:app --port 8000
    curl -s localhost:8000/v1/decisions -H 'content-type: application/json' -d '{
      "state": {"ticket": "I was charged twice for order 4471."},
      "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
                             "criteria": {"billing": "Payments and refunds", "shipping": "Deliveries", "tech": "Bugs"}}}}'

Request and answer shapes follow the /v1/systemone convention used by Jev, Von
and Laya: questions are `choice` (criteria: key -> description), `noul`
(yes/no; criteria optional) or `score` (criteria: ordered list of levels).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "jevbench"))
from run_ollama import ollama_fetcher, score_case  # noqa: E402

MODEL = os.environ.get("S1_MODEL", os.environ.get("S1_TEACHER_MODEL", "s1-gemma4-26b"))
CALIBRATION = Path(os.environ.get("S1_CALIBRATION", ROOT / "api" / "calibration_s1.json"))
VERSION = "s1-gemma4-26b-v1"
# Ollama does not add <bos> to a model imported from safetensors, and Gemma 4 degrades without it, so the API
# prepends it; set S1_ADD_BOS=0 only for a model that adds it itself (Ollama's library gemma4).
FETCH = ollama_fetcher("<bos>" if os.environ.get("S1_ADD_BOS", "1") == "1" else "")


class Question(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str
    criteria: dict[str, str | None] | list[str] | None = None


class DecisionRequest(BaseModel):
    state: Any
    questions: dict[str, Question] = Field(min_length=1)


def load_calibration() -> dict:
    """Per-type defaults: a temperature for choice and score, Platt (a, b) for noul. Identity if absent."""
    if CALIBRATION.exists():
        return json.loads(CALIBRATION.read_text(encoding="utf-8"))
    return {"choice": {"temperature": 1.0}, "score": {"temperature": 1.0}, "noul": {"a": 1.0, "b": 0.0}}


def softmax_t(probs: dict[str, float], t: float) -> dict[str, float]:
    logits = {k: math.log(max(p, 1e-15)) / t for k, p in probs.items()}
    m = max(logits.values())
    exp = {k: math.exp(v - m) for k, v in logits.items()}
    z = sum(exp.values())
    return {k: v / z for k, v in exp.items()}


def answer(question: Question, by_key: dict[str, float], cal: dict) -> dict:
    if question.type == "noul":
        p = by_key["true"] / (by_key["true"] + by_key["false"])
        logit = math.log(max(p, 1e-15)) - math.log(max(1 - p, 1e-15))
        z = cal["noul"]["a"] * logit + cal["noul"]["b"]
        p_cal = 1 / (1 + math.exp(-max(min(z, 50), -50)))
        return {"type": "noul", "noul": p_cal, "noul_raw": p}
    probs = softmax_t(by_key, cal[question.type]["temperature"])
    best = max(probs, key=probs.get)
    if question.type == "choice":
        return {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}
    levels = question.criteria
    return {"type": "score", "score": sum(int(k) * p for k, p in probs.items()), "confidence": probs[best],
            "legend": {str(i): text for i, text in enumerate(levels)}, "probabilities": probs}


def normalise(q: Question) -> dict:
    """The case-shaped question the scorer expects; noul gets default criteria when none are given."""
    if q.type == "noul":
        criteria = q.criteria if isinstance(q.criteria, dict) and q.criteria else {}
        return {"type": "noul", "instructions": q.instructions,
                "criteria": {"true": criteria.get("true") or "Yes, the condition holds.",
                             "false": criteria.get("false") or "No, the condition does not hold."}}
    if q.type == "choice" and not (isinstance(q.criteria, dict) and len(q.criteria) >= 2):
        raise HTTPException(422, "choice questions need criteria: a mapping of at least two option keys to descriptions")
    if q.type == "score" and not (isinstance(q.criteria, list) and len(q.criteria) >= 2):
        raise HTTPException(422, "score questions need criteria: an ordered list of at least two levels")
    return {"type": q.type, "instructions": q.instructions, "criteria": q.criteria}


app = FastAPI(title="s1-model", version="1.0.0")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "model": MODEL, "version": VERSION,
            "calibration": str(CALIBRATION) if CALIBRATION.exists() else "identity"}


@app.post("/v1/decisions")
def decisions(req: DecisionRequest) -> dict:
    cal = load_calibration()
    t0 = time.perf_counter()
    answers, tokens = {}, 0
    with httpx.Client(timeout=600) as client:
        for key, q in req.questions.items():
            question = normalise(q)
            case = {"state": req.state, "question": question, "task_type": question["type"]}
            try:
                scored = score_case(client, MODEL, case, fetch=FETCH)
            except httpx.HTTPError as exc:
                raise HTTPException(502, f"model call failed: {exc}") from exc
            answers[key] = answer(q, scored["by_key"], cal)
            tokens += scored["input_tokens"]
    return {"model": VERSION, "answers": answers,
            "usage": {"latency_ms": round((time.perf_counter() - t0) * 1000, 1), "input_tokens": tokens}}
