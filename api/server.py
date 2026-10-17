"""v1 /v1/decisions API: the s1 decision model behind a typed-decision endpoint.

Serves the fine-tuned Gemma 4 26B-A4B decision model (LoRA trained with <bos>, merged; Hugging Face
j-raghavan/s1-gemma4-26b-decision) through one of two backends, chosen with S1_BACKEND:

  ollama (default)  a local Ollama, int4, published as `jrlabs01/s1`; calibration api/calibration_s1.json
  vllm              a vLLM OpenAI-compatible server at VLLM_URL serving the Hugging Face weights;
                    calibration api/calibration_s1_vllm.json

One forward pass per question reads the option-letter probabilities (two passes for more than 26 options). Neither
Ollama (for an imported model) nor Gemma 4's tokenizer under vLLM adds <bos>, so the API prepends it (S1_ADD_BOS).
Each calibration is fitted per answer type on the dev splits scored through that backend (the test sets were not
used). S1_CALIBRATION overrides the file.

    uv run --extra api uvicorn api.server:app --port 8000
    S1_BACKEND=vllm VLLM_URL=http://localhost:8001 uv run --extra api uvicorn api.server:app --port 8000
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
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "jevbench"))
from run_ollama import OLLAMA, ollama_fetcher, score_case, vllm_fetcher  # noqa: E402

BACKEND = os.environ.get("S1_BACKEND", "ollama")
if BACKEND not in ("ollama", "vllm"):
    raise ValueError(f"S1_BACKEND must be ollama or vllm, not {BACKEND!r}")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8001")
# The model name the backend serves: the Ollama tag, or vLLM's --served-model-name (by default the path or Hub id
# given to `vllm serve`).
DEFAULT_MODEL = {"ollama": "jrlabs01/s1", "vllm": "j-raghavan/s1-gemma4-26b-decision"}[BACKEND]
MODEL = os.environ.get("S1_MODEL", os.environ.get("S1_TEACHER_MODEL", DEFAULT_MODEL))
DEFAULT_CALIBRATION = {"ollama": "calibration_s1.json", "vllm": "calibration_s1_vllm.json"}[BACKEND]
CALIBRATION = Path(os.environ.get("S1_CALIBRATION", ROOT / "api" / DEFAULT_CALIBRATION))
VERSION = "s1-gemma4-26b-v1"
# Gemma 4 degrades without <bos>, and neither backend adds it here, so the API prepends it; set S1_ADD_BOS=0 only
# for a model that adds it itself (Ollama's library gemma4).
BOS = "<bos>" if os.environ.get("S1_ADD_BOS", "1") == "1" else ""
FETCH = vllm_fetcher(VLLM_URL, BOS) if BACKEND == "vllm" else ollama_fetcher(BOS)
# S1_CHOICE_ORDERS=2 also scores each choice question with its options reversed and averages the two: about a third
# fewer answers that change with option order on JevBench, at twice the model calls for choice questions.
CHOICE_ORDERS = int(os.environ.get("S1_CHOICE_ORDERS", "1"))
if CHOICE_ORDERS not in (1, 2):
    raise ValueError(f"S1_CHOICE_ORDERS must be 1 or 2, not {CHOICE_ORDERS}")


class Question(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str
    criteria: dict[str, str | None] | list[str] | None = None


class DecisionRequest(BaseModel):
    state: Any
    questions: dict[str, Question] = Field(min_length=1)


class CalibrationError(ValueError):
    pass


def load_calibration() -> dict:
    """Per-type defaults: a temperature for choice and score, Platt (a, b) for noul. Identity if absent.
    Raises CalibrationError for a file that would make answers meaningless (missing keys, non-positive or
    non-finite temperature, non-finite Platt parameters)."""
    if not CALIBRATION.exists():
        return {"choice": {"temperature": 1.0}, "score": {"temperature": 1.0}, "noul": {"a": 1.0, "b": 0.0}}
    try:
        cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        temps = [float(cal[t]["temperature"]) for t in ("choice", "score")]
        platt = [float(cal["noul"]["a"]), float(cal["noul"]["b"])]
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"{CALIBRATION}: missing or malformed field ({exc!r})") from exc
    if not all(math.isfinite(t) and t > 0 for t in temps) or not all(math.isfinite(x) for x in platt):
        raise CalibrationError(f"{CALIBRATION}: temperatures must be finite and positive, Platt parameters finite")
    return cal


def ollama_status(timeout: float = 5.0) -> tuple[bool, str]:
    """Whether the Ollama server answers and has MODEL loaded or available."""
    base = OLLAMA.removesuffix("/api/generate")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        return False, f"Ollama unreachable at {base}: {exc}"
    names = {m.get("name", "") for m in r.json().get("models", [])}
    if MODEL not in names and f"{MODEL}:latest" not in names:
        return False, f"model {MODEL} not found in Ollama"
    return True, "ok"


def vllm_status(timeout: float = 5.0) -> tuple[bool, str]:
    """Whether the vLLM server answers and serves MODEL."""
    base = VLLM_URL.rstrip("/")
    try:
        r = httpx.get(f"{base}/v1/models", timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        return False, f"vLLM unreachable at {base}: {exc}"
    names = {m.get("id", "") for m in r.json().get("data", [])}
    if MODEL not in names:
        return False, f"model {MODEL} not served by vLLM (serving: {', '.join(sorted(names)) or 'nothing'})"
    return True, "ok"


def backend_status() -> tuple[bool, str]:
    return vllm_status() if BACKEND == "vllm" else ollama_status()


def softmax_t(probs: dict[str, float], t: float) -> dict[str, float]:
    logits = {k: math.log(max(p, 1e-15)) / t for k, p in probs.items()}
    m = max(logits.values())
    exp = {k: math.exp(v - m) for k, v in logits.items()}
    z = sum(exp.values())
    return {k: v / z for k, v in exp.items()}


def answer(question: Question, by_key: dict[str, float], cal: dict) -> dict:
    if not by_key or sum(by_key.values()) <= 0:
        raise HTTPException(502, "the model returned no probability for any option")
    if question.type == "noul":
        yes, no = by_key.get("true", 0.0), by_key.get("false", 0.0)
        if yes + no <= 0:
            raise HTTPException(502, "the model returned no probability for yes or no")
        p = yes / (yes + no)
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
def healthz(response: Response) -> dict:
    """Ready only if the calibration is valid and the backend serves the model; 503 otherwise."""
    model_ok, model_msg = backend_status()
    try:
        load_calibration()
        cal_ok, cal_msg = True, str(CALIBRATION) if CALIBRATION.exists() else "identity"
    except CalibrationError as exc:
        cal_ok, cal_msg = False, str(exc)
    ok = model_ok and cal_ok
    if not ok:
        response.status_code = 503
    return {"ok": ok, "model": MODEL, "version": VERSION, "backend": BACKEND, BACKEND: model_msg, "calibration": cal_msg}


@app.post("/v1/decisions")
def decisions(req: DecisionRequest) -> dict:
    try:
        cal = load_calibration()
    except CalibrationError as exc:
        raise HTTPException(500, f"server calibration is invalid: {exc}") from exc
    t0 = time.perf_counter()
    answers, tokens = {}, 0
    with httpx.Client(timeout=600) as client:
        for key, q in req.questions.items():
            question = normalise(q)
            case = {"state": req.state, "question": question, "task_type": question["type"]}
            try:
                scored = score_case(client, MODEL, case, fetch=FETCH)
                by_key, tokens = scored["by_key"], tokens + scored["input_tokens"]
                if question["type"] == "choice" and CHOICE_ORDERS == 2:
                    flipped = {**question, "criteria": dict(reversed(list(question["criteria"].items())))}
                    second = score_case(client, MODEL, {**case, "question": flipped}, fetch=FETCH)
                    by_key = {k: (p + second["by_key"][k]) / 2 for k, p in by_key.items()}
                    tokens += second["input_tokens"]
            except httpx.HTTPError as exc:
                raise HTTPException(502, f"model call failed: {exc}") from exc
            answers[key] = answer(q, by_key, cal)
    return {"model": VERSION, "answers": answers,
            "usage": {"latency_ms": round((time.perf_counter() - t0) * 1000, 1), "input_tokens": tokens}}
