"""s1 decision model demo: a Gradio app on Hugging Face ZeroGPU.

Same inference code as examples/quickstart.py (copied in as quickstart.py at deploy time), the released bf16
weights, and per-type calibration fitted on the dev splits (calibration.json), returning the same answer shape as
the repository's POST /v1/decisions. The `decide` endpoint is what the project website's playground calls.

Environment: S1_MODEL_PATH (a folder or Hub id; default: the model repo mounted at /models/s1, else the Hub id),
S1_DEVICE (cuda),
S1_GPU_SIZE (xlarge: the bf16 weights need about 52 GB).
"""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path

import gradio as gr
import quickstart as qs
import spaces
import torch
from s1_space import BadRequest, answer, parse
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

HERE = Path(__file__).resolve().parent
MODEL_ID = "j-raghavan/s1-gemma4-26b-decision"
MOUNT = "/models/s1"  # the model repo, mounted read-only as a Space volume
# An explicit S1_MODEL_PATH (a local folder or a Hub id) always wins; otherwise the mount, else the Hub.
MODEL_PATH = os.environ.get("S1_MODEL_PATH") or (MOUNT if (Path(MOUNT) / "config.json").exists() else MODEL_ID)
DEVICE = os.environ.get("S1_DEVICE", "cuda")
GPU_SIZE = os.environ.get("S1_GPU_SIZE", "xlarge")

print(f"s1 demo: loading {MODEL_PATH} on {DEVICE}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
# On ZeroGPU the model is placed on cuda at import time (CUDA emulation outside @spaces.GPU), as the docs require.
model = Gemma4ForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(DEVICE).eval()


@spaces.GPU(size=GPU_SIZE, duration=30)
def score_all(state, questions: dict) -> dict[str, dict[str, float]]:
    return {name: qs.decide(model, tok, state, q) for name, q in questions.items()}


def decide(state_text: str, questions_text: str) -> dict:
    """The API endpoint: state (JSON or text) and questions (JSON) in, typed calibrated answers out."""
    try:
        state, questions = parse(state_text, questions_text)
    except BadRequest as exc:
        return {"error": str(exc)}
    t0 = time.perf_counter()
    raw = score_all(state, questions)
    answers = {name: answer(q["type"], q.get("criteria"), raw[name]) for name, q in questions.items()}
    return {"model": "s1-gemma4-26b-v1", "answers": answers,
            "usage": {"latency_ms": round((time.perf_counter() - t0) * 1000, 1), "questions": len(questions)}}


# ---- UI -----------------------------------------------------------------------------------------------------

EXAMPLES = json.loads((HERE / "examples.json").read_text(encoding="utf-8"))


def bar(label: str, p: float, top: bool) -> str:
    return (f'<div class="s1-row{" s1-top" if top else ""}"><span class="s1-label">{html.escape(label)}</span>'
            f'<span class="s1-track"><span class="s1-fill" style="width:{p * 100:.1f}%"></span></span>'
            f'<span class="s1-p">{p:.2f}</span></div>')


def render(result: dict, questions_text: str) -> str:
    if "error" in result:
        return f'<div class="s1-error">{html.escape(result["error"])}</div>'
    questions = json.loads(questions_text)
    cards = []
    for name, a in result["answers"].items():
        q = questions[name]
        if a["type"] == "noul":
            head = f'{"Yes" if a["noul"] >= 0.5 else "No"} <small>P(yes) = {a["noul"]:.2f}</small>'
            rows = bar("yes", a["noul"], a["noul"] >= 0.5) + bar("no", 1 - a["noul"], a["noul"] < 0.5)
        elif a["type"] == "choice":
            head = f'{html.escape(a["choice"])} <small>confidence {a["confidence"]:.2f}</small>'
            rows = "".join(bar(k, p, k == a["choice"]) for k, p in sorted(a["probabilities"].items(), key=lambda kv: -kv[1]))
        else:
            best = max(a["probabilities"], key=a["probabilities"].get)
            head = f'{html.escape(a["legend"][best])} <small>expected level {a["score"]:.2f}</small>'
            rows = "".join(bar(a["legend"][k], p, k == best) for k, p in a["probabilities"].items())
        cards.append(f'<div class="s1-card"><div class="s1-q"><code>{html.escape(name)}</code> · {q["type"]} · '
                     f'{html.escape(q["instructions"])}</div><div class="s1-a">{head}</div>{rows}</div>')
    cards.append(f'<div class="s1-meta">{result["usage"]["questions"]} question(s) in {result["usage"]["latency_ms"]:.0f} ms '
                 '(includes GPU allocation)</div>')
    return "".join(cards)


def run_ui(state_text: str, questions_text: str):
    result = decide(state_text, questions_text)
    return render(result, questions_text), result


CSS = """
.s1-card{border:1px solid var(--border-color-primary);border-radius:10px;padding:12px 14px;margin:0 0 10px}
.s1-q{font-size:.85em;opacity:.8;margin-bottom:4px}.s1-a{font-size:1.15em;font-weight:600;margin-bottom:8px}
.s1-a small{font-weight:400;opacity:.7;margin-left:6px}
.s1-row{display:grid;grid-template-columns:minmax(80px,30%) 1fr 44px;gap:8px;align-items:center;font-size:.9em;margin:3px 0}
.s1-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.s1-p{text-align:right;font-variant-numeric:tabular-nums}
.s1-track{height:8px;border-radius:4px;background:var(--neutral-200);overflow:hidden}
.s1-fill{display:block;height:100%;background:var(--neutral-500)}.s1-top .s1-fill{background:var(--color-accent)}
.s1-top .s1-label{font-weight:600}.s1-meta{font-size:.8em;opacity:.7}
.s1-error{border:1px solid #d33;border-radius:8px;padding:10px;color:#d33}
"""

with gr.Blocks(title="s1 decision model") as demo:
    gr.Markdown(
        "# s1 decision model\n"
        "Typed decisions with calibrated probabilities, in one forward pass. Give it a **state** (JSON or text) and "
        "**questions** (`choice`, `noul` for yes/no, `score` for an ordered scale). "
        "[Model](https://huggingface.co/j-raghavan/s1-gemma4-26b-decision) · "
        "[GitHub](https://github.com/j-raghavan/s1-decision-model) · "
        "[Leaderboard](https://j-raghavan.github.io/s1-decision-model/leaderboard.html)")
    with gr.Row():
        with gr.Column(scale=1):
            example = gr.Dropdown(choices=[e["name"] for e in EXAMPLES], value=EXAMPLES[0]["name"], label="Example")
            state = gr.Code(value=EXAMPLES[0]["state"], language="json", label="State", lines=10)
            questions = gr.Code(value=EXAMPLES[0]["questions"], language="json", label="Questions", lines=14)
            go = gr.Button("Decide", variant="primary")
        with gr.Column(scale=1):
            cards = gr.HTML(label="Answers")
            raw = gr.JSON(label="Response")
    example.change(lambda n: next((e["state"], e["questions"]) for e in EXAMPLES if e["name"] == n),
                   example, [state, questions], api_name=False)
    go.click(run_ui, [state, questions], [cards, raw], api_name=False)
    gr.api(decide, api_name="decide")
    gr.Markdown("Runs on Hugging Face ZeroGPU: the first request after a pause waits for a GPU. "
                "Not affiliated with TypeSafe AI (Jev). Do not use as the only control for high-stakes decisions.")

if __name__ == "__main__":
    demo.queue(max_size=32).launch(css=CSS)
