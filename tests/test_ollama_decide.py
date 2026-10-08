"""examples/ollama_decide.py: the prompt must equal the training renderer, and reading the probabilities must keep the
most likely spelling of each option letter (Ollama lists both " A" and "A"; keeping the wrong one inverted answers)."""

import importlib.util
import json
import math
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "jevbench"))
import run_ollama  # noqa: E402

spec = importlib.util.spec_from_file_location("ollama_decide", ROOT / "examples" / "ollama_decide.py")
od = importlib.util.module_from_spec(spec)
spec.loader.exec_module(od)

QUESTIONS = [
    {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": "Bugs", "c": "third"}},
    {"type": "noul", "instructions": "Is it urgent?", "criteria": od.NOUL},
    {"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high"]},
]
STATE = {"ticket": "charged twice", "n": 2}


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q["type"])
def test_prompt_matches_training_renderer(q):
    options = run_ollama.options_for(q)
    assert od.options_for(q) == options
    expected = "<bos>" + run_ollama.render({"state": STATE, "question": q}, run_ollama.labels_for(len(options)),
                                           options, "gemma4")
    assert od.prompt_for(STATE, q, options) == expected


def test_first_spelling_of_each_letter_wins():
    top = [{"token": " A", "logprob": -0.36}, {"token": " B", "logprob": -2.74},
           {"token": "A", "logprob": -5.24}, {"token": "B", "logprob": -6.0}]
    probs, mass = od.option_probs(top, [("true", "yes"), ("false", "no")])
    assert probs["true"] > 0.9
    assert mass == pytest.approx(math.exp(-0.36) + math.exp(-2.74))


def test_script_end_to_end_against_a_mock_ollama():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            top = [{"token": " A", "logprob": -0.1}, {"token": "A", "logprob": -7.0}, {"token": " B", "logprob": -2.5}]
            out = json.dumps({"logprobs": [{"top_logprobs": top}], "prompt_eval_count": 50,
                              "total_duration": 5e7}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(out)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        out = subprocess.run([sys.executable, str(ROOT / "examples" / "ollama_decide.py"),
                              "--host", f"http://127.0.0.1:{httpd.server_address[1]}", "--state", '{"x": 1}',
                              "--question", "Urgent?", "--yesno"], capture_output=True, text=True, timeout=60)
    finally:
        httpd.shutdown()
    assert out.returncode == 0, out.stderr
    assert "answer: yes" in out.stdout
    req = seen[0]
    assert req["raw"] is True and req["logprobs"] is True and req["prompt"].startswith("<bos>")
    assert req["options"]["num_predict"] == 1 and req["prompt"].endswith("Answer:")
