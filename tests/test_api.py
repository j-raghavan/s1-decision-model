"""The /v1/decisions API: request validation, answer shapes, calibration and health, with the model call stubbed
out, plus one end-to-end test through the real Ollama client against an in-process mock server."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import api.server as server  # noqa: E402


@pytest.fixture
def client(monkeypatch, tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text('{"choice": {"temperature": 1.0}, "score": {"temperature": 1.0}, "noul": {"a": 1.0, "b": 0.0}}')
    monkeypatch.setattr(server, "CALIBRATION", cal)

    def fake_score(client, model, case, fetch=None):
        assert fetch is server.FETCH  # the API must use the <bos>-prepending fetcher
        q = case["question"]
        if q["type"] == "noul":
            by_key = {"true": 0.8, "false": 0.2}
        elif q["type"] == "choice":
            keys = list(q["criteria"])
            by_key = {k: (0.7 if i == 0 else 0.3 / (len(keys) - 1)) for i, k in enumerate(keys)}
        else:
            by_key = {str(i): (0.6 if i == 1 else 0.4 / (len(q["criteria"]) - 1)) for i in range(len(q["criteria"]))}
        return {"by_key": by_key, "label_mass": 1.0, "calls": 1, "input_tokens": 10}

    monkeypatch.setattr(server, "score_case", fake_score)
    return TestClient(server.app)


def test_all_question_types(client):
    r = client.post("/v1/decisions", json={"state": {"ticket": "charged twice"}, "questions": {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": "Bugs"}},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high"]}}})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == server.VERSION
    a = body["answers"]
    assert a["team"]["choice"] == "billing" and abs(sum(a["team"]["probabilities"].values()) - 1) < 1e-9
    assert a["urgent"]["noul"] == pytest.approx(0.8)
    assert a["severity"]["score"] == pytest.approx(0 * 0.2 + 1 * 0.6 + 2 * 0.2)


def test_rejects_bad_questions(client):
    r = client.post("/v1/decisions", json={"state": {}, "questions": {
        "x": {"type": "choice", "instructions": "?", "criteria": {"only": "one"}}}})
    assert r.status_code == 422


def test_healthz_reports_model_and_version(client, monkeypatch):
    monkeypatch.setattr(server, "ollama_status", lambda: (True, "ok"))
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] and r.json()["version"] == server.VERSION


def test_healthz_503_when_ollama_is_down(client, monkeypatch):
    monkeypatch.setattr(server, "ollama_status", lambda: (False, "Ollama unreachable"))
    r = client.get("/healthz")
    assert r.status_code == 503 and r.json()["ok"] is False


@pytest.mark.parametrize("bad", ['{"choice": {"temperature": 0}, "score": {"temperature": 1}, "noul": {"a": 1, "b": 0}}',
                                 '{"choice": {"temperature": 1}, "score": {"temperature": 1}}',
                                 '{"choice": {"temperature": "NaN"}, "score": {"temperature": 1}, "noul": {"a": 1, "b": 0}}'])
def test_invalid_calibration_is_reported_not_crashed(client, monkeypatch, tmp_path, bad):
    cal = tmp_path / "bad.json"
    cal.write_text(bad)
    monkeypatch.setattr(server, "CALIBRATION", cal)
    monkeypatch.setattr(server, "ollama_status", lambda: (True, "ok"))
    r = client.post("/v1/decisions", json={"state": {}, "questions": {
        "x": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}}}})
    assert r.status_code == 500 and "calibration is invalid" in r.json()["detail"]
    assert client.get("/healthz").status_code == 503


def test_missing_yes_no_probability_is_a_502(client, monkeypatch):
    monkeypatch.setattr(server, "score_case", lambda *a, **k: {"by_key": {"maybe": 1.0}, "input_tokens": 1})
    r = client.post("/v1/decisions", json={"state": {}, "questions": {"q": {"type": "noul", "instructions": "?"}}})
    assert r.status_code == 502


def test_end_to_end_through_the_ollama_client(monkeypatch, tmp_path):
    """Real score_case and fetcher against a mock Ollama: the prompt must reach the model with exactly one <bos>."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import run_ollama

    prompts = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = json.dumps({"models": [{"name": f"{server.MODEL}:latest"}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompts.append(req["prompt"])
            top = [{"token": " A", "logprob": -0.1}, {"token": " B", "logprob": -2.5}, {"token": " C", "logprob": -4.0}]
            body = json.dumps({"logprobs": [{"top_logprobs": top}], "prompt_eval_count": 42}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/api/generate"
    monkeypatch.setattr(run_ollama, "OLLAMA", url)
    monkeypatch.setattr(server, "OLLAMA", url)
    cal = tmp_path / "cal.json"
    cal.write_text('{"choice": {"temperature": 1.0}, "score": {"temperature": 1.0}, "noul": {"a": 1.0, "b": 0.0}}')
    monkeypatch.setattr(server, "CALIBRATION", cal)
    try:
        c = TestClient(server.app)
        assert c.get("/healthz").status_code == 200
        r = c.post("/v1/decisions", json={"state": {"ticket": "charged twice"}, "questions": {
            "team": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": "Payments", "shipping": "Deliveries", "tech": "Bugs"}}}})
    finally:
        httpd.shutdown()
    assert r.status_code == 200, r.text
    ans = r.json()["answers"]["team"]
    assert ans["choice"] == "billing" and abs(sum(ans["probabilities"].values()) - 1) < 1e-9
    assert prompts and all(p.startswith("<bos>") and not p.startswith("<bos><bos>") for p in prompts)
    assert prompts[0].endswith("Answer:") and "billing: Payments" in prompts[0]
