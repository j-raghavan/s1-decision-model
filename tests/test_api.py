"""The /v1/decisions API: request validation, answer shapes and calibration, with the model call stubbed out."""

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


def test_healthz_reports_model_and_version(client):
    h = client.get("/healthz").json()
    assert h["ok"] and h["version"] == server.VERSION
