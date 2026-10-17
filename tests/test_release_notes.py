"""Every milestone file in results/releases must match the committed results before CI publishes its release."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_note", ROOT / "scripts" / "release_note.py")
rn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rn)
RELEASES = sorted((ROOT / "results" / "releases").glob("v*.json"))


@pytest.mark.parametrize("path", RELEASES, ids=lambda p: p.name)
def test_release_numbers_match_the_results(path):
    release = json.loads(path.read_text())
    assert rn.check(release, path) == []
    note = rn.render(release)
    assert rn.MODEL_URL in note and all(f"{r['accuracy']:.3f}" in note for r in release["jevbench"])


def test_a_wrong_number_is_caught(tmp_path):
    release = json.loads(RELEASES[-1].read_text())
    release["jevbench"][0]["accuracy"] += 0.001
    path = tmp_path / RELEASES[-1].name
    assert rn.check(release, path)
