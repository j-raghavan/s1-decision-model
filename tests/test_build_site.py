"""The project site builds from the committed results, shows the right headline numbers, and stays clean.

    uv run --no-project --with pytest pytest -q tests/test_build_site.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
def _load_builder():
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mean_accuracy(suite: str, run_id: str) -> float:
    """Independent recomputation: mean headline over the non-score slices of the scores file."""
    payload = json.loads((ROOT / "results" / suite / f"{run_id}.scores.json").read_text(encoding="utf-8"))
    return statistics.fmean(e["headline"]["value"] for e in payload["slices"].values() if e["task_type"] != "score")


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("site") / "_site"
    _load_builder().build(out)
    return out


def _row(html: str, table_id: str, run_id: str) -> str:
    table = re.search(rf'<table[^>]*id="{table_id}".*?</table>', html, re.S)
    assert table, f"table {table_id} missing"
    row = re.search(rf'<tr data-run="{re.escape(run_id)}".*?</tr>', table.group(0), re.S)
    assert row, f"row {run_id} missing from {table_id}"
    return row.group(0)


def _mean_cell(row: str) -> str:
    return re.search(r'<span class="val">([^<]+)</span>', row).group(1)


def test_pages_exist(site: Path):
    for name in ("index.html", "leaderboard.html", "assets/style.css", "assets/site.js"):
        assert (site / name).is_file(), name


@pytest.mark.parametrize("run_id", ["g26-bos-vllm", "jev-1.13.0"])
def test_jevbench_rows(site: Path, run_id: str):
    html = (site / "leaderboard.html").read_text(encoding="utf-8")
    assert _mean_cell(_row(html, "lb-jevbench", run_id)) == f"{_mean_accuracy('jevbench', run_id):.3f}"


def test_custom_s1_row(site: Path):
    html = (site / "leaderboard.html").read_text(encoding="utf-8")
    assert _mean_cell(_row(html, "lb-custom", "g26-bos-vllm")) == f"{_mean_accuracy('custom', 'g26-bos-vllm'):.3f}"


def test_index_headline_numbers(site: Path):
    html = (site / "index.html").read_text(encoding="utf-8")
    for run_id in ("g26-bos-vllm", "jev-1.13.0"):
        assert f"{_mean_accuracy('jevbench', run_id):.3f}" in html
    assert "s1 (Gemma 4 26B-A4B + s1 fine-tune)" in (site / "leaderboard.html").read_text(encoding="utf-8")
    assert "Jev 1.13 (TypeSafe AI, closed)" in (site / "leaderboard.html").read_text(encoding="utf-8")


def test_int4_row_only_when_scored(site: Path):
    html = (site / "leaderboard.html").read_text(encoding="utf-8")
    present = (ROOT / "results" / "jevbench" / "s1-int4-ollama.scores.json").exists()
    assert ('data-run="s1-int4-ollama"' in html) == present


def test_not_affiliated_statement(site: Path):
    for name in ("index.html", "leaderboard.html"):
        assert "not affiliated" in (site / name).read_text(encoding="utf-8")
