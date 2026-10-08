"""Shared paths for the JevBench runs.

The upstream harness and dataset are downloaded into data/raw/ (git-ignored):

    git clone https://github.com/Leanmcp/jevbench data/raw/jevbench-harness
    hf download Leanmcp/jevbench --repo-type dataset --local-dir data/raw/jevbench-data --exclude "images/*"
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
HARNESS = RAW / "jevbench-harness" / "jev_bench"
DATASET = RAW / "jevbench-data"
CASES = DATASET / "cases"
REFERENCE_PREDICTIONS = DATASET / "predictions"

SUBSET = ROOT / "data" / "eval" / "jevbench_subset.jsonl"
SUBSET_IDS = ROOT / "results" / "jevbench" / "subset_case_ids.txt"
RUNS = ROOT / "results" / "raw" / "jevbench"
SCORES = ROOT / "results" / "jevbench"

# The 12 text slices scored in the JevBench paper. The two image slices are skipped.
TEXT_SLICES = [
    "aegis2",
    "aegis2_response",
    "atbench500",
    "banking77",
    "jailbreak_classification",
    "medmcqa",
    "medqa_usmle",
    "mmlu_pro",
    "prompt_injections",
    "pubmedqa",
    "scienceqa_text",
    "sst5",
]


# An eval suite is a case file plus where its runs and scores live. slices=None means "every slice in the cases".
SUITES = {
    "jevbench": {"cases": SUBSET, "runs": RUNS, "scores": SCORES, "slices": TEXT_SLICES},
    "custom": {"cases": ROOT / "data" / "eval" / "custom_v0.jsonl", "runs": ROOT / "results" / "raw" / "custom",
               "scores": ROOT / "results" / "custom", "slices": None},
    # Dev splits, disjoint from the test sets above: for checkpoint selection and data-mix decisions only.
    "jevbench_dev": {"cases": ROOT / "data" / "eval" / "jevbench_dev.jsonl", "runs": ROOT / "results" / "raw" / "jevbench_dev",
                     "scores": ROOT / "results" / "jevbench_dev", "slices": TEXT_SLICES},
    "custom_dev": {"cases": ROOT / "data" / "eval" / "custom_dev.jsonl", "runs": ROOT / "results" / "raw" / "custom_dev",
                   "scores": ROOT / "results" / "custom_dev", "slices": None},
    # 2,341 cases (1,851 yes/no and choice), new seed, no case shared with test: sized to measure a 1.5-point change
    "custom_dev2": {"cases": ROOT / "data" / "eval" / "custom_dev_v2.jsonl", "runs": ROOT / "results" / "raw" / "custom_dev2",
                    "scores": ROOT / "results" / "custom_dev2", "slices": None},
}


def suite_slices(suite: str) -> list[str]:
    cfg = SUITES[suite]
    return cfg["slices"] or sorted({c["slice"] for c in read_jsonl(cfg["cases"])})


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_record(case: dict, run_id: str) -> dict:
    """Fields every prediction record carries, matching the harness format."""
    return {
        "run_id": run_id,
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "slice": case["slice"],
        "source": case["source"],
        "task_type": case["task_type"],
        "perturbation": case["perturbation"],
        "n_options": case["n_options"],
        "gold": case["gold"],
        "truncated_state": case["truncated"],
    }
