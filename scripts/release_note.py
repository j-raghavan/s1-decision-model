"""Render the GitHub release note for a milestone from results/releases/v<version>.json, after checking its numbers.

Each number in the file must match the committed results: accuracy and ECE against results/jevbench/<run>.scores.json
(the mean over the 11 accuracy slices, as score.py reports it), and a calibrated accuracy against its row in
results/calibration_from_dev.md. The release workflow runs this for every release file whose tag does not exist yet.

    python3 scripts/release_note.py results/releases/v1.0.0.json      # prints the note, or exits 1 on a mismatch
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = "https://huggingface.co/j-raghavan/s1-gemma4-26b-decision"


def measured(run: str) -> tuple[float, float]:
    """Mean accuracy and mean ECE over the accuracy slices of a scored run, rounded as score.py prints them."""
    slices = json.loads((ROOT / "results" / "jevbench" / f"{run}.scores.json").read_text())["slices"].values()
    acc = [s["headline"]["value"] for s in slices if s["task_type"] != "score" and "headline" in s]
    ece = [s["metrics"]["ece_10_bins"] for s in slices if s["metrics"].get("ece_10_bins") is not None]
    return round(statistics.fmean(acc), 3), round(statistics.fmean(ece), 3)


def check(release: dict, path: Path) -> list[str]:
    problems = []
    if path.stem != f"v{release['version']}":
        problems.append(f"{path.name}: version {release['version']} does not match the file name")
    table = (ROOT / "results" / "calibration_from_dev.md").read_text()
    for row in release["jevbench"]:
        acc, ece = measured(row["run"])
        if (row["accuracy"], row["ece"]) != (acc, ece):
            problems.append(f"{row['system']}: file says {row['accuracy']} / {row['ece']}, {row['run']} scored {acc} / {ece}")
        cal = row.get("calibrated")
        if cal:  # row: "<system> jevbench  raw / ece  per-task / ece  default / ece  [...]"; per-task is the claim
            line = next((ln for ln in table.splitlines() if ln.startswith(cal["row"] + " ") and " jevbench " in ln), "")
            numbers = line.split(" jevbench ", 1)[-1].split()
            if len(numbers) < 4 or numbers[3] != f"{cal['accuracy']:.3f}":
                problems.append(f"{row['system']}: calibrated {cal['accuracy']} is not the per-task value in the "
                                f"'{cal['row']}' jevbench row of calibration_from_dev.md")
    return problems


def render(release: dict) -> str:
    lines = [
        "s1 is my open, calibrated decision model: it picks an option, answers yes or no, or scores on a scale, "
        "with a probability, in one forward pass.",
        "",
        f"Model: {MODEL_URL}",
        "",
        "**JevBench (1,700 cases)**",
        "",
        "| | Accuracy | ECE |",
        "| --- | --- | --- |",
    ]
    for row in release["jevbench"]:
        acc = f"{row['accuracy']:.3f}"
        if row.get("calibrated"):
            acc += f" ({row['calibrated']['accuracy']:.3f} calibrated)"
        lines.append(f"| {row['system']} | {acc} | {row['ece']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    path = Path(sys.argv[1])
    release = json.loads(path.read_text())
    problems = check(release, path)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(render(release), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
