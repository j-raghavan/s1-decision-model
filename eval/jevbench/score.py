"""Score every run on the subset with the upstream JevBench scorer and write a comparison.

Reference systems (Jev 1.13.0, Laya 421M) come from the predictions published
with the dataset, filtered to the subset's case ids, so all systems are scored
on identical cases with identical metric code.

    uv run eval/jevbench/score.py                  # JevBench subset
    uv run eval/jevbench/score.py --suite custom   # custom structured-state / PowerPoint set

Writes <scores>/<run_id>.scores.json and <scores>/summary.md for the suite.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys

from paths import HARNESS, REFERENCE_PREDICTIONS, ROOT, SUBSET_IDS, SUITES, read_jsonl, suite_slices, write_jsonl

sys.path.insert(0, str(HARNESS / "workspace"))
import common as C  # noqa: E402  (upstream harness, MIT)
import score_runs  # noqa: E402

REFERENCES = {"jev-1.13.0": "Jev 1.13.0 (closed API)", "laya-421m-en": "Laya 421M"}
SYSTEM_ORDER = ["jev-1.13.0", "laya-421m-en", "von-1.3", "gemma4-12b", "gemma4-26b", "gemma4-12b+26b", "s1-v1", "s1-v2", "s1-v2-final", "s1-v3", "gptoss-120b", "gemma4-31b", "gemma4-26b+31b", "inkling", "gemma4-26b+31b+inkling", "g26-hf-zeroshot", "g26-pilot", "gemma4-26b-bf16-vllm", "g26-pilot-vllm", "g26-pilot2-vllm", "g26-fullA-vllm", "base-bos-vllm", "g26-bos-vllm"]


def stage_references(ids: set[str], RUNS) -> None:
    for run_id in REFERENCES:
        src = REFERENCE_PREDICTIONS / run_id
        rows = [r for r in read_jsonl(src / "predictions.jsonl") if r["case_id"] in ids]
        write_jsonl(RUNS / run_id / "predictions.jsonl", rows)
        (RUNS / run_id / "meta.json").write_text((src / "meta.json").read_text(encoding="utf-8"), encoding="utf-8")


def score(run_id: str, RUNS, SCORES) -> dict:
    C.RUNS, C.ROOT = RUNS, ROOT
    argv, sys.argv = sys.argv, ["score_runs.py", "--run-id", run_id]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            score_runs.main()
    finally:
        sys.argv = argv
    payload = json.loads((RUNS / run_id / "scores.json").read_text(encoding="utf-8"))
    (SCORES / f"{run_id}.scores.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def cell(entry: dict | None) -> tuple[str, float | None, float | None]:
    if not entry or "headline" not in entry:
        return "n/a", None, None
    h, m = entry["headline"], entry["metrics"]
    ece = m.get("ece_10_bins")
    value = h["value"]
    text = f"{value:.3f}" + (f" / {ece:.3f}" if ece is not None else "")
    if entry["coverage"] < 1:
        text += f" ({entry['coverage']*100:.0f}% cov)"
    return text, value, ece


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="jevbench", choices=sorted(SUITES))
    args = ap.parse_args()
    RUNS, SCORES = SUITES[args.suite]["runs"], SUITES[args.suite]["scores"]
    TEXT_SLICES = suite_slices(args.suite)
    SCORES.mkdir(parents=True, exist_ok=True)
    if args.suite == "jevbench":
        ids = set(SUBSET_IDS.read_text(encoding="utf-8").split())
        stage_references(ids, RUNS)
    else:
        ids = {c["case_id"] for c in read_jsonl(SUITES[args.suite]["cases"])}

    runs = [p.name for p in RUNS.iterdir() if (p / "predictions.jsonl").exists() and not p.name.startswith("smoke")]
    runs.sort(key=lambda r: (SYSTEM_ORDER.index(r) if r in SYSTEM_ORDER else 99, r))
    results = {r: score(r, RUNS, SCORES) for r in runs}

    header = "| slice | type | " + " | ".join(runs) + " |"
    intro = {
        "jevbench": f"{len(ids)} cases: 100 families per text slice, seed 20261004 (`eval/jevbench/build_subset.py`). "
                    "Jev and Laya are the predictions published with the dataset, filtered to the same cases.",
        "custom": f"{len(ids)} rule-labelled cases from `eval/custom/generate.py` (structured-state and PowerPoint "
                  "families, balanced answers). Evaluation only.",
    }[args.suite]
    lines = [
        f"# {args.suite} results",
        "",
        intro + " Each cell is headline / ECE. Headline is accuracy (higher is better), except score slices, where it is "
        "MAE of the expected level (lower is better) and the upstream scorer reports no ECE. ECE uses 10 equal-width bins "
        "on the top probability.",
        "",
        header,
        "|" + " --- |" * (len(runs) + 2),
    ]
    acc_by_run: dict[str, list[float]] = {r: [] for r in runs}
    ece_by_run: dict[str, list[float]] = {r: [] for r in runs}
    for slice_key in TEXT_SLICES:
        task_type = next((p["slices"][slice_key]["task_type"] for p in results.values() if slice_key in p["slices"]), "?")
        row = [slice_key, task_type]
        for r in runs:
            text, value, ece = cell(results[r]["slices"].get(slice_key))
            row.append(text)
            if value is not None and task_type != "score":
                acc_by_run[r].append(value)
            if ece is not None:
                ece_by_run[r].append(ece)
        lines.append("| " + " | ".join(row) + " |")

    def mean(xs: list[float], n: int) -> str:
        return f"{statistics.fmean(xs):.3f}" if len(xs) == n and xs else "n/a"

    # the accuracy slices a complete run covers; partial runs (e.g. 2-slice diagnostics) show n/a instead of
    # shrinking the count for every run
    n_acc = max((sum(1 for e in p["slices"].values() if e["task_type"] != "score") for p in results.values()), default=0)
    lines.append(f"| **mean accuracy ({n_acc} slices)** | | " + " | ".join(mean(acc_by_run[r], n_acc) for r in runs) + " |")
    lines.append(f"| **mean ECE ({n_acc} slices)** | | " + " | ".join(mean(ece_by_run[r], n_acc) for r in runs) + " |")

    lines += ["", "## Option-order sensitivity", "",
              "Share of families whose correctness flips when the options are reordered (lower is better).", "",
              header, "|" + " --- |" * (len(runs) + 2)]
    for slice_key in TEXT_SLICES:
        row, any_bias = [slice_key, ""], False
        for r in runs:
            pb = (results[r]["slices"].get(slice_key) or {}).get("position_bias")
            if pb and pb.get("flipped_by_reordering") is not None:
                any_bias = True
                row.append(f"{pb['flipped_by_reordering']:.3f}")
            else:
                row.append("n/a")
        if any_bias:
            lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Latency", "",
              "Median latency per request in ms. Jev is a network API call (concurrency 32); Laya ran on a local "
              "server of unrecorded hardware at concurrency 8, so its numbers include queueing; local runs are "
              "sequential on an Apple M5 MacBook Air (32 GB, fanless). Not directly comparable across columns.", "",
              header, "|" + " --- |" * (len(runs) + 2)]
    for slice_key in TEXT_SLICES:
        row = [slice_key, ""]
        for r in runs:
            rt = (results[r]["slices"].get(slice_key) or {}).get("runtime") or {}
            row.append(f"{rt['latency_ms_p50']:.0f}" if rt.get("latency_ms_p50") else "n/a")
        lines.append("| " + " | ".join(row) + " |")

    (SCORES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
