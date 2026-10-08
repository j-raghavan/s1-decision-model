"""Deployable calibration: calibrators fitted on the dev splits, scored on the test sets.

calibrate.py cross-fits each slice's calibrator on the test set itself (fair for comparing systems, but a
deployed model never sees test labels). Here every calibrator is fitted only on dev predictions of the same
system and applied to its test predictions:

  per-task  one calibrator per slice (Platt for yes/no, temperature for choice), as a deployment would fit for a
            known task from a few hundred labelled examples
  default   one calibrator per answer type pooled over all dev slices, for tasks it has never seen

    uv run eval/jevbench/calibrate_from_dev.py
"""

from __future__ import annotations

import statistics

from calibrate import as_logits, fit_platt, fit_temperature, metrics, softmax
from paths import SUITES, read_jsonl

# system -> (dev run id, test run id); every pair scored through vLLM bf16
SYSTEMS = {
    "untuned": ("base-vllm", "gemma4-26b-bf16-vllm"),
    "pilot 2": ("pilot2-vllm", "g26-pilot2-vllm"),
    "option A (step 500)": ("F-step-500-vllm", "g26-fullA-vllm"),
    "untuned + <bos>": ("base-bos-vllm", "base-bos-vllm"),
    "<bos> fine-tune (step 1500)": ("bos-F-step-1500-vllm", "g26-bos-vllm"),
}
PAIRS = {"jevbench": "jevbench_dev", "custom": "custom_dev2"}


def items(suite: str, run: str) -> dict[str, list]:
    """slice -> [(log-probs, gold index, task type)] for yes/no and choice cases."""
    out: dict[str, list] = {}
    path = SUITES[suite]["runs"] / run / "predictions.jsonl"
    for rec in read_jsonl(path):
        if rec["task_type"] == "score" or rec["slice"] == "sst5":
            continue
        it = as_logits(rec)
        if it is not None:
            out.setdefault(rec["slice"], []).append((it[0], it[1], rec["task_type"]))
    return out


def fit(rows: list, task_type: str):
    pairs = [(z, g) for z, g, _ in rows]
    return (fit_platt if task_type == "noul" else fit_temperature)(pairs)


def main() -> int:
    print("Accuracy / ECE on the test sets with calibrators fitted on dev only (mean over slices).\n")
    print(f"{'system':22s} {'suite':9s} {'raw':>15s} {'per-task (dev)':>16s} {'default (dev)':>15s}  per-task fallbacks")
    for name, (dev_run, test_run) in SYSTEMS.items():
        for test_suite, dev_suite in PAIRS.items():
            dev, test = items(dev_suite, dev_run), items(test_suite, test_run)
            pooled = {t: fit([r for rows in dev.values() for r in rows if r[2] == t], t)
                      for t in {r[2] for rows in dev.values() for r in rows}}
            raw, task, default, fallbacks = [], [], [], []
            for s, rows in test.items():
                t = rows[0][2]
                cal = fit(dev[s], t) if len(dev.get(s, [])) >= 20 else None
                if cal is None:
                    fallbacks.append(s)
                raw.append(metrics([(softmax(z), g) for z, g, _ in rows]))
                task.append(metrics([((cal or pooled[t])(z), g) for z, g, _ in rows]))
                default.append(metrics([(pooled[t](z), g) for z, g, _ in rows]))
            fmt = lambda m: f"{statistics.mean(a for a, _ in m):.3f} / {statistics.mean(e for _, e in m):.3f}"  # noqa: E731
            print(f"{name:22s} {test_suite:9s} {fmt(raw):>15s} {fmt(task):>16s} {fmt(default):>15s}  {fallbacks or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
