"""How much accuracy and calibration a per-slice post-hoc calibrator recovers.

For each run and slice, the families are split in two by a fixed hash. A
calibrator is fitted on one half and applied to the other, then the halves swap
(2-fold cross-fitting), so no case is scored with a calibrator fitted on it.

  choice  one temperature on the option log-probabilities (never changes the answer)
  noul    Platt scaling: a scale and a bias on the log-odds, which can move the
          yes/no threshold and so can change the answer

Probabilities are kept at full precision (clipped only at 1e-15, for exact zeros):
an LLM's saturated probabilities still rank cases through their tiny values, and
clipping them at 1e-6 throws that signal away.

    uv run eval/jevbench/calibrate.py [--suite custom]

Writes results/jevbench/calibration.md. sst5 is skipped, as the upstream scorer
reports no ECE for it.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics

import numpy as np

from paths import SUITES, read_jsonl, suite_slices

TEMPS = np.exp(np.linspace(np.log(0.25), np.log(50.0), 200))
EPS = 1e-15
SYSTEMS = ["jev-1.13.0", "laya-421m-en", "von-1.3", "gemma4-12b", "gemma4-26b", "gemma4-12b+26b", "s1-v1", "s1-v2", "s1-v2-final", "s1-v3", "gptoss-120b", "gemma4-31b", "gemma4-26b+31b", "inkling", "gemma4-26b+31b+inkling", "g26-hf-zeroshot", "g26-pilot", "gemma4-26b-bf16-vllm", "g26-pilot-vllm", "g26-pilot2-vllm", "g26-fullA-vllm", "base-bos-vllm", "g26-bos-vllm"]


def half(family_id: str) -> int:
    return hashlib.sha256(family_id.encode()).digest()[0] & 1


def as_logits(rec: dict) -> tuple[np.ndarray, int] | None:
    """Log-probabilities over options and the gold index; noul becomes [false, true]."""
    if rec.get("error") or rec.get("pred") is None:
        return None
    if rec["task_type"] == "noul":
        p = min(max(float(rec["p_true"]), EPS), 1 - EPS)
        return np.log([1 - p, p]), int(bool(rec["gold"]))
    probs = rec.get("probabilities")
    if not probs or str(rec["gold"]) not in probs:
        return None
    keys = list(probs)
    p = np.clip(np.array([float(probs[k]) for k in keys]), EPS, None)
    return np.log(p / p.sum()), keys.index(str(rec["gold"]))


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def fit_temperature(items: list[tuple[np.ndarray, int]]):
    def nll(t: float) -> float:
        return -sum(np.log(softmax(z / t)[g] + 1e-300) for z, g in items) / len(items)

    t = float(min(TEMPS, key=nll))
    return lambda z: softmax(z / t)


def fit_platt(items: list[tuple[np.ndarray, int]], l2: float = 1e-2):
    """Logistic regression of gold on the log-odds, by Newton's method with a small L2 penalty."""
    x = np.array([z[1] - z[0] for z, _ in items])
    y = np.array([float(g) for _, g in items])
    X = np.c_[x, np.ones_like(x)]
    w = np.array([0.1, 0.0])
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -50, 50)))
        grad = X.T @ (p - y) / len(y) + l2 * w
        hess = (X.T * (p * (1 - p))) @ X / len(y) + l2 * np.eye(2)
        w -= np.linalg.solve(hess, grad)

    def apply(z: np.ndarray) -> np.ndarray:
        p = 1 / (1 + np.exp(-np.clip(w[0] * (z[1] - z[0]) + w[1], -50, 50)))
        return np.array([1 - p, p])

    return apply


def metrics(scored: list[tuple[np.ndarray, int]], bins: int = 10) -> tuple[float, float]:
    """Accuracy and ECE on the top probability; scored holds (probability vector, gold index)."""
    conf = np.array([p.max() for p, _ in scored])
    correct = np.array([float(p.argmax() == g) for p, g in scored])
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        mask = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.sum() / len(conf) * abs(correct[mask].mean() - conf[mask].mean())
    return float(correct.mean()), total


def calibrate_slice(rows: list[tuple[str, tuple]], task_type: str) -> tuple[tuple, tuple]:
    fit = fit_platt if task_type == "noul" else fit_temperature
    raw = [(softmax(z), g) for _, (z, g) in rows]
    cal = []
    for h in (0, 1):
        train = [it for fid, it in rows if half(fid) != h]
        if not train:
            continue
        apply = fit(train)
        cal += [(apply(z), g) for fid, (z, g) in rows if half(fid) == h]
    return metrics(raw), metrics(cal)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="jevbench", choices=sorted(SUITES))
    args = ap.parse_args()
    RUNS, SCORES = SUITES[args.suite]["runs"], SUITES[args.suite]["scores"]
    slices = [s for s in suite_slices(args.suite) if s != "sst5"]
    systems = [s for s in SYSTEMS if (RUNS / s / "predictions.jsonl").exists()]
    out: dict[str, dict[str, tuple]] = {}
    for system in systems:
        by_slice: dict[str, list] = {}
        slice_type: dict[str, str] = {}
        for rec in read_jsonl(RUNS / system / "predictions.jsonl"):
            if rec["slice"] not in slices or rec["task_type"] == "score":
                continue
            slice_type[rec["slice"]] = rec["task_type"]
            item = as_logits(rec)
            if item is not None:
                by_slice.setdefault(rec["slice"], []).append((rec["family_id"], item))
        out[system] = {s: calibrate_slice(rows, slice_type[s]) for s, rows in by_slice.items()}
        out[system]["_types"] = slice_type

    lines = [
        "# Post-hoc calibration test",
        "",
        "Accuracy / ECE (10 bins) before → after one calibrator per slice, 2-fold cross-fitted by family "
        "(`eval/jevbench/calibrate.py`). Choice slices use a temperature, which never changes the answer; "
        "yes/no slices use Platt scaling (scale + bias on the log-odds), which can move the yes/no threshold. "
        "Probabilities are kept at full precision. Score (ordinal) slices are excluded.",
        "",
        "| slice | " + " | ".join(systems) + " |",
        "|" + " --- |" * (len(systems) + 1),
    ]
    slices = [s for s in slices if any(s in out[system] for system in systems)]
    for s in slices:
        row = [s]
        for system in systems:
            if s in out[system]:
                (a0, e0), (a1, e1) = out[system][s]
                row.append(f"{a0:.3f} / {e0:.3f} → {a1:.3f} / {e1:.3f}")
            else:
                row.append("n/a")
        lines.append("| " + " | ".join(row) + " |")
    slices = [s for s in slices if any(s in out[system] for system in systems)]
    types = {s: t for system in systems for s, t in out[system]["_types"].items()}
    for label, keep in (("**mean, yes/no slices**", "noul"), ("**mean, all slices**", None)):
        wanted = [s for s in slices if keep is None or types.get(s) == keep]
        row = [label]
        for system in systems:
            rows = [out[system][s] for s in wanted if s in out[system]]
            if wanted and len(rows) == len(wanted):
                m = [statistics.fmean(r[i][j] for r in rows) for i in (0, 1) for j in (0, 1)]
                row.append(f"{m[0]:.3f} / {m[1]:.3f} → {m[2]:.3f} / {m[3]:.3f}")
            else:
                row.append("n/a")
        lines.append("| " + " | ".join(row) + " |")

    SCORES.mkdir(parents=True, exist_ok=True)
    (SCORES / "calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
