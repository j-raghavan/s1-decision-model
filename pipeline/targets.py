"""Build soft training targets: calibrated teacher probabilities blended with gold labels.

For each task family, a calibrator is fitted on the teacher's distribution
against the gold answers (Platt scaling for yes/no, a temperature for choice and
score), 2-fold cross-fitted so no row is calibrated with parameters fitted on
itself. Families with fewer than --min-family rows fall back to the per-type
defaults in api/calibration.json.

    target = gold_weight * one_hot(gold) + (1 - gold_weight) * calibrated_teacher

The gold weight follows how reliable the gold label is (see gold_weight_for):
0.9 where gold is exact by construction (rule generators), 0.7 for objective
answer keys, 0.5 for subjective or crowd-noisy labels. The teacher keeps a say
in how plausible the other options are, but cannot outvote a known-correct answer.

Rows where the teacher puts less than --conflict probability on the gold answer
are kept but flagged (`teacher_conflict`), so noisy source labels can be audited.

    uv run pipeline/targets.py --rows data/train/structured_v1.jsonl data/train/sources_v1.jsonl \\
        --labels data/labels/*.labels.jsonl

Writes data/train/targets_v1.jsonl and prints per-family teacher agreement.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "jevbench"))
from calibrate import fit_platt, fit_temperature, softmax  # noqa: E402

OUT = ROOT / "data" / "train" / "targets_v1.jsonl"
DEFAULTS = ROOT / "api" / "calibration.json"
EPS = 1e-15


OBJECTIVE_KEYS = {"arc_science", "csqa_commonsense", "aqua_math", "dbpedia_topic", "clinc_intent", "massive_intent",
                  "boolq_passage_qa"}
SUBJECTIVE = {"civil_toxicity", "snli_nli", "wanli_nli"}


def gold_weight_for(family: str, provenance: dict) -> float:
    if family.startswith("syn_"):  # model-generated gold, already consensus-filtered against the teacher
        return 0.7
    if family.startswith("sni_"):  # dataset labels across many tasks, some subjective
        return 0.6
    if provenance.get("generator"):  # rule-generated: gold is exact by construction
        return 0.9
    if family in OBJECTIVE_KEYS:
        return 0.7
    if family in SUBJECTIVE:
        return 0.5
    raise SystemExit(f"no gold weight defined for family {family}; add it to targets.py")


def synthetic_holdout(family: str, frac: float = 0.1) -> bool:
    """A fixed ~10% of synthetic families are held out whole for unseen-family validation."""
    return int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def half(row_id: str) -> int:
    return hashlib.sha256(row_id.encode()).digest()[0] & 1


def keys_of(row: dict) -> list[str]:
    q = row["question"]
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


def gold_key(row: dict) -> str:
    g = row["gold"]
    if row["task_type"] == "noul":
        return "true" if g else "false"
    return str(g)


def logits(probs: dict[str, float], keys: list[str]) -> np.ndarray:
    p = np.clip(np.array([float(probs.get(k, 0.0)) for k in keys]), EPS, None)
    return np.log(p / p.sum())


def default_calibrator(task_type: str, cal: dict):
    if task_type == "noul":
        a, b = cal["noul"]["a"], cal["noul"]["b"]

        def apply(z: np.ndarray) -> np.ndarray:
            p = 1 / (1 + np.exp(-np.clip(a * (z[1] - z[0]) + b, -50, 50)))
            return np.array([1 - p, p])
        return apply
    t = cal[task_type]["temperature"]
    return lambda z: softmax(z / t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", nargs="+", type=Path, required=True)
    ap.add_argument("--conflict", type=float, default=0.1)
    ap.add_argument("--min-family", type=int, default=100)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    labels = {}
    for path in args.labels:
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            if not rec.get("error"):
                labels[rec["row_id"]] = rec
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    missing = 0
    for path in args.rows:
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row["row_id"] in labels:
                by_family[row["family"]].append(row)
            else:
                missing += 1
    defaults = json.loads(DEFAULTS.read_text(encoding="utf-8"))

    out_rows, report = [], []
    for family, rows in sorted(by_family.items()):
        items = []
        for r in rows:
            keys = keys_of(r)
            items.append((r, keys, logits(labels[r["row_id"]]["probs"], keys), keys.index(gold_key(r))))
        calibrated: dict[str, np.ndarray] = {}
        # Mixed-type families (an intent source yields both choice and noul rows) are calibrated per type.
        for t in sorted({r["task_type"] for r in rows}):
            group = [it for it in items if it[0]["task_type"] == t]
            fit = fit_platt if t == "noul" else fit_temperature
            for h in (0, 1):
                train = [(z, g) for r, _, z, g in group if half(r["row_id"]) != h]
                apply = fit(train) if len(train) >= args.min_family // 2 else default_calibrator(t, defaults)
                for r, _, z, _g in group:
                    if half(r["row_id"]) == h:
                        calibrated[r["row_id"]] = apply(z)
        agree = conflicts = 0
        dropped_consensus = 0
        if family.startswith("syn_"):
            # A generated family where generator and teacher agree on under half the rows has an ambiguous or
            # flawed policy; drop it whole rather than keep its few agreeing rows.
            consensus = sum(int(np.argmax(z)) == g for _, _, z, g in items) / len(items)
            if consensus < 0.5:
                report.append((family, 0, 0.0, 0, len(items)))
                continue
        for r, keys, z, g in items:
            if family.startswith("syn_") and int(np.argmax(z)) != g:
                dropped_consensus += 1  # generator and teacher disagree: the generated answer is not trusted
                continue
            p_cal = calibrated[r["row_id"]]
            onehot = np.eye(len(keys))[g]
            w = gold_weight_for(family, r["provenance"])
            target = w * onehot + (1 - w) * p_cal
            conflict = bool(p_cal[g] < args.conflict)
            agree += int(p_cal.argmax() == g)
            conflicts += int(conflict)
            out_rows.append({
                "row_id": r["row_id"], "family": family, "task_type": r["task_type"], "state": r["state"],
                "question": r["question"], "gold": r["gold"],
                "target": {k: round(float(v), 6) for k, v in zip(keys, target)},
                "teacher_calibrated": {k: round(float(v), 6) for k, v in zip(keys, p_cal)},
                "teacher": labels[r["row_id"]]["teacher"], "teacher_conflict": conflict, "gold_weight": w,
                "holdout_family": bool(r.get("holdout_family")) or (family.startswith("syn_") and synthetic_holdout(family)),
                "provenance": r["provenance"],
            })
        report.append((family, len(items) - dropped_consensus, agree / max(1, len(items) - dropped_consensus),
                       conflicts, dropped_consensus))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{'family':30s} {'rows':>6s} {'teacher=gold':>12s} {'conflicts':>9s} {'no_consensus':>12s}")
    for fam, n, acc, c, d in report:
        print(f"{fam:30s} {n:6d} {acc:12.3f} {c:9d} {d:12d}")
    syn = [x for x in report if x[0].startswith("syn_")]
    if syn:
        kept, dropped = sum(x[1] for x in syn), sum(x[4] for x in syn)
        print(f"synthetic: {len(syn)} families, kept {kept} rows, dropped {dropped} without generator/teacher consensus")
    print(f"total {len(out_rows)} targets -> {args.out}; {missing} rows had no teacher label and were skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
