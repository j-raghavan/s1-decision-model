"""Turn the filtered Super-NaturalInstructions decision tasks into training rows.

Reads pipeline/sni_manifest.json (the audited task list: English, 2-20 fixed
answers, permissive instance licenses, no eval-set sources, no sources excluded
by the provenance policy) and the local clone of allenai/natural-instructions.

Each instance becomes a choice question: the task's written definition is the
instruction, the instance input is the state, and the task's answer set is the
options. Instances per task are capped so no task dominates, and a fixed set of
whole tasks is marked `holdout_family`, so validation measures generalization to
instructions the model has never seen rather than memorization.

    uv run pipeline/sni.py --per-task 800 --holdout-tasks 20

Writes data/train/sni_v2.jsonl (git-ignored).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNI = ROOT / "data" / "raw" / "sni" / "tasks"
MANIFEST = ROOT / "pipeline" / "sni_manifest.json"
OUT = ROOT / "data" / "train" / "sni_v2.jsonl"


def key_of(label: str) -> str:
    k = "".join(ch if ch.isalnum() else "_" for ch in label.strip().lower()).strip("_")
    return k[:40] or "empty"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-task", type=int, default=800)
    ap.add_argument("--holdout-tasks", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    files = [t["file"] for t in manifest["tasks"]]
    holdout = set(rng.sample(files, args.holdout_tasks))
    rows, skipped = [], collections.Counter()
    for entry in manifest["tasks"]:
        task = json.loads((SNI / entry["file"]).read_text(encoding="utf-8"))
        definition = " ".join(task["Definition"]).strip()
        instances = task["Instances"]
        labels = sorted({inst["output"][0].strip() for inst in instances if inst["output"]})
        keys = {}
        for lab in labels:
            k = key_of(lab)
            while k in keys.values():
                k += "_"
            keys[lab] = k
        if len(set(keys.values())) != len(labels) or not 2 <= len(labels) <= 20:
            skipped["label set unusable"] += 1
            continue
        rng.shuffle(instances)
        family = "sni_" + entry["file"].removesuffix(".json")
        made = 0
        for inst in instances:
            if made >= args.per_task:
                break
            if not inst["output"] or len({o.strip() for o in inst["output"]}) != 1:
                skipped["ambiguous gold"] += 1  # more than one accepted answer: not a single-choice decision
                continue
            gold = keys[inst["output"][0].strip()]
            criteria = dict(rng.sample([(keys[l], l) for l in labels], len(labels)))  # shuffled option order
            rows.append({
                "row_id": f"{family}-{made}", "family": family, "task_type": "choice",
                "state": {"input": inst["input"]},
                "question": {"type": "choice", "instructions": definition, "criteria": criteria},
                "gold": gold, "n_options": len(labels), "holdout_family": family in {"sni_" + f.removesuffix(".json") for f in holdout},
                "provenance": {"source": "Super-NaturalInstructions", "hub_id": "allenai/natural-instructions",
                               "task": entry["file"], "task_source": entry["source"], "license": entry["license"],
                               "origin": "AI2 collection; per-task source reviewed in pipeline/sni_manifest.json"},
            })
            made += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fams = {r["family"] for r in rows}
    held = {r["family"] for r in rows if r["holdout_family"]}
    print(f"{len(rows):,} rows from {len(fams)} tasks ({len(held)} held out as whole families, "
          f"{sum(r['holdout_family'] for r in rows):,} rows); skipped: {dict(skipped)}")
    print("option counts:", dict(sorted(collections.Counter(r["n_options"] for r in rows).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
