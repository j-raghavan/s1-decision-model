"""Sample a JevBench dev split for checkpoint selection, disjoint from the test subset.

Takes N families per text slice from the families the test subset did not use (build_subset.py), with
its own seed, so a training run can pick checkpoints and settle the data mix on dev and report the test
subset once. Cases whose state text also appears in the training targets or in the test subset are dropped, so dev
measures generalisation rather than recall and never shares a case with test. SST-5 sentences are restored as in build_subset.py.

    uv run eval/jevbench/build_dev.py --families 50
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys

from build_subset import restore_sst5
from paths import CASES, ROOT, SUBSET, TEXT_SLICES, read_jsonl, write_jsonl

DEV = ROOT / "data" / "eval" / "jevbench_dev.jsonl"
TRAINING = [ROOT / "data" / "train" / "targets_v2.jsonl", ROOT / "data" / "train" / "injection_v1.jsonl"]


def texts(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from texts(v)


def norm_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:20]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", type=int, default=50, help="families sampled per slice")
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()

    test_cases = read_jsonl(SUBSET)
    test_families = {c["family_id"] for c in test_cases}
    # different families can still carry the same text; such cases would let dev peek at test
    test_texts = {norm_hash(t) for c in test_cases for t in texts(c.get("state")) if len(t) >= 40}
    test_states = {json.dumps(c.get("state"), sort_keys=True) for c in test_cases}
    trained = set()
    for path in TRAINING:
        for line in path.open(encoding="utf-8"):
            trained |= {norm_hash(t) for t in texts(json.loads(line)["state"]) if len(t) >= 40}
    rng = random.Random(args.seed)
    dev: list[dict] = []
    for slice_key in TEXT_SLICES:
        families: dict[str, list[dict]] = collections.defaultdict(list)
        for case in read_jsonl(CASES / f"{slice_key}.jsonl"):
            if case["family_id"] not in test_families:
                families[case["family_id"]].append(case)
        chosen = sorted(families)
        rng.shuffle(chosen)
        picked = [c for fid in sorted(chosen[: args.families]) for c in families[fid]]
        if slice_key == "sst5":
            restore_sst5(picked)
        clean = [c for c in picked
                 if not any(norm_hash(t) in trained or norm_hash(t) in test_texts for t in texts(c.get("state")) if len(t) >= 40)
                 and json.dumps(c.get("state"), sort_keys=True) not in test_states]
        dev.extend(clean)
        print(f"{slice_key:26s} {min(args.families, len(families)):4d} families  {len(clean):5d} cases"
              f"  ({len(picked) - len(clean)} dropped: text also in training data or the test subset)")
    overlap = test_families & {c["family_id"] for c in dev}
    if overlap:
        raise SystemExit(f"dev shares {len(overlap)} families with the test subset")
    write_jsonl(DEV, dev)
    print(f"total {len(dev)} cases -> {DEV} (0 families shared with the test subset)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
