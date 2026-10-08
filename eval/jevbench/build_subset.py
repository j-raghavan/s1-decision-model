"""Sample a fixed subset of JevBench text cases for local benchmarking.

Takes N families per slice (a family is a case plus its option-reordered twin,
when the slice has one), with a fixed seed, so every system is scored on the
same cases. SST-5 ships without its sentence text; it is restored from the
pinned SetFit/sst5 revision and checked against each case's state_hash.

    uv run eval/jevbench/build_subset.py --families 100
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import urllib.request

from paths import CASES, SUBSET, SUBSET_IDS, TEXT_SLICES, read_jsonl, write_jsonl

SST5_URL = "https://huggingface.co/datasets/SetFit/sst5/resolve/{rev}/test.jsonl"


def state_hash(state: dict) -> str:
    # Mirrors jev_bench/workspace/common.py: state_json + content_hash.
    text = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def restore_sst5(cases: list[dict]) -> None:
    revision = cases[0]["revision"]
    with urllib.request.urlopen(SST5_URL.format(rev=revision)) as resp:
        rows = [json.loads(line) for line in resp.read().decode("utf-8").splitlines() if line.strip()]
    for case in cases:
        state = {"sentence": rows[case["row_index"]]["text"]}
        if state_hash(state) != case["state_hash"]:
            raise SystemExit(f"SST-5 hash mismatch for {case['case_id']}; check state_json convention")
        case["state"] = state
        case.pop("state_withheld", None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", type=int, default=100, help="families sampled per slice")
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    subset: list[dict] = []
    for slice_key in TEXT_SLICES:
        cases = read_jsonl(CASES / f"{slice_key}.jsonl")
        families: dict[str, list[dict]] = collections.defaultdict(list)
        for case in cases:
            families[case["family_id"]].append(case)
        chosen = sorted(families)
        rng.shuffle(chosen)
        picked = [c for fid in sorted(chosen[: args.families]) for c in families[fid]]
        if slice_key == "sst5":
            restore_sst5(picked)
        subset.extend(picked)
        print(f"{slice_key:26s} {min(args.families, len(families)):4d} families  {len(picked):5d} cases")

    write_jsonl(SUBSET, subset)
    SUBSET_IDS.parent.mkdir(parents=True, exist_ok=True)
    SUBSET_IDS.write_text("".join(c["case_id"] + "\n" for c in subset), encoding="utf-8")
    print(f"total {len(subset)} cases -> {SUBSET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
