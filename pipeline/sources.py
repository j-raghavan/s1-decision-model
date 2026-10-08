"""Turn the v1 public training sources into typed decision rows with their gold labels.

Each row has the same shape as an eval case (state, question, task_type, gold),
so the teacher-labeling runner can score it unchanged, plus a provenance record:
source, Hub id, license, origin and the source row it came from.

Two guards run before anything is written:
  1. Every source must be listed in lineage.yaml with `use: train`. A source
     marked `eval-only`, or missing, is refused.
  2. No row may share a text field with the eval sets (JevBench subset and the
     custom set); any overlap aborts the build.

    uv run --extra data pipeline/sources.py                # all sources, default caps
    uv run --extra data pipeline/sources.py --scale 0.01   # quick smoke build

Writes data/train/sources_v1.jsonl (git-ignored; stays on this machine).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_lineage import parse_lineage  # noqa: E402

OUT = ROOT / "data" / "train" / "sources_v1.jsonl"
EVAL_FILES = [ROOT / "data" / "eval" / "jevbench_subset.jsonl", ROOT / "data" / "eval" / "custom_v0.jsonl"]
MASSIVE_PARQUET = "hf://datasets/AmazonScience/massive@refs%2Fconvert%2Fparquet/en-US/train/0000.parquet"


# --------------------------------------------------------------------------
# Question helpers
# --------------------------------------------------------------------------

def humanize(label: str) -> str:
    return label.replace("_", " ").strip()


def choice(instructions: str, options: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    return {"type": "noul", "instructions": instructions,
            "criteria": {"true": true or "Yes, the condition holds.", "false": false or "No, the condition does not hold."}}


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def pick_options(rng: random.Random, gold: str, pool: list[str], k_range: tuple[int, int]) -> list[str]:
    """Gold plus distractors from pool, shuffled; k is drawn from k_range and capped by the pool size."""
    k = min(rng.randint(*k_range), len(pool))
    distractors = rng.sample([p for p in pool if p != gold], k - 1)
    opts = distractors + [gold]
    rng.shuffle(opts)
    return opts


# --------------------------------------------------------------------------
# Source converters. Each yields (state, question, gold, source_row_index).
# --------------------------------------------------------------------------

def intents(rng, rows, names, field, label_field, family, oos_label=None) -> Iterator:
    pool = [n for n in names if n != oos_label]
    phrasings = ["A user message is supplied in the state. Which intent does it express?",
                 "What is the user trying to do in the message in the state?",
                 "Classify the request in the state into one of the listed intents."]
    for i, r in rows:
        gold_name = names[r[label_field]]
        state = {"message": r[field]}
        if rng.random() < 0.75:
            if gold_name == oos_label:
                opts = pick_options(rng, pool[0], pool, (4, 11))
                opts = [o for o in opts][: len(opts) - 1] + ["none_of_these"]
                rng.shuffle(opts)
                gold = "none_of_these"
            else:
                opts = pick_options(rng, gold_name, pool, (3, 12))
                if rng.random() < 0.3:
                    opts.append("none_of_these")
                gold = gold_name
            criteria = {o: ("None of the listed intents" if o == "none_of_these" else humanize(o)) for o in opts}
            yield state, choice(rng.choice(phrasings), criteria), gold, i
        else:
            target = gold_name if (rng.random() < 0.5 and gold_name != oos_label) else rng.choice(pool)
            q = noul(f"Is the user's intent in the state '{humanize(target)}'?",
                     f"The user's intent is {humanize(target)}.", "The user's intent is something else.")
            yield state, q, target == gold_name, i


def civil_comments(rng, rows) -> Iterator:
    levels = ["not toxic", "mildly toxic", "toxic", "severely toxic"]
    for i, r in rows:
        t = r["toxicity"]
        state = {"comment": r["text"]}
        if rng.random() < 0.6:
            q = noul("A user comment is supplied in the state. Is the comment toxic (rude, disrespectful or unreasonable "
                     "enough to make someone leave a discussion)?", "The comment is toxic.", "The comment is not toxic.")
            yield state, q, t >= 0.5, i
        else:
            level = 0 if t < 0.1 else 1 if t < 0.5 else 2 if t < 0.8 else 3
            yield state, score("Rate how toxic the comment in the state is.", levels), level, i


def nli(rng, rows, premise, hypothesis, label_of) -> Iterator:
    for i, r in rows:
        label = label_of(r)
        if label is None:
            continue
        state = {"premise": r[premise], "hypothesis": r[hypothesis]}
        roll = rng.random()
        if roll < 0.5:
            q = choice("Given the premise in the state, is the hypothesis entailed, contradicted, or neither?",
                       dict(rng.sample([("entailment", "The premise entails the hypothesis"),
                                        ("contradiction", "The premise contradicts the hypothesis"),
                                        ("neutral", "Neither; the premise does not settle it")], 3)))
            yield state, q, label, i
        elif roll < 0.8:
            q = noul("Does the premise in the state guarantee that the hypothesis is true?",
                     "The hypothesis must be true given the premise.", "The hypothesis is not guaranteed by the premise.")
            yield state, q, label == "entailment", i
        else:
            q = noul("Do the premise and hypothesis in the state contradict each other?",
                     "They contradict each other.", "They do not contradict each other.")
            yield state, q, label == "contradiction", i


def boolq(rng, rows) -> Iterator:
    for i, r in rows:
        question = r["question"].strip().rstrip("?") + "?"
        q = noul(f"Based only on the passage in the state: {question[0].upper() + question[1:]}", "Yes.", "No.")
        yield {"passage": r["passage"]}, q, bool(r["answer"]), i


def dbpedia(rng, rows, names) -> Iterator:
    pool = list(range(len(names)))
    for i, r in rows:
        opts = pick_options(rng, r["label"], pool, (4, 8))
        criteria = {humanize(names[o]).lower().replace(" ", "_"): humanize(names[o]) for o in opts}
        gold = humanize(names[r["label"]]).lower().replace(" ", "_")
        q = choice("An encyclopedia entry is supplied in the state. What kind of entity does it describe?", criteria)
        yield {"title": r["title"], "text": r["content"].strip()}, q, gold, i


def multiple_choice(rng, rows, question_of, options_of, gold_of, instructions) -> Iterator:
    for i, r in rows:
        options = options_of(r)
        gold = gold_of(r)
        if not options or gold not in options:
            continue
        items = list(options.items())
        rng.shuffle(items)
        yield {"question": question_of(r)}, choice(instructions, {k.lower(): v for k, v in items}), gold.lower(), i


def aqua_options(r) -> dict[str, str]:
    out = {}
    for o in r["options"]:
        m = re.match(r"\s*([A-E])\s*\)\s*(.*)", o)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


# --------------------------------------------------------------------------
# Registry: hub id -> (default cap, loader). Caps target ~150K rows at --scale 1.
# --------------------------------------------------------------------------

def load(hub_id: str, config: str | None = None, split: str = "train", **kw):
    from datasets import load_dataset
    return load_dataset(hub_id, config, split=split, **kw)


def sample_rows(rng: random.Random, ds, cap: int, filt: Callable | None = None) -> list[tuple[int, dict]]:
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        r = ds[i]
        if filt is None or filt(r):
            out.append((i, r))
            if len(out) >= cap:
                break
    return out


def build_clinc(rng, cap):
    ds = load("clinc/clinc_oos", "plus")
    names = ds.features["intent"].names
    return intents(rng, sample_rows(rng, ds, cap), names, "text", "intent", "clinc_intent", oos_label="oos")


def build_massive(rng, cap):
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=MASSIVE_PARQUET, split="train")
    return intents(rng, sample_rows(rng, ds, cap), ds.features["intent"].names, "utt", "intent", "massive_intent")


def build_civil(rng, cap):
    ds = load("google/civil_comments")
    # Balance: half the rows from comments rated toxic (>= 0.5), half from the rest.
    tox = sample_rows(rng, ds, cap // 2, lambda r: r["toxicity"] >= 0.5 and 20 <= len(r["text"]) <= 1500)
    non = sample_rows(rng, ds, cap - len(tox), lambda r: r["toxicity"] < 0.5 and 20 <= len(r["text"]) <= 1500)
    rows = tox + non
    rng.shuffle(rows)
    return civil_comments(rng, rows)


def build_snli(rng, cap):
    ds = load("stanfordnlp/snli")
    names = {0: "entailment", 1: "neutral", 2: "contradiction"}
    return nli(rng, sample_rows(rng, ds, cap, lambda r: r["label"] in names), "premise", "hypothesis", lambda r: names.get(r["label"]))


def build_wanli(rng, cap):
    ds = load("alisawuffles/WANLI")
    return nli(rng, sample_rows(rng, ds, cap), "premise", "hypothesis",
               lambda r: r["gold"] if r["gold"] in ("entailment", "neutral", "contradiction") else None)


def build_boolq(rng, cap):
    return boolq(rng, sample_rows(rng, load("google/boolq"), cap))


def build_dbpedia(rng, cap):
    ds = load("fancyzhx/dbpedia_14")
    return dbpedia(rng, sample_rows(rng, ds, cap), ds.features["label"].names)


def build_arc(rng, cap):
    rows = []
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        ds = load("allenai/ai2_arc", cfg)
        rows += [(f"{cfg}:{i}", r) for i, r in sample_rows(rng, ds, cap)]
    rng.shuffle(rows)
    return multiple_choice(rng, rows[:cap], lambda r: r["question"],
                           lambda r: dict(zip(r["choices"]["label"], r["choices"]["text"])), lambda r: r["answerKey"],
                           "A science question is supplied in the state. Which answer is correct?")


def build_csqa(rng, cap):
    return multiple_choice(rng, sample_rows(rng, load("tau/commonsense_qa"), cap), lambda r: r["question"],
                           lambda r: dict(zip(r["choices"]["label"], r["choices"]["text"])), lambda r: r["answerKey"],
                           "A commonsense question is supplied in the state. Which answer is most sensible?")


def build_aqua(rng, cap):
    return multiple_choice(rng, sample_rows(rng, load("deepmind/aqua_rat", "raw"), cap), lambda r: r["question"],
                           aqua_options, lambda r: r["correct"],
                           "A math word problem is supplied in the state. Which answer is correct?")


SOURCES: dict[str, tuple[str, int, Callable]] = {
    "clinc/clinc_oos": ("clinc_intent", 15000, build_clinc),
    "AmazonScience/massive": ("massive_intent", 11000, build_massive),
    "google/civil_comments": ("civil_toxicity", 30000, build_civil),
    "stanfordnlp/snli": ("snli_nli", 25000, build_snli),
    "alisawuffles/WANLI": ("wanli_nli", 15000, build_wanli),
    "google/boolq": ("boolq_passage_qa", 9400, build_boolq),
    "fancyzhx/dbpedia_14": ("dbpedia_topic", 15000, build_dbpedia),
    "allenai/ai2_arc": ("arc_science", 3400, build_arc),
    "tau/commonsense_qa": ("csqa_commonsense", 9700, build_csqa),
    "deepmind/aqua_rat": ("aqua_math", 10000, build_aqua),
}


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def lineage_guard() -> dict[str, dict]:
    entries = {e.get("hub_id"): e for e in parse_lineage(ROOT / "lineage.yaml")["datasets"] if e.get("hub_id")}
    for hub_id in SOURCES:
        e = entries.get(hub_id)
        if e is None:
            raise SystemExit(f"REFUSED: {hub_id} is not in lineage.yaml; add it and run scripts/check_lineage.py first")
        if e.get("use") != "train":
            raise SystemExit(f"REFUSED: {hub_id} is marked use: {e.get('use')} in lineage.yaml and may not be used for training")
    return entries


def text_fields(obj) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from text_fields(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from text_fields(v)


def norm_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:20]


def eval_hashes() -> set[str]:
    out = set()
    for path in EVAL_FILES:
        if not path.exists():
            raise SystemExit(f"{path} is missing; build the eval sets first so overlap can be checked")
        for line in path.open(encoding="utf-8"):
            case = json.loads(line)
            out |= {norm_hash(t) for t in text_fields(case.get("state")) if len(t) >= 40}
    return out


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every source cap (0.01 for a smoke build)")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--only", nargs="*", help="build only these Hub ids")
    args = ap.parse_args()

    lineage = lineage_guard()
    banned = eval_hashes()
    rows, overlaps = [], []
    for hub_id, (family, cap, build) in SOURCES.items():
        if args.only and hub_id not in args.only:
            continue
        rng = random.Random(f"{args.seed}:{hub_id}")
        n = max(1, int(cap * args.scale))
        made = 0
        for state, q, gold, src_idx in build(rng, n):
            if any(norm_hash(t) in banned for t in text_fields(state) if len(t) >= 40):
                overlaps.append((hub_id, src_idx))
                continue
            e = lineage[hub_id]
            rows.append({
                "row_id": f"{family}-{made}", "family": family, "task_type": q["type"], "state": state, "question": q,
                "gold": gold, "n_options": 2 if q["type"] == "noul" else len(q["criteria"]),
                "provenance": {"source": e["name"], "hub_id": hub_id, "license": e["license"], "origin": e["origin"],
                               "source_row": src_idx},
            })
            made += 1
        types = collections.Counter(r["task_type"] for r in rows if r["family"] == family)
        print(f"{family:18s} {made:6d} rows  {dict(types)}")

    if overlaps:
        print(f"ABORT: {len(overlaps)} rows share text with the eval sets, e.g. {overlaps[:3]}")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"total {len(rows)} rows -> {OUT}  (eval overlap check: 0 rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
