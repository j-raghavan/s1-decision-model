"""Make long-input training rows by padding states with irrelevant context, so gold labels stay valid.

The model has to find the decisive fields in a large state and ignore the rest; that is the skill long
inputs (agent traces, log dumps, multi-document states) need, and v1/v2 never saw an input over ~1K tokens.

Each chosen row gets one extra state field, `unrelated_context`, holding deterministic filler: log lines
from other services, tickets for other customers, notes on other cases. It never mentions the row's own
entities or fields, so it cannot change the answer, and the row keeps its target distribution. Generated
with templates and a seeded RNG; no model involved.

    uv run pipeline/long_context.py --targets data/train/targets_v3.jsonl --n 4000 \\
        --out data/train/targets_v3_long.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SERVICES = ["billing-worker", "search-indexer", "auth-gateway", "media-transcoder", "notification-fanout", "geo-cache",
            "report-builder", "ledger-sync", "feature-store", "image-resizer"]
LEVELS = ["INFO", "INFO", "INFO", "DEBUG", "WARN"]
VERBS = ["completed", "scheduled", "retried", "skipped", "refreshed", "rotated", "archived", "queued"]
OBJECTS = ["batch", "snapshot", "index shard", "cache entry", "export", "webhook delivery", "nightly job", "thumbnail set"]
TOPICS = ["office move logistics", "quarterly offsite agenda", "printer replacement", "parking permits", "cafeteria menu change",
          "badge photo retakes", "conference room booking etiquette", "holiday schedule", "plant watering rota", "wifi password rotation"]
NAMES = ["Ilse Varga", "Tobias Renn", "Mireille Okafor", "Dev Patel-Quinn", "Hana Lindqvist", "Oscar Mbeki", "Ruth Calloway", "Akira Sato"]


def filler(rng: random.Random, target_chars: int) -> dict:
    logs, tickets, notes = [], [], []
    size = 0
    while size < target_chars:
        kind = rng.random()
        if kind < 0.5:
            line = (f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z "
                    f"{rng.choice(LEVELS)} {rng.choice(SERVICES)} {rng.choice(OBJECTS)} {rng.choice(VERBS)} "
                    f"id={rng.getrandbits(32):08x} dur_ms={rng.randint(3, 900)}")
            logs.append(line)
        elif kind < 0.8:
            t = {"ticket": f"OPS-{rng.randint(10000, 99999)}", "requester": rng.choice(NAMES), "topic": rng.choice(TOPICS),
                 "status": rng.choice(["open", "waiting", "closed"]), "note": f"Follow up about {rng.choice(TOPICS)} next week."}
            tickets.append(t)
            line = json.dumps(t)
        else:
            line = (f"Meeting note ({rng.choice(NAMES)}): discussed {rng.choice(TOPICS)} and {rng.choice(TOPICS)}; "
                    f"no decisions; revisit in {rng.randint(2, 6)} weeks.")
            notes.append(line)
        size += len(line) + 4
    return {"note": "Unrelated records from other teams, included for context only.", "other_service_logs": logs,
            "other_tickets": tickets, "other_meeting_notes": notes}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--min-chars", type=int, default=7000, help="~2K tokens")
    ap.add_argument("--max-chars", type=int, default=14000,
                    help="~4K tokens: v3 trains long rows up to ~4K so a 40 GB GPU fits them; 4K-8K is measured, not trained")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in args.targets.open(encoding="utf-8")]
    # Only rows whose state is a JSON object (structured decisions) and not held out for validation.
    pool = [r for r in rows if isinstance(r["state"], dict) and not r.get("holdout_family") and "~" not in r["row_id"]]
    picked = rng.sample(pool, min(args.n, len(pool)))
    out = []
    for r in picked:
        state = dict(r["state"])
        state["unrelated_context"] = filler(rng, rng.randint(args.min_chars, args.max_chars))
        out.append(r | {"row_id": r["row_id"] + "~long", "state": state, "long_context": True})
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    lens = sorted(len(json.dumps(r["state"])) for r in out)
    print(f"{len(out):,} long rows -> {args.out}; state chars p10 {lens[len(lens)//10]:,}, median {lens[len(lens)//2]:,}, "
          f"max {lens[-1]:,}; families {len({r['family'] for r in out})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
