"""Provenance and licence gate for everything that feeds the shipped model.

Every model and dataset used to build s1 is listed in lineage.yaml. Entries with `use: train` or `use: teacher`
feed the released weights and must pass two allowlists:

1. Models: the Hugging Face organisation of the model, and of every ancestor in its `base_model` chain on the Hub,
   must be on APPROVED_MODEL_ORGS. Walking the chain catches a fine-tune or merge of an unapproved model even when
   its own name looks clean. Adding an organisation is a reviewed change to this file.
2. Licences: every training dataset, and every Super-NaturalInstructions task in pipeline/sni_manifest.json, must
   carry a licence on PERMISSIVE_LICENCES (no non-commercial, no unknown, no custom terms), so the weights and the
   training data can both be redistributed.

Entries marked `use: eval-only` never feed training, labeling or a shipped artifact; they are listed but not gated.

    uv run scripts/check_lineage.py            # needs network for the Hub walk
    uv run scripts/check_lineage.py --offline  # allowlist and licence checks only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Hugging Face organisations whose models may be used as a base, teacher or data generator (lowercase).
APPROVED_MODEL_ORGS = {"google", "openai", "answerdotai"}

# Licences that allow redistributing both derived weights and the training rows (with attribution where required).
# Spellings are normalised by norm_licence; ALIASES maps the variants used on dataset cards and in the SNI manifest.
PERMISSIVE_LICENCES = {
    "apache-2.0", "mit", "bsd", "cc0-1.0", "cc-by-3.0", "cc-by-4.0", "cc-by-sa", "cc-by-sa-3.0", "cc-by-sa-4.0",
    "oanc",  # Open American National Corpus: distributed without restriction
}
ALIASES = {"cc0:-public-domain": "cc0-1.0", "cc0": "cc0-1.0"}

HUB_API = "https://huggingface.co/api/{kind}/{hub_id}"
GATED_USES = {"train", "teacher"}


def parse_lineage(path: Path) -> dict[str, list[dict]]:
    """Minimal YAML reader for lineage.yaml's two flat lists of mappings."""
    out: dict[str, list[dict]] = {"models": [], "datasets": []}
    section, current = None, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #")[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section = line[:-1]
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            current = {}
            out.setdefault(section, []).append(current)
            stripped = stripped[2:]
        if current is not None and ":" in stripped:
            key, _, value = stripped.partition(":")
            current[key.strip()] = value.strip().strip('"')
    return out


def norm_licence(text: str) -> str:
    key = text.strip().lower().replace(" ", "-").replace("_", "-")
    return ALIASES.get(key, key)


def permissive(text: str) -> bool:
    return norm_licence(text) in PERMISSIVE_LICENCES


def org_of(hub_id: str) -> str | None:
    return hub_id.split("/")[0].lower() if "/" in hub_id else None


def hub_parents(hub_id: str, kind: str, attempts: int = 4) -> list[str]:
    """base_model entries from the Hub card. Retries transient network errors with backoff; still fails closed
    (the caller reports a violation) if the Hub stays unreachable, since an unverified lineage must not pass."""
    url = HUB_API.format(kind=kind, hub_id=hub_id)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                card = json.load(resp).get("cardData") or {}
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt * 2)
    base = card.get("base_model") or []
    return [base] if isinstance(base, str) else list(base)


def walk_ancestors(hub_id: str, depth: int = 0, seen: set[str] | None = None) -> list[str]:
    seen = seen if seen is not None else set()
    if depth > 8 or hub_id in seen:
        return []
    seen.add(hub_id)
    chain = []
    for parent in hub_parents(hub_id, "models"):
        chain.append(parent)
        chain += walk_ancestors(parent, depth + 1, seen)
    return chain


def model_problems(e: dict, offline: bool) -> list[str]:
    hub_id = e.get("hub_id", "")
    if not hub_id:
        return ["no hub_id: a model that feeds the release must be traceable on the Hub"]
    problems = []
    if org_of(hub_id) not in APPROVED_MODEL_ORGS:
        problems.append(f"organisation of {hub_id} is not on the approved list")
    if not offline:
        try:
            for ancestor in walk_ancestors(hub_id):
                if org_of(ancestor) not in APPROVED_MODEL_ORGS:
                    problems.append(f"ancestor {ancestor} is not from an approved organisation")
        except (urllib.error.URLError, TimeoutError) as exc:
            problems.append(f"could not walk base_model chain ({exc}); rerun online")
    if not permissive(e.get("license", "")):
        problems.append(f"licence '{e.get('license')}' is not on the permissive list")
    return problems


def sni_problems(manifest: Path = ROOT / "pipeline" / "sni_manifest.json") -> list[str]:
    tasks = json.loads(manifest.read_text(encoding="utf-8"))["tasks"]
    return [f"task {t['file']}: licence {t['license']} is not on the permissive list"
            for t in tasks if not t["license"] or not all(permissive(x) for x in t["license"])]


def dataset_problems(e: dict) -> list[str]:
    if e.get("name", "").startswith("Super-NaturalInstructions"):
        return sni_problems()
    if not permissive(e.get("license", "")):
        return [f"licence '{e.get('license')}' is not on the permissive list"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="skip the Hub base_model walk")
    ap.add_argument("--file", default=str(ROOT / "lineage.yaml"))
    args = ap.parse_args()

    lineage = parse_lineage(Path(args.file))
    failures = []
    for kind, entries in (("model", lineage["models"]), ("dataset", lineage["datasets"])):
        for e in entries:
            label = e.get("name", "?")
            if e.get("use") not in GATED_USES:
                print(f"--    {kind:7s} {label} ({e.get('use')}; not gated)")
                continue
            problems = model_problems(e, args.offline) if kind == "model" else dataset_problems(e)
            failures += [f"{kind} {label}: {p}" for p in problems]
            if not problems:
                print(f"ok    {kind:7s} {label}")

    for f in failures:
        print(f"FAIL  {f}")
    if failures:
        print(f"\n{len(failures)} provenance or licence violation(s). Remove the component, or mark it eval-only.")
        return 1
    print("\nprovenance and licence check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
