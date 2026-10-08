"""Render LINEAGE.md from lineage.yaml, so the human-readable table can never drift from the gated source.

    uv run --no-project python scripts/render_lineage.py          # rewrite LINEAGE.md
    uv run --no-project python scripts/render_lineage.py --check  # fail if LINEAGE.md is stale (CI)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "LINEAGE.md"

_spec = importlib.util.spec_from_file_location("check_lineage", ROOT / "scripts" / "check_lineage.py")
check_lineage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_lineage)

HEADER = """# Lineage

Every model and dataset used to build s1: what it is, who made it, its licence and how it was used. This file is
generated from [lineage.yaml](lineage.yaml) by `scripts/render_lineage.py`; edit the YAML, not this file.

## Provenance policy

Everything that feeds the released weights (`use: train` or `use: teacher`) passes
[scripts/check_lineage.py](scripts/check_lineage.py) in CI:

- **Approved model organisations only.** A base model, teacher or data generator must come from an organisation on
  the allowlist ({orgs}), and so must every ancestor in its `base_model` chain on the Hugging Face Hub, so a
  fine-tune or merge of another model cannot slip in under a clean name.
- **Permissive licences only.** Every training dataset, and every Super-NaturalInstructions task used, carries a
  licence that allows redistributing both the derived weights and the rows ({licences}). No non-commercial,
  unknown or custom-terms sources.
- **Eval sets never train.** `eval-only` entries are used to measure the model and for nothing else; the dev splits
  are deduplicated against the test sets and the training rows (`eval/jevbench/build_dev.py`).

A few Super-NaturalInstructions sources were also excluded by the provenance policy or for overlapping an eval set;
they are listed by name in [pipeline/sni_manifest.json](pipeline/sni_manifest.json).

`use` values: `train` (rows or weights in the released model), `teacher` (labels or generated data used for
training), `eval-only` (measurement only).
"""


def row(e: dict, cols: list[str]) -> str:
    cells = []
    for c in cols:
        v = e.get(c, "")
        if c == "name" and e.get("hub_id"):
            v = f"{v} ([`{e['hub_id']}`](https://huggingface.co/{'' if e.get('_kind') == 'model' else 'datasets/'}{e['hub_id']}))"
        cells.append(str(v).replace("|", "\\|"))
    return "| " + " | ".join(cells) + " |"


def render() -> str:
    lineage = check_lineage.parse_lineage(ROOT / "lineage.yaml")
    out = [HEADER.format(orgs=", ".join(f"`{o}`" for o in sorted(check_lineage.APPROVED_MODEL_ORGS)),
                         licences=", ".join(sorted(check_lineage.PERMISSIVE_LICENCES)))]
    groups = [("Models", "model", lineage["models"], ["name", "use", "role", "organisation", "license"]),
              ("Datasets", "dataset", lineage["datasets"], ["name", "use", "organisation", "license"])]
    for title, kind, entries, cols in groups:
        for use, heading in (("train", "used in training"), ("teacher", "used as teacher or generator"),
                             ("eval-only", "evaluation only")):
            chosen = [dict(e, _kind=kind) for e in entries if e.get("use") == use]
            if not chosen:
                continue
            out.append(f"## {title} {heading}\n")
            out.append("| " + " | ".join(c.capitalize() if c != "license" else "Licence" for c in cols) + " |")
            out.append("| " + " | ".join("---" for _ in cols) + " |")
            out += [row(e, cols) for e in chosen]
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if LINEAGE.md differs from what lineage.yaml renders")
    args = ap.parse_args()
    text = render()
    if args.check:
        if OUT.read_text(encoding="utf-8") != text:
            print("LINEAGE.md is stale: run scripts/render_lineage.py")
            return 1
        print("LINEAGE.md is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
