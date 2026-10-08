# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build the static project site (GitHub Pages) from the committed results files.

    uv run scripts/build_site.py                 # writes _site/
    uv run --no-project python scripts/build_site.py --out /tmp/site

Standard library only, so it runs without the project's training or API dependencies. Templates, CSS and
JS live in site/; every number on the leaderboard is computed here from results/<suite>/*.scores.json.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import statistics
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
SITE_SRC = ROOT / "site"

GITHUB = "https://github.com/j-raghavan/s1-decision-model"
HF_MODEL = "https://huggingface.co/j-raghavan/s1-gemma4-26b-decision"
HF_DATASET = "https://huggingface.co/datasets/j-raghavan/s1-decision-data"
MODEL_ID = "j-raghavan/s1-gemma4-26b-decision"
# Link or button to a hosted demo. Empty: the page shows "Demo coming soon".
# The playground calls the ZeroGPU Space (space/, scripts/deploy_space.py). Off until that Space is deployed, so the
# site never links to a demo that cannot answer.
LIVE_DEMO = False
DEMO_URL = "playground.html" if LIVE_DEMO else "https://ollama.com/jrlabs01/s1"
SPACE_ID = "j-raghavan/s1-decision-demo"  # the ZeroGPU Space that serves the playground (space/ in this repo)
GRADIO_CLIENT = "https://cdn.jsdelivr.net/npm/@gradio/client@2.7.1/dist/index.min.js"

# Public leaderboard: run id -> (display name, short column name, tag). Order is the default display order;
# the tables sort by mean accuracy. A run is shown only if its scores file exists for that suite.
SYSTEMS: dict[str, tuple[str, str, str]] = {
    "g26-bos-vllm": ("s1 (Gemma 4 26B-A4B + s1 fine-tune), bf16", "s1 bf16", "s1"),
    "s1-int4-ollama": ("s1, int4 via Ollama", "s1 int4", "s1"),
    "jev-1.13.0": ("Jev 1.13 (TypeSafe AI, closed)", "Jev 1.13", "closed"),
    "base-bos-vllm": ("Gemma 4 26B-A4B, untuned (with <bos>)", "Gemma 4 26B-A4B", "untuned"),
    "gemma4-31b": ("Gemma 4 31B, untuned", "Gemma 4 31B", "untuned"),
    "gemma4-12b": ("Gemma 4 12B, untuned", "Gemma 4 12B", "untuned"),
    "gptoss-120b": ("gpt-oss-120b, untuned", "gpt-oss-120b", "untuned"),
    "laya-421m-en": ("Laya 421M", "Laya 421M", ""),
    "von-1.3": ("Von 1.3", "Von 1.3", ""),
    "s1-v2": ("s1 encoder (ModernBERT-large 396M, earlier approach)", "s1 encoder", "earlier"),
}
S1_RUN = "g26-bos-vllm"
JEV_RUN = "jev-1.13.0"
BASE_RUN = "base-bos-vllm"
TAG_LABELS = {"s1": "s1", "closed": "closed", "untuned": "untuned", "earlier": "earlier s1", "": ""}

# Calibrated JevBench means (per-task calibrators fitted on the dev splits only, eval/jevbench/calibrate_from_dev.py),
# as stated in the README status table. Kept as constants because the README is prose, not data.
CALIBRATED_README = {"s1": 0.817, "jev": 0.856, "s1_custom": 0.981}

NOT_SCORE = "score"  # task type whose headline is MAE (lower is better); excluded from accuracy means


# ---------------------------------------------------------------------------------------------------------------
# Aggregation. Mirrors eval/jevbench/score.py, which cannot be imported here (it loads the upstream harness at
# import time): per slice, the headline value is accuracy for choice and yes/no slices and MAE for score slices;
# the mean accuracy is the plain mean over the non-score slices, the mean ECE the mean of ece_10_bins over the
# slices that report it, and either mean is n/a unless every accuracy slice of the suite is present.
# ---------------------------------------------------------------------------------------------------------------

def load_scores(results: Path, suite: str, run_id: str) -> dict | None:
    path = results / suite / f"{run_id}.scores.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def slice_cell(entry: dict | None) -> dict | None:
    """Headline, ECE, coverage and CI for one slice, or None if the slice was not scored."""
    if not entry or "headline" not in entry:
        return None
    h = entry["headline"]
    return {
        "type": entry["task_type"],
        "value": h["value"],
        "metric": h.get("metric", ""),
        "ci": h.get("ci95_bootstrap_by_family"),
        "ece": entry.get("metrics", {}).get("ece_10_bins"),
        "coverage": entry.get("coverage", 1.0),
    }


def summarize(payload: dict, acc_slices: list[str]) -> dict:
    cells = {k: slice_cell(e) for k, e in payload["slices"].items()}
    acc = [cells[s]["value"] for s in acc_slices if cells.get(s)]
    ece = [cells[s]["ece"] for s in acc_slices if cells.get(s) and cells[s]["ece"] is not None]
    n = len(acc_slices)
    return {
        "cells": cells,
        "mean_acc": statistics.fmean(acc) if acc and len(acc) == n else None,
        "mean_ece": statistics.fmean(ece) if ece and len(ece) == n else None,
        "n_acc": n,
        "partial": any((cells.get(s) or {}).get("coverage", 0) < 1 for s in acc_slices),
        "scored_at": payload.get("scored_at", ""),
    }


def suite_table(results: Path, suite: str) -> dict:
    """Every displayed system that has a scores file for the suite, with its summary."""
    payloads = {r: p for r in SYSTEMS if (p := load_scores(results, suite, r)) is not None}
    types: dict[str, str] = {}
    for p in payloads.values():
        for k, e in p["slices"].items():
            types.setdefault(k, e["task_type"])
    slices = sorted(types)
    acc_slices = [s for s in slices if types[s] != NOT_SCORE]
    rows = {r: summarize(p, acc_slices) for r, p in payloads.items()}
    return {"suite": suite, "slices": slices, "types": types, "acc_slices": acc_slices, "rows": rows}


# ---------------------------------------------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------------------------------------------

esc = html.escape


def f3(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def gh(path: str) -> str:
    return f"{GITHUB}/blob/main/{path}"


def tag_html(tag: str) -> str:
    return f'<span class="tag tag-{tag}">{esc(TAG_LABELS[tag])}</span>' if tag else ""


def leaderboard_table(t: dict, table_id: str, caption: str) -> str:
    rows = t["rows"]
    score_slices = [s for s in t["slices"] if t["types"][s] == NOT_SCORE]
    best = max((r["mean_acc"] for r in rows.values() if r["mean_acc"] is not None), default=1.0)
    head = ['<th scope="col" data-sort="text">System</th>',
            f'<th scope="col" data-sort="num" class="num" aria-sort="descending">Mean accuracy<br><small>{len(t["acc_slices"])} slices, higher is better</small></th>',
            '<th scope="col" data-sort="num" class="num">Mean ECE<br><small>lower is better</small></th>']
    head += [f'<th scope="col" data-sort="num" class="num">{esc(s)} MAE<br><small>lower is better</small></th>'
             for s in score_slices]
    order = sorted(rows, key=lambda r: -(rows[r]["mean_acc"] or -1))
    body = []
    for r in order:
        name, _, tag = SYSTEMS[r]
        s = rows[r]
        bar = 0 if s["mean_acc"] is None else round(100 * s["mean_acc"] / best, 1)
        cls = ' class="hl"' if tag == "s1" else ""
        cells = [f'<th scope="row" data-v="{esc(name)}"><span class="sys">{esc(name)}</span> {tag_html(tag)}</th>',
                 f'<td class="num" data-v="{s["mean_acc"] if s["mean_acc"] is not None else -1}">'
                 f'<span class="bar" style="--w:{bar}%"></span><span class="val">{f3(s["mean_acc"])}</span></td>',
                 f'<td class="num" data-v="{s["mean_ece"] if s["mean_ece"] is not None else 99}">{f3(s["mean_ece"])}</td>']
        for sl in score_slices:
            c = s["cells"].get(sl)
            cells.append(f'<td class="num" data-v="{c["value"] if c else 99}">{f3(c["value"] if c else None)}</td>')
        body.append(f'<tr data-run="{esc(r)}"{cls}>' + "".join(cells) + "</tr>")
    return (f'<div class="table-wrap" role="region" aria-label="{esc(caption)}" tabindex="0">'
            f'<table class="sortable lb" id="{table_id}"><caption>{esc(caption)}</caption>'
            f'<thead><tr>{"".join(head)}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def slice_matrix(t: dict, caption: str) -> str:
    rows = t["rows"]
    order = sorted(rows, key=lambda r: -(rows[r]["mean_acc"] or -1))
    head = '<th scope="col">Slice</th><th scope="col">Type</th>' + "".join(
        f'<th scope="col" class="num">{esc(SYSTEMS[r][1])}</th>' for r in order)
    body = []
    for sl in t["slices"]:
        typ = t["types"][sl]
        tds = []
        for r in order:
            c = rows[r]["cells"].get(sl)
            if c is None:
                tds.append('<td class="num muted">n/a</td>')
                continue
            title = f' title="95% CI {c["ci"][0]:.3f} to {c["ci"][1]:.3f} (bootstrap by family)"' if c.get("ci") else ""
            txt = f3(c["value"])
            if c["ece"] is not None:
                txt += f' <span class="ece">/ {c["ece"]:.3f}</span>'
            if c["coverage"] < 1:
                txt += f' <span class="cov">({c["coverage"] * 100:.0f}% cov)</span>'
            tds.append(f'<td class="num"{title}>{txt}</td>')
        label = "MAE" if typ == NOT_SCORE else ("yes/no" if typ == "noul" else typ)
        body.append(f'<tr><th scope="row"><code>{esc(sl)}</code></th><td>{esc(label)}</td>{"".join(tds)}</tr>')
    return (f'<div class="table-wrap" role="region" aria-label="{esc(caption)}" tabindex="0">'
            f'<table class="matrix"><caption>{esc(caption)}</caption><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def code_block(code: str, lang: str) -> str:
    return f'<pre class="code" data-lang="{lang}"><code>{esc(code.strip())}</code></pre>'


PY_SNIPPET = f'''
import json, torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

REPO = "{MODEL_ID}"
tok = AutoTokenizer.from_pretrained(REPO, subfolder="bf16")
model = Gemma4ForConditionalGeneration.from_pretrained(
    REPO, subfolder="bf16", dtype=torch.bfloat16, device_map="auto")

state = {{"ticket": "I was charged twice for order 4471."}}
options = {{"billing": "Payments and refunds", "shipping": "Deliveries", "tech": "Bugs"}}
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(options)]
lines = "\\n".join(f"{{L}}) {{k}}: {{d}}" for L, (k, d) in zip(letters, options.items()))
user = ("You are a decision model. Read the state and answer the question by choosing one option.\\n\\n"
        f"STATE:\\n{{json.dumps(state, ensure_ascii=False, indent=1)}}\\n\\n"
        "QUESTION: Which team should handle this?\\n\\n"
        f"OPTIONS:\\n{{lines}}\\n\\nAnswer with the option letter only.")
# Gemma 4's tokenizer does not add <bos> itself: prepend it, then prefill "Answer:".
prompt = (tok.bos_token + "<|turn>user\\n" + user + "<turn|>\\n"
          "<|turn>model\\n<|channel>thought\\n<channel|>Answer:")

ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
with torch.no_grad():
    logits = model(**ids).logits[0, -1]          # one forward pass, no generation
letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in letters]
probs = torch.softmax(logits[letter_ids].float(), -1)
print(dict(zip(options, probs.tolist())))         # raw (uncalibrated) probabilities
'''

OLLAMA_SNIPPET = r'''
ollama pull jrlabs01/s1        # the int4 build, about 17 GB
# Send raw prompts that start with <bos>: Ollama's chat formatting changes the prompt and the answers.
curl -s localhost:11434/api/generate -d '{
  "model": "jrlabs01/s1", "raw": true, "stream": false,
  "logprobs": true, "top_logprobs": 20,
  "options": {"temperature": 0, "num_predict": 1},
  "prompt": "<bos><|turn>user\nYou are a decision model. Read the state and answer the question by choosing one option.\n\nSTATE:\n{\n \"ticket\": \"I was charged twice for order 4471.\"\n}\n\nQUESTION: Which team should handle this?\n\nOPTIONS:\nA) billing: Payments and refunds\nB) shipping: Deliveries\nC) tech: Bugs\n\nAnswer with the option letter only.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Answer:"
}'
# Normalise the top_logprobs of the single token over "A", "B", "C".
'''

API_SNIPPET = '''
uv run --extra api uvicorn api.server:app --port 8000
curl -s localhost:8000/v1/decisions -H 'content-type: application/json' -d '{
  "state": {"ticket": "I was charged twice for order 4471."},
  "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
                         "criteria": {"billing": "Payments and refunds", "shipping": "Deliveries", "tech": "Bugs"}}}}'
'''


# ---------------------------------------------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------------------------------------------

def render(name: str, **values: str) -> str:
    return Template((SITE_SRC / "templates" / name).read_text(encoding="utf-8")).substitute(**values)


def page(title: str, description: str, active: str, content: str) -> str:
    nav = {"index": "", "leaderboard": "", "playground": ""}
    nav[active] = ' aria-current="page"'
    return render("layout.html", title=esc(title), description=esc(description), content=content,
                  nav_index=nav["index"], nav_leaderboard=nav["leaderboard"],
                  nav_playground=(f'<a href="playground.html"{nav["playground"]}>Playground</a>' if LIVE_DEMO else ""),
                  github=GITHUB, hf_model=HF_MODEL,
                  methodology=gh("docs/methodology.md"), license=gh("LICENSE"))


def card(value: str, label: str, detail: str) -> str:
    return (f'<div class="card"><div class="card-value">{value}</div><div class="card-label">{label}</div>'
            f'<div class="card-detail">{detail}</div></div>')


def build_index(jev: dict, custom: dict) -> str:
    j, c = jev["rows"], custom["rows"]
    s1, ref, base = j[S1_RUN], j[JEV_RUN], j[BASE_RUN]
    cards = [
        card(f3(s1["mean_acc"]), "JevBench mean accuracy, raw",
             f'Jev 1.13: {f3(ref["mean_acc"])}. Untuned base with &lt;bos&gt;: {f3(base["mean_acc"])}. '
             f'{s1["n_acc"]} slices, 1,700 held-out cases.'),
        card(f'{CALIBRATED_README["s1"]:.3f}', "JevBench, calibrated",
             f'Jev 1.13: {CALIBRATED_README["jev"]:.3f}. Calibrators fitted on dev splits only.'),
        card(f3(s1["mean_ece"]), "Mean calibration error (ECE), raw",
             f'Jev 1.13: {f3(ref["mean_ece"])}. Untuned base: {f3(base["mean_ece"])}. 10 bins, lower is better.'),
    ]
    if S1_RUN in c:
        detail = (f'Untuned base: {f3(c[BASE_RUN]["mean_acc"])}. ' if BASE_RUN in c else "")
        cards.append(card(f3(c[S1_RUN]["mean_acc"]), "Structured-decision set, raw",
                          detail + f'Calibrated: {CALIBRATED_README["s1_custom"]:.3f}. JSON-state and slide-layout '
                          'rules; Jev has no predictions on this set.'))
    if DEMO_URL:
        demo = (f'<a class="btn btn-primary" href="{esc(DEMO_URL)}">Try it in the playground</a>' if LIVE_DEMO else
                f'<a class="btn btn-primary" href="{esc(DEMO_URL)}" rel="noopener">Run it locally with Ollama</a>')
    else:
        demo = '<span class="btn btn-disabled" aria-disabled="true">Demo coming soon</span>'
    return page(
        "s1: a calibrated System One decision model",
        "s1 is an open-weights decision model: typed answers with calibrated probabilities in one forward pass.",
        "index",
        render("index.html", cards="".join(cards), demo=demo, github=GITHUB, hf_model=HF_MODEL,
               hf_dataset=HF_DATASET, license=gh("LICENSE"),
               py_snippet=code_block(PY_SNIPPET, "python"), ollama_snippet=code_block(OLLAMA_SNIPPET, "bash"),
               api_snippet=code_block(API_SNIPPET, "bash"),
               jev_raw=f3(ref["mean_acc"]), s1_raw=f3(s1["mean_acc"]),
               s1_sst5=f3((s1["cells"].get("sst5") or {}).get("value")),
               jev_sst5=f3((ref["cells"].get("sst5") or {}).get("value")),
               methodology=gh("docs/methodology.md"), data_card=gh("DATA_CARD.md"), model_card=gh("MODEL_CARD.md"),
               lineage=gh("LINEAGE.md"), api_src=gh("api/server.py")))


def build_leaderboard(jev: dict, custom: dict) -> str:
    int4_note = "" if "s1-int4-ollama" in jev["rows"] else (
        '<p class="note">The int4 Ollama build of s1 will be added here once its JevBench scores are published.</p>')
    return page(
        "Leaderboard: s1 decision model",
        "JevBench and structured-decision results for s1, Jev 1.13 and untuned open models.",
        "leaderboard",
        render("leaderboard.html",
               jev_table=leaderboard_table(jev, "lb-jevbench", "JevBench test subset, 1,700 cases"),
               jev_matrix=slice_matrix(jev, "JevBench per-slice results: headline / ECE"),
               custom_table=leaderboard_table(custom, "lb-custom", "Structured-decision test set"),
               custom_matrix=slice_matrix(custom, "Structured-decision per-slice results: headline / ECE"),
               n_jev_acc=str(jev["rows"][S1_RUN]["n_acc"]) if S1_RUN in jev["rows"] else "?",
               n_custom_acc=str(len(custom["acc_slices"])),
               n_custom_score=str(len(custom["slices"]) - len(custom["acc_slices"])),
               int4_note=int4_note,
               s1_cal=f'{CALIBRATED_README["s1"]:.3f}', jev_cal=f'{CALIBRATED_README["jev"]:.3f}',
               methodology=gh("docs/methodology.md"), data_card=gh("DATA_CARD.md"), model_card=gh("MODEL_CARD.md"),
               lineage=gh("LINEAGE.md"),
               jev_summary=gh("results/jevbench/summary.md"), custom_summary=gh("results/custom/summary.md"),
               scorer=gh("eval/jevbench/score.py"), cal_script=gh("eval/jevbench/calibrate_from_dev.py"),
               subset_script=gh("eval/jevbench/build_subset.py"), dev_script=gh("eval/jevbench/build_dev.py"),
               custom_gen=gh("eval/custom/generate.py")))


def check_readout(readme: Path) -> None:
    """Warn (not fail) when the README no longer states the calibrated figures this site repeats."""
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    missing = [f"{v:.3f}" for v in CALIBRATED_README.values() if f"{v:.3f}" not in text]
    if missing:
        print(f"warning: README.md no longer mentions {', '.join(missing)}; update CALIBRATED_README", file=sys.stderr)


def build_playground() -> str:
    return page("Playground: s1 decision model",
                "Try s1: give it a state and typed questions, get calibrated probabilities from the released model.",
                "playground",
                render("playground.html", space_id=SPACE_ID, space_url=f"https://huggingface.co/spaces/{SPACE_ID}",
                       client_url=GRADIO_CLIENT, hf_model=HF_MODEL))


def build(out: Path, results: Path = ROOT / "results") -> dict:
    jev, custom = suite_table(results, "jevbench"), suite_table(results, "custom")
    for need in (S1_RUN, JEV_RUN, BASE_RUN):
        if need not in jev["rows"]:
            raise SystemExit(f"missing results/jevbench/{need}.scores.json")
    out = out.resolve()
    if out == ROOT or out in ROOT.parents or (out / ".git").exists():
        raise SystemExit(f"refusing to replace {out}: choose an output directory of its own")
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    for f in (SITE_SRC / "assets").iterdir():
        shutil.copy2(f, out / "assets" / f.name)
    (out / "index.html").write_text(build_index(jev, custom), encoding="utf-8")
    (out / "leaderboard.html").write_text(build_leaderboard(jev, custom), encoding="utf-8")
    if LIVE_DEMO:
        (out / "playground.html").write_text(build_playground(), encoding="utf-8")
        shutil.copy2(ROOT / "space" / "examples.json", out / "assets" / "examples.json")  # one source for both UIs
    else:
        (out / "assets" / "playground.js").unlink(missing_ok=True)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    check_readout(ROOT / "README.md")
    return {"jevbench": jev, "custom": custom}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    args = ap.parse_args()
    tables = build(args.out)
    for suite, t in tables.items():
        print(f"{suite}: {len(t['acc_slices'])} accuracy slices")
        for r, s in sorted(t["rows"].items(), key=lambda kv: -(kv[1]["mean_acc"] or -1)):
            print(f"  {r:16s} acc {f3(s['mean_acc'])}  ece {f3(s['mean_ece'])}  {SYSTEMS[r][0]}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
