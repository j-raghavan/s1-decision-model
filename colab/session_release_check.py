"""Notebook cell: check the published model exactly as a user would get it, before the release goes public.

  1. download j-raghavan/s1-gemma4-26b-decision from the Hugging Face Hub (root layout, as the cards say)
  2. run examples/quickstart.py unchanged, and the Python snippet from MODEL_CARD.md unchanged
  3. on the 200 fixed JevBench canary cases (2-26 options), compare examples/quickstart.py's decide() with
     - the repository's batched scorer (s1/decoder.py option_log_probs) on the same loaded model: the top answer must
       match on at least 99% of cases (the two differ only in batching and padding)
     - the released model's saved bf16 test predictions through vLLM (results/raw/jevbench/g26-bos-vllm): the top
       answer must match on at least 95% (the merge check measured 0.970 between HF and vLLM)
  4. pack the report into /content/release_check_out.tgz; the Hub token is deleted from the VM at the end

Expects /content/s1_release_check.tgz and, while the model repo is private, /content/hf_token.

    colab exec -s <session> -f colab/session_release_check.py

C_* environment variables shrink everything for a local dry run (a small public model, CPU, a few cases).
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MODEL = os.environ.get("C_MODEL", "j-raghavan/s1-gemma4-26b-decision")
BASE = Path(os.environ.get("C_BASE", "/content"))
LIMIT = int(os.environ.get("C_LIMIT", 0)) or None
CUDA = os.environ.get("C_DEVICE", "cuda") == "cuda"
SAME_MIN, REF_MIN = 0.99, 0.95
W = BASE / "s1"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
REPORT = BASE / "release_check.json"
TOKEN = BASE / "hf_token"
report: dict = {}


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    report[key] = value
    REPORT.write_text(json.dumps(report, indent=1))


def run_script(args: list[str], log: str, env: dict) -> tuple[int, list[str]]:
    p = subprocess.run(args, cwd=W, stdout=open(BASE / log, "w"), stderr=subprocess.STDOUT, env=env, timeout=1800)
    lines = [ln for ln in (BASE / log).read_text().splitlines() if ln.strip() and "warn" not in ln.lower()]
    return p.returncode, lines[-6:]


def main() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_release_check.tgz -C {W}", shell=True, check=True)
    if TOKEN.exists():
        os.environ["HF_TOKEN"] = TOKEN.read_text().strip()
    if CUDA:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.18.0", "accelerate==1.15.0"],
                       check=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    t0 = time.time()
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)  # the whole repo, as from_pretrained would fetch it
    status(f"downloaded {MODEL} in {time.time() - t0:.0f}s")
    env = {**os.environ, "PYTHONPATH": str(W)}

    # 2a. the example script, unchanged except for the model id in a dry run
    script = (W / "examples" / "quickstart.py").read_text()
    if MODEL != "j-raghavan/s1-gemma4-26b-decision":  # dry run: small model, plain CPU load
        script = script.replace('"j-raghavan/s1-gemma4-26b-decision"', json.dumps(MODEL)).replace('device_map="auto"', "device_map=None")
    (W / "qs_run.py").write_text(script)
    rc, tail = run_script([sys.executable, "qs_run.py"], "quickstart.log", env)
    record("quickstart_script", {"exit": rc, "output": tail})
    status(f"examples/quickstart.py: exit {rc}; {' | '.join(tail[-3:])}")

    # 2b. the model card's Python block, unchanged
    card = (W / "MODEL_CARD.md").read_text()
    snippet = re.search(r"```python\n(.*?)```", card, re.S).group(1)
    if MODEL != "j-raghavan/s1-gemma4-26b-decision":
        snippet = snippet.replace('"j-raghavan/s1-gemma4-26b-decision"', json.dumps(MODEL)).replace('device_map="auto"', "device_map=None")
    (W / "card_snippet.py").write_text(snippet)
    rc, tail = run_script([sys.executable, "card_snippet.py"], "card_snippet.log", env)
    record("model_card_snippet", {"exit": rc, "output": tail})
    status(f"MODEL_CARD.md snippet: exit {rc}; {' | '.join(tail[-2:])}")

    # 3. agreement checks on the canary cases
    sys.path[:0] = [str(W), str(W / "eval" / "jevbench"), str(W / "examples")]
    import quickstart as qs
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    from paths import SUITES, read_jsonl
    from s1 import decoder
    from s1.decide import load_predictions

    ids = set((W / "data/eval/canary_ids.txt").read_text().split())
    cases = [c for c in read_jsonl(SUITES["jevbench"]["cases"]) if c["case_id"] in ids
             and 2 <= len(qs.options_for(c["question"])) <= 26][:LIMIT]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto" if CUDA else None).eval()
    t0 = time.time()
    mine = [qs.decide(model, tok, c["state"], c["question"]) for c in cases]
    secs = (time.time() - t0) / max(1, len(cases))
    readout = decoder.Readout(tok)
    device = "cuda" if CUDA else "cpu"
    with torch.no_grad():  # option_log_probs leaves gradients to its caller, as run_decoder.py does
        repo = [lp for i in range(0, len(cases), 8)
                for lp in decoder.option_log_probs(model, tok, readout, cases[i:i + 8], device)]
    top = lambda d: max(d, key=d.get)  # noqa: E731
    probs = lambda lp: {k: torch.tensor(v).exp().item() for k, v in lp.items()}  # noqa: E731
    repo = [probs(lp) for lp in repo]
    # disagreements: rescore one at a time with the repo scorer, to separate batching/padding numerics from a real
    # difference between the two code paths
    differ = [i for i in range(len(cases)) if top(mine[i]) != top(repo[i])]
    with torch.no_grad():
        single = {i: probs(decoder.option_log_probs(model, tok, readout, [cases[i]], device)[0]) for i in differ}
    ref = load_predictions(W / "results/raw/jevbench/g26-bos-vllm/predictions.jsonl")
    gold = lambda c: str(c.get("gold", "")).lower()  # noqa: E731
    rows = []
    for i, c in enumerate(cases):
        r = ref.get(c["case_id"])
        rows.append({"case_id": c["case_id"], "gold": c.get("gold"), "quickstart": mine[i], "repo_batch8": repo[i],
                     "repo_single": single.get(i), "saved_vllm_pred": r["pred"] if r else None})
    with (BASE / "per_case.jsonl").open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    n = max(1, len(cases))
    same_repo = sum(top(a) == top(b) for a, b in zip(mine, repo)) / n
    same_single = sum(top(mine[i]) == top(single[i]) for i in differ)  # of the batch-8 disagreements
    max_diff = max(abs(a[k] - b[k]) for a, b in zip(mine, repo) for k in a)
    with_ref = [(m, ref[c["case_id"]]) for m, c in zip(mine, cases) if c["case_id"] in ref]
    same_ref = sum(str(top(m)).lower() == str(r["pred"]).lower() for m, r in with_ref) / max(1, len(with_ref))
    acc_mine = sum(str(top(m)).lower() == gold(c) for m, c in zip(mine, cases)) / n
    acc_repo = sum(str(top(b)).lower() == gold(c) for b, c in zip(repo, cases)) / n
    # the quickstart path is one prompt at a time, so the like-for-like reference is the repo scorer at batch 1
    same_unbatched = (len(cases) - len(differ) + same_single) / n
    ok = same_unbatched >= SAME_MIN and (same_ref >= REF_MIN or MODEL != "j-raghavan/s1-gemma4-26b-decision")
    record("agreement", {"cases": len(cases), "same_top_as_repo_batch8": round(same_repo, 4),
                         "batch8_disagreements": len(differ), "of_which_match_repo_single": same_single,
                         "same_top_as_repo_unbatched": round(same_unbatched, 4),
                         "max_probability_difference_vs_batch8": round(max_diff, 5),
                         "cases_with_saved_vllm": len(with_ref), "same_top_as_saved_bf16_vllm": round(same_ref, 4),
                         "accuracy_quickstart": round(acc_mine, 4), "accuracy_repo_batch8": round(acc_repo, 4),
                         "seconds_per_decision_hf": round(secs, 3),
                         "required": {"repo_unbatched": SAME_MIN, "saved_vllm": REF_MIN}, "pass": ok})
    status(f"agreement on {len(cases)} cases: quickstart vs repo batch-8 {same_repo:.3f} ({len(differ)} differ, "
           f"{same_single} of them match the repo at batch 1 -> unbatched {same_unbatched:.3f}); vs saved bf16 vLLM "
           f"{same_ref:.3f} on {len(with_ref)}; accuracy quickstart {acc_mine:.3f} / repo {acc_repo:.3f} "
           f"-> {'PASS' if ok else 'FAIL'}")
    status("RELEASE CHECK DONE" if ok and report["quickstart_script"]["exit"] == 0
           and report["model_card_snippet"]["exit"] == 0 else "RELEASE CHECK FAILED")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
T0 = time.time()
try:
    main()
except Exception as exc:
    status(f"ABORTED: {exc!r}"[:300])
finally:
    TOKEN.unlink(missing_ok=True)
    status(f"token removed from VM: {not TOKEN.exists()}")
rc = subprocess.run(f"cd {BASE} && tar czf release_check_out.tgz release_check.json $(ls per_case.jsonl 2>/dev/null) $(ls *.log)", shell=True).returncode
status(f"packed release_check_out.tgz exit {rc}")
status(f"SESSION DONE, elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("C_HOLD_MIN", 10)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
