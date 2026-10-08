"""Notebook cell: merge the <bos> fine-tune (step 1500), verify the merge, upload bf16 weights to the Hugging Face repo (private until release).

The VM disk (113 GB, about 43 GB used by the system) cannot hold base + merged copies of the 52 GB model, and the
base ships as one 49.9 GB file. So:

  1. check free RAM and disk; download the base weights
  2. merge in RAM (scripts/merge_lora.py --consume-base): each base file is loaded, deleted from disk, and written
     back merged as 5 GB pieces, so peak disk is about the base size
  3. verify: the merged model through HF (with <bos>) on 200 fixed JevBench cases must agree >= 93% with the
     fine-tuned adapter run scored through vLLM (results/raw/jevbench/g26-bos-vllm); the untuned model agrees 87%
  4. upload the merged weights to the Hugging Face repo, private until release (only if verified), then delete the token from the VM

The int4 Ollama package is built afterwards from the uploaded weights on a machine with the disk for it.

Expects /content/s1_pkg2.tgz, /content/adapter_final.tgz (adapter_final/) and /content/hf_token.

    colab exec -s <session> -f colab/session_package2.py

Q_* environment variables shrink everything for a local dry run.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL = os.environ.get("Q_MODEL", "google/gemma-4-26B-A4B-it")
REPO = os.environ.get("Q_REPO", "j-raghavan/s1-gemma4-26b-decision")
BASE = Path(os.environ.get("Q_BASE", "/content"))
DEVICE_ARGS = os.environ.get("Q_DEVICE_ARGS", "")
AGREE_MIN = float(os.environ.get("Q_AGREE_MIN", 0.93))
MIN_CASES = int(os.environ.get("Q_MIN_CASES", 196))  # of the 200 canary cases
CONSUME = os.environ.get("Q_CONSUME", "1") == "1"
PIECE_GB = os.environ.get("Q_PIECE_GB", "5")
W = BASE / "s1"
MERGED = BASE / "merged"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
REPORT = BASE / "package.json"
TOKEN = BASE / "hf_token"
results: dict = {}


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    results[key] = value
    REPORT.write_text(json.dumps(results, indent=1))


def run(args: list[str], log: str, timeout_min: float) -> int:
    try:
        return subprocess.run(args, cwd=W, stdout=open(BASE / log, "w"), stderr=subprocess.STDOUT,
                              timeout=timeout_min * 60).returncode
    except subprocess.TimeoutExpired:
        return -9


def resources() -> dict:
    try:
        ram = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError, AttributeError):  # not available on macOS
        ram = None
    return {"free_ram_gb": round(ram, 1) if ram else None, "free_disk_gb": round(shutil.disk_usage(BASE).free / 1e9, 1)}


def main() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_pkg2.tgz -C {W} && tar xzf {BASE}/adapter_final.tgz -C {BASE}",
                   shell=True, check=True)
    status(f"start: {resources()}")
    if not DEVICE_ARGS:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.18.0", "peft==0.21.2",
                        "accelerate==1.15.0", "safetensors"], check=True)
        # Colab preinstalls torchao 0.10, which PEFT rejects; nothing here uses it.
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

    args = [sys.executable, "scripts/merge_lora.py", MODEL, str(BASE / "adapter_final"), str(MERGED),
            "--max-piece-gb", PIECE_GB] + (["--consume-base"] if CONSUME else [])
    t0 = time.time()
    rc = run(args, "merge.log", 60)
    tail = (BASE / "merge.log").read_text().strip().splitlines()[-1:]
    record("merge", {"exit": rc, "summary": tail, "minutes": round((time.time() - t0) / 60, 1), "after": resources()})
    status(f"merge: exit {rc} in {(time.time() - t0) / 60:.1f} min; {tail}; {resources()}")
    if rc != 0:
        status("STOPPED: merge failed; nothing uploaded")
        return

    rc = run([sys.executable, "eval/jevbench/run_decoder.py", "--model", str(MERGED), "--run-id", "merged-hf",
              "--suite", "jevbench", "--case-ids", "data/eval/canary_ids.txt", "--batch", "16",
              *DEVICE_ARGS.split()], "verify.log", 30)
    sys.path.insert(0, str(W))
    from s1.decide import agreement, load_predictions, slice_mean_accuracy
    m = load_predictions(W / "results/raw/jevbench/merged-hf/predictions.jsonl") \
        if (W / "results/raw/jevbench/merged-hf/predictions.jsonl").exists() else {}
    ref = load_predictions(W / "results/raw/jevbench/g26-bos-vllm/predictions.jsonl")
    share, n = agreement(m, ref)
    ok = rc == 0 and n >= MIN_CASES and share >= AGREE_MIN
    record("verify", {"exit": rc, "agreement_with_adapter_run": share, "n": n, "required": AGREE_MIN, "pass": ok,
                      "acc_merged": round(slice_mean_accuracy(m), 4) if m else None,
                      "acc_adapter_same_cases": round(slice_mean_accuracy({k: ref[k] for k in m if k in ref}), 4) if m else None})
    status(f"verify: merged (HF, <bos>) vs adapter run (vLLM): agreement {share:.3f} on {n} cases -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        status("STOPPED: merged model not verified; nothing uploaded")
        return

    if not TOKEN.exists():
        record("upload", {"skipped": "no token"})
        status("upload: skipped (no token)")
        return
    os.environ["HF_HUB_DISABLE_XET"] = "1"  # plain uploads: no local chunk cache filling the disk
    from huggingface_hub import HfApi
    api = HfApi(token=TOKEN.read_text().strip())
    api.create_repo(REPO, private=True, exist_ok=True)
    if not api.repo_info(REPO).private:
        record("upload", {"error": "repo is public; not uploading"})
        status("STOPPED: repo is public; nothing uploaded")
        return
    (MERGED / "README.md").write_text(
        "# s1 decision model: Gemma 4 26B-A4B, LoRA merged (private)\n\n"
        "Typed decisions (choice, yes/no, ordinal score) in one forward pass, read from the option-letter logits after "
        "the reply prefix `Answer:`. Prompts must start with `<bos>`; the tokenizer does not add it.\n\n"
        "Base: google/gemma-4-26B-A4B-it (Apache-2.0). Fine-tune: LoRA r=64 on attention and dense MLP, 2,718 steps on "
        "~87K rows; snapshot 1500 chosen on dev splits.\n\n"
        "Test (JevBench 1,700-case subset): 0.808 raw, 0.817 calibrated; custom set 0.971 raw, 0.981 calibrated.\n\n"
        f"Merge check: {json.dumps(results.get('verify'))}\n")
    t0 = time.time()
    api.upload_folder(folder_path=str(MERGED), repo_id=REPO, path_in_repo="bf16",
                      commit_message="Merged bf16 weights (step 1500, trained with <bos>)")
    files = [f.path for f in api.list_repo_tree(REPO, path_in_repo="bf16")]
    record("upload", {"repo": REPO, "private": True, "files": len(files), "minutes": round((time.time() - t0) / 60, 1)})
    status(f"upload: {len(files)} files to {REPO}/bf16 in {(time.time() - t0) / 60:.1f} min")
    status("PACKAGE DONE")


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
T0 = time.time()
try:
    main()
except Exception as exc:
    status(f"ABORTED: {exc!r}"[:300])
finally:
    TOKEN.unlink(missing_ok=True)  # the upload token never outlives the session
    status(f"token removed from VM: {not TOKEN.exists()}")
rc = subprocess.run(f"cd {W} && tar czf {BASE}/pkg_out.tgz results/raw -C {BASE} package.json $(cd {BASE} && ls *.log)",
                    shell=True).returncode
status(f"packed pkg_out.tgz exit {rc}")
status(f"SESSION DONE, elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("Q_HOLD_MIN", 10)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
