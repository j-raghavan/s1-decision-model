"""Notebook cell: decoder full run, option A (one automated session, every decision by a pre-registered rule).

Method: docs/methodology.md. Rules: s1/decide.py, tested against saved runs in
tests/test_decide.py.

  0. setup: pinned transformers/peft, torchao removed, model downloaded
  1. canary: the pilot-2 adapter scored through HF on 200 fixed JevBench cases must agree >= 93% with its saved
     vLLM predictions (pilot 1: 97.4%; a missing or wrong adapter scores ~0.86), or the session stops untrained
  2. full run on targets_full_new (150K rows, 30% policy-reasoning rows, injection rows as in pilot 2):
     4,688 steps, snapshots every 500 (pulled to the Mac by colab/pull_snapshots.sh), 4-hour cap
  3. the untuned model and every snapshot scored on jevbench_dev and custom_dev2 (vLLM 0.31.0 in its own
     environment, every adapter verified by a request and by a pilot-2 canary after swapping; HF if vLLM is
     unusable); pilot 2 is scored too, for reference only
  4. choice (decide.choose_snapshot_ni): the best JevBench-dev snapshot among those with full coverage and
     custom-dev accuracy at most 1.5 points below the untuned model (paired); none eligible: the least-losing
     complete snapshot is scored and the failure is reported
  5. the chosen snapshot is scored once on the test sets; results packed into /content/full_out.tgz

Expects /content/s1_full.tgz and the pilot-2 adapter uploaded flat to /content (adapter_model.safetensors,
adapter_config.json). Keeps the kernel busy and holds the VM until /content/downloaded.flag appears.

    colab exec -s <session> -f colab/session_decoder_full_a.py

F_* environment variables shrink everything for a local dry run with a small model.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = os.environ.get("F_MODEL", "google/gemma-4-26B-A4B-it")
BASE = Path(os.environ.get("F_BASE", "/content"))
CUDA = os.environ.get("F_DEVICE", "cuda") == "cuda"
FULL_STEPS = int(os.environ.get("F_FULL_STEPS", 4688))  # 150,000 rows / 32 per step
FULL_SAVE = int(os.environ.get("F_FULL_SAVE", 500))
# The trainer fits its step count into 85% of this budget after timing 10 steps, so 255 bounds training to about
# 217 minutes: the full 4,688 steps at the measured 2.77 s/step, fewer steps if the GPU runs slower.
FULL_MINUTES = float(os.environ.get("F_FULL_MINUTES", 255))
BATCH = os.environ.get("F_BATCH", "16")
VAL_LIMIT = os.environ.get("F_VAL_LIMIT", "400")
LIMIT = ["--limit", os.environ["F_LIMIT"]] if os.environ.get("F_LIMIT") else []
CANARY_MIN = float(os.environ.get("F_CANARY_MIN", 0.93))
VLLM_VERSION = "0.31.0"  # the version that served the pilot-2 LoRA (pilot-2 vllm.log)
W = BASE / "s1"
STATUS = BASE / "session.log"
FLAG = BASE / "downloaded.flag"
REPORT = BASE / "decisions.json"
PILOT2 = BASE / "adapter_pilot2"
VENV = BASE / "vllm_env"
DEV = ("jevbench_dev", "custom_dev2")
TARGETS = "data/train/targets_full_new.jsonl"
decisions: dict = {}


def status(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    decisions[key] = value
    REPORT.write_text(json.dumps(decisions, indent=1))


def run(args: list[str], log: str, timeout_min: float) -> int:
    try:
        return subprocess.run(args, cwd=W, stdout=open(BASE / log, "w"), stderr=subprocess.STDOUT,
                              timeout=timeout_min * 60).returncode
    except subprocess.TimeoutExpired:
        return -9


def rules():
    sys.path.insert(0, str(W))
    import s1.decide as d
    return d


def preds(suite: str, run_id: str) -> dict:
    p = W / f"results/raw/{suite}/{run_id}/predictions.jsonl"
    return rules().load_predictions(p) if p.exists() else {}


def n_dev_cases(suite: str) -> int:
    """Yes/no and choice cases the untuned model must answer. With F_LIMIT (dry runs, HF only) the HF runner scores
    the first N cases of the file, so only those count."""
    sys.path.insert(0, str(W / "eval" / "jevbench"))
    from paths import SUITES  # the same registry the scoring runners read, so file names cannot disagree
    lines = open(SUITES[suite]["cases"]).readlines()
    lines = lines[: int(LIMIT[1])] if LIMIT else lines
    return sum(1 for line in lines if '"task_type": "score"' not in line)


def hf_score(run_id: str, suite: str, adapter: Path | None, extra: tuple = ()) -> int:
    args = [sys.executable, "eval/jevbench/run_decoder.py", "--model", MODEL, "--run-id", run_id, "--suite", suite,
            "--batch", BATCH, *extra, *LIMIT] + (["--adapter", str(adapter)] if adapter else [])
    return run(args, f"hf_{run_id}_{suite}.log", 30)


def train(targets: str, out: Path, steps: int, save_every: int, minutes: float, log: str) -> int:
    # evaluation inside training is off: every decision is taken on the dev splits afterwards
    return run([sys.executable, "-m", "s1.train_decoder", "--model", MODEL, "--targets", targets, "--out", str(out),
                "--max-steps", str(steps), "--batch", BATCH, "--accum", "2", "--lr", "1e-4", "--rank", "64",
                "--save-every", str(save_every), "--eval-every", "1000000", "--val-limit", VAL_LIMIT,
                "--skip-initial-eval", "--max-minutes", str(minutes)], log, minutes + 30)


def steps_run(log: str, planned: int) -> int:
    """Steps the trainer actually ran (--max-minutes can shrink the plan), from its final log record."""
    for line in reversed(open(BASE / log, errors="replace").read().splitlines()):
        if line.startswith("{") and '"done": true' in line:
            try:
                return int(json.loads(line)["steps"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return planned


def snapshots(out: Path, steps: int, save_every: int) -> dict[str, Path]:
    """step-N snapshots plus the final adapter; a step-N right before the end duplicates the final one."""
    margin = max(1, min(10, save_every // 2))  # 10 steps for the real intervals (135, 500)
    found = {p.name: p for p in sorted(out.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
             if (p / "adapter_model.safetensors").exists() and int(p.name.split("-")[1]) < steps - margin}
    if (out / "last" / "adapter_model.safetensors").exists():
        found["last"] = out / "last"
    return found


# ---------------------------------------------------------------- vLLM in its own environment

def vllm_env() -> bool:
    """vLLM in a separate venv, so its torch never replaces the one training uses."""
    if (VENV / "bin" / "python").exists():
        return True
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv", "httpx", "tqdm"], check=True,
                       stdout=open(BASE / "vllm_install.log", "w"), stderr=subprocess.STDOUT)
        subprocess.run(["uv", "venv", str(VENV), "--python", sys.executable], check=True,
                       stdout=open(BASE / "vllm_install.log", "a"), stderr=subprocess.STDOUT)
        subprocess.run(["uv", "pip", "install", "--python", str(VENV / "bin" / "python"), f"vllm=={VLLM_VERSION}"],
                       check=True, stdout=open(BASE / "vllm_install.log", "a"), stderr=subprocess.STDOUT, timeout=1200)
        return True
    except Exception as exc:
        status(f"vLLM environment failed ({exc!r})"[:300] + "; scoring falls back to HF")
        return False


def probe(name: str) -> bool:
    """One 1-token completion through an adapter: a server can be healthy while an adapter fails to load."""
    body = json.dumps({"model": name, "prompt": "Answer:", "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request("http://localhost:8000/v1/completions", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status == 200 and bool(json.loads(r.read())["choices"])
    except Exception:
        return False


def start_vllm(loras: dict[str, Path]):
    """vLLM serving the base model and every adapter, verified by a real request per adapter; None means use HF."""
    if not (CUDA and vllm_env()):
        return None
    common = [str(VENV / "bin" / "python"), "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--port", "8000",
              "--max-model-len", "8192", "--max-logprobs", "20", "--gpu-memory-utilization", "0.90", "--enforce-eager",
              "--enable-lora", "--max-lora-rank", "64", "--lora-modules", *(f"{n}={p}" for n, p in loras.items())]
    # candidates are scored one at a time, so one adapter slot on the GPU is enough (the rest wait in CPU memory);
    # the second form is what served pilot 2, in case this vLLM rejects the CPU-cache option
    for extra in (["--max-loras", "1", "--max-cpu-loras", str(max(1, len(loras)))], ["--max-loras", str(max(1, len(loras)))]):
        server = subprocess.Popen(common + extra, stdout=open(BASE / "vllm.log", "a"), stderr=subprocess.STDOUT,
                                  start_new_session=True,
                                  env={**os.environ, "PATH": f"{VENV / 'bin'}:{os.environ.get('PATH', '')}"})  # own process group: the engine child dies with it
        t0, healthy = time.time(), False
        while time.time() - t0 < 600 and server.poll() is None:
            try:
                with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
                    healthy = r.status == 200
            except Exception:
                pass
            if healthy:
                break
            time.sleep(10)
        if healthy:
            failed = [n for n in [MODEL, *loras] if not probe(n)]
            if not failed:
                status(f"vLLM ready in {time.time() - t0:.0f}s ({' '.join(extra)}), all {len(loras)} adapters answer")
                return server
            status(f"vLLM up but these did not answer: {failed[:5]}")
        else:
            status(f"vLLM did not become healthy with {' '.join(extra)} (exit {server.poll()})")
        stop_vllm(server)
    status("vLLM unusable; scoring falls back to HF")
    return None


def stop_vllm(server) -> None:
    """Stop the server and its engine child (whole process group), so the GPU is free before training again."""
    if server is None:
        return
    import signal
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=60)
    except ProcessLookupError:
        pass
    time.sleep(10)


def gpu_used_gb() -> float:
    out = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True,
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[0]) / 1024 if out else 0.0


def ensure_gpu_free() -> bool:
    """Before training: no leftover vLLM process may hold GPU memory."""
    if not CUDA:
        return True
    for _ in range(3):
        used = gpu_used_gb()
        if used < 4:
            return True
        status(f"GPU still has {used:.0f} GB in use before training; killing leftover vLLM processes")
        subprocess.run("pkill -9 -f vllm", shell=True)
        time.sleep(20)
    return gpu_used_gb() < 4


def vllm_score(model: str, run_id: str, suite: str, log: str, extra: tuple = ()) -> int:
    """Score through the running vLLM server; error rows are removed and retried once."""
    args = [sys.executable, "eval/jevbench/run_ollama.py", "--backend", "vllm", "--template", "gemma4", "--model", model,
            "--run-id", run_id, "--suite", suite, "--workers", "64", *extra, *([] if extra else LIMIT)]
    rc = run(args, log, 20)
    p = W / f"results/raw/{suite}/{run_id}/predictions.jsonl"
    if p.exists():
        lines = p.read_text().splitlines(keepends=True)

        def answered(line: str) -> bool:  # a cut-off last line (timeout kill) counts as unanswered
            try:
                return json.loads(line).get("error") is None and line.endswith("\n")
            except json.JSONDecodeError:
                return False
        good = [line for line in lines if answered(line)]
        if len(good) < len(lines):
            p.write_text("".join(good))
            rc = run(args, log.replace(".log", "_retry.log"), 20)
    return rc


def vllm_canary(tag: str) -> bool:
    """Through the running server, pilot 2 on the 200 canary cases must reproduce its saved vLLM predictions
    (served alone in pilot 2). Proves the adapter weights are applied under this server's adapter swapping:
    the untuned model would agree only ~0.86."""
    rid = f"canary-pilot2-vllm-{tag.replace(' ', '')}"
    rc = vllm_score("pilot2", rid, "jevbench", f"canary_vllm_{tag.replace(' ', '')}.log", ("--case-ids", "data/eval/canary_ids.txt"))
    share, n = rules().agreement(preds("jevbench", rid), preds("jevbench", "g26-pilot2-vllm"))
    want = sum(1 for _ in open(W / "data" / "eval" / "canary_ids.txt"))
    ok = rc == 0 and n >= 0.98 * want and share >= 0.95
    status(f"{tag}: vLLM canary (pilot 2 after adapter swapping) agreement {share:.3f} on {n}/{want} -> {'PASS' if ok else 'FAIL'}")
    return ok


def score_dev(candidates: dict[str, Path], tag: str) -> str:
    """Untuned + candidates on both dev splits. Returns the engine used ('vllm' or 'hf')."""
    server = start_vllm(candidates | {"pilot2": PILOT2})
    engine = "vllm" if server is not None else "hf"
    for suite in DEV:
        if engine == "vllm":
            rc = vllm_score(MODEL, f"base-{engine}", suite, f"dev_base_{suite}.log")
        else:
            done = len(preds(suite, "base-hf")) >= rules().MIN_COVERAGE * n_dev_cases(suite)
            rc = 0 if done else hf_score("base-hf", suite, None)
        status(f"{tag} dev {suite}: untuned exit {rc}")
        for name, path in candidates.items():
            if engine == "vllm":
                rc = vllm_score(name, f"{name}-{engine}", suite, f"dev_{name}_{suite}.log")
            else:
                rc = hf_score(f"{name}-{engine}", suite, path)
            status(f"{tag} dev {suite}: {name} exit {rc}")
    if engine == "vllm" and not vllm_canary(tag):
        stop_vllm(server)
        status(f"{tag}: vLLM results discarded; re-scoring through HF")
        for suite in DEV:
            done = len(preds(suite, "base-hf")) >= rules().MIN_COVERAGE * n_dev_cases(suite)
            status(f"{tag} dev {suite} (HF): untuned exit {0 if done else hf_score('base-hf', suite, None)}")
            for name, path in candidates.items():
                status(f"{tag} dev {suite} (HF): {name} exit {hf_score(f'{name}-hf', suite, path)}")
        return "hf"
    stop_vllm(server)
    return engine


def judge(candidates: list[str], engine: str, tag: str):
    d = rules()
    base_c, base_j = preds("custom_dev2", f"base-{engine}"), preds("jevbench_dev", f"base-{engine}")
    if not d.base_complete(base_c, base_j, n_dev_cases("custom_dev2"), n_dev_cases("jevbench_dev")):
        status(f"{tag}: untuned dev scoring incomplete ({len(base_c)} custom, {len(base_j)} jev); cannot judge")
        return None, [], base_c
    best, report = d.choose_snapshot_ni(base_c, base_j, {n: (preds("custom_dev2", f"{n}-{engine}"),
                                                             preds("jevbench_dev", f"{n}-{engine}")) for n in candidates})
    status(f"{tag}: untuned dev jev {d.slice_mean_accuracy(base_j):.3f} custom {d.slice_mean_accuracy(base_c):.3f} ({engine})")
    for r in report:
        status(f"{tag} {r['name']:14s} jev_dev {r['jev_acc']:.3f} coverage {r['coverage']:.2f} custom change "
               f"{r['custom_change'] * 100:+.2f} pts (-{r['custom_newly_wrong']}/+{r['custom_newly_right']}) "
               f"{'eligible' if r['eligible'] else 'NOT eligible'}")
    return best, report, base_c


# ---------------------------------------------------------------- stages

def setup() -> None:
    subprocess.run(f"mkdir -p {W} && tar xzf {BASE}/s1_full.tgz -C {W}", shell=True, check=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):  # uploaded flat by scripts/colab_upload.sh
        if (BASE / name).exists():
            PILOT2.mkdir(exist_ok=True)
            (BASE / name).rename(PILOT2 / name)
    if CUDA:
        status(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                              capture_output=True, text=True).stdout.strip())
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.18.0", "peft==0.21.2",
                        "accelerate==1.15.0", "httpx", "tqdm"], check=True)
        # Colab preinstalls torchao 0.10, which PEFT rejects when it builds LoRA layers; nothing here uses it.
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    t0 = time.time()
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "tokenizer*"])
    status(f"setup done, model downloaded in {time.time() - t0:.0f}s, pilot-2 adapter present: "
           f"{(PILOT2 / 'adapter_model.safetensors').exists()}")


def canary() -> bool:
    rc = hf_score("canary-pilot2-hf", "jevbench", PILOT2, ("--case-ids", "data/eval/canary_ids.txt"))
    share, n = rules().agreement(preds("jevbench", "canary-pilot2-hf"), preds("jevbench", "g26-pilot2-vllm"))
    want = sum(1 for _ in open(W / "data" / "eval" / "canary_ids.txt")) if not LIMIT else int(LIMIT[1])
    ok = rc == 0 and n >= 0.98 * want and share >= CANARY_MIN
    record("canary", {"exit": rc, "agreement": share, "cases": n, "expected_cases": want, "required": CANARY_MIN, "pass": ok})
    status(f"canary: HF vs saved vLLM agreement {share:.3f} on {n}/{want} cases (need {CANARY_MIN}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def full_run() -> dict[str, Path]:
    if not ensure_gpu_free():
        status("GPU memory not free before the full run; stopping")
        record("full_run", {"error": "GPU not free"})
        return {}
    rc = train(TARGETS, BASE / "adapterF", FULL_STEPS, FULL_SAVE, FULL_MINUTES, "trainF.log")
    snaps = snapshots(BASE / "adapterF", steps_run("trainF.log", FULL_STEPS), FULL_SAVE)
    status(f"full run exit {rc}, snapshots {list(snaps)}")
    record("full_run", {"targets": TARGETS, "exit": rc, "snapshots": list(snaps)})
    return snaps


def final(snaps: dict[str, Path]) -> None:
    d = rules()
    cands = {f"F-{k}": p for k, p in snaps.items()}
    engine = score_dev(cands | {"pilot2": PILOT2}, "final")  # pilot 2 for reference only
    best, report, _ = judge(list(cands), engine, "final")
    ref = rules().choose_snapshot_ni(preds("custom_dev2", f"base-{engine}"), preds("jevbench_dev", f"base-{engine}"),
                                     {"pilot2": (preds("custom_dev2", f"pilot2-{engine}"), preds("jevbench_dev", f"pilot2-{engine}"))})[1]
    if ref:
        r = ref[0]
        status(f"reference pilot2 jev_dev {r['jev_acc']:.3f} custom change {r['custom_change'] * 100:+.2f} pts (not a candidate)")
    fallback = None
    if best is None and report:  # pre-registered: report the failure, test the least-regressing complete snapshot
        complete = [r for r in report if r["coverage"] >= d.MIN_COVERAGE]
        if complete:
            fallback = max(complete, key=lambda r: (r["custom_change"], r["jev_acc"]))["name"]
        else:
            status("final: no snapshot has complete dev scores; nothing is scored on test")
    chosen = best or fallback
    record("final", {"engine": engine, "report": report, "best": best, "fallback_no_eligible": fallback, "pilot2_reference": ref})
    status(f"chosen snapshot: {chosen}" + (" (NO snapshot met the custom-dev margin)" if best is None else ""))
    if chosen is None:
        return
    path = cands[chosen]
    server = start_vllm({chosen: path, "pilot2": PILOT2}) if engine == "vllm" else None
    if server is not None and not vllm_canary("test"):
        stop_vllm(server)
        server = None
    test_engine = "vllm" if server is not None else "hf"
    for suite in ("jevbench", "custom"):  # the test sets, scored once
        if test_engine == "vllm":
            rc = vllm_score(chosen, "g26-full-vllm", suite, f"test_{suite}.log")
        else:
            rc = hf_score("g26-full-hf", suite, path)
        status(f"TEST {suite} exit {rc}")
    stop_vllm(server)
    rid = f"g26-full-{test_engine}"
    base_rid = "gemma4-26b-bf16-vllm" if test_engine == "vllm" else "g26-hf-zeroshot"  # same engine as the candidate
    jev, cust = preds("jevbench", rid), preds("custom", rid)
    reg = d.regression(preds("custom", base_rid), cust)
    reg["accuracy_change"] = d.accuracy_change(preds("custom", base_rid), cust)
    reg["non_inferior"] = d.non_inferior(preds("custom", base_rid), cust)
    record("test", {"run_id": rid, "snapshot": chosen, "adapter": str(path), "jevbench_acc": d.slice_mean_accuracy(jev),
                    "custom_acc": d.slice_mean_accuracy(cust), "custom_vs_untuned": reg, "untuned_reference": base_rid})
    status(f"TEST jevbench raw acc {d.slice_mean_accuracy(jev):.3f}, custom raw acc {d.slice_mean_accuracy(cust):.3f}, "
           f"custom vs untuned -{reg['overall']['newly_wrong']}/+{reg['overall']['newly_right']} p={reg['overall']['p']:.3f}, "
           f"custom change {reg['accuracy_change'] * 100:+.2f} pts, within 1.5-point margin: {reg['non_inferior']}")


def pack() -> None:
    chosen = decisions.get("test", {}).get("adapter")
    extra = str(Path(chosen).relative_to(BASE)) if chosen else ""
    rc = subprocess.run(f"cd {W} && tar czf {BASE}/full_out.tgz results/raw -C {BASE} decisions.json {extra} "
                        f"$(cd {BASE} && ls *.log)", shell=True).returncode
    status(f"packed full_out.tgz exit {rc}, {(BASE / 'full_out.tgz').stat().st_size / 1e6:.0f} MB")


def main() -> None:
    setup()
    if not canary():
        status("STOPPED: canary failed; nothing trained")
        return
    snaps = full_run()
    if snaps:
        final(snaps)


FLAG.unlink(missing_ok=True)
STATUS.write_text("")
T0 = time.time()
try:
    main()
except Exception as exc:  # logged so the watcher sees it; results so far are still packed
    status(f"ABORTED: {exc!r}"[:300])
try:
    pack()
except Exception as exc:
    status(f"pack failed: {exc!r}"[:200])
status(f"SESSION DONE, elapsed {(time.time() - T0) / 60:.0f} min")
t0 = time.time()
while not FLAG.exists() and time.time() - t0 < float(os.environ.get("F_HOLD_MIN", 20)) * 60:
    time.sleep(15)
status("download confirmed" if FLAG.exists() else "no download confirmation; releasing")
