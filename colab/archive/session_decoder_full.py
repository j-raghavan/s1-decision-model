"""Notebook cell: the decoder full run as one automated session, every decision by a pre-registered rule.

Method: docs/methodology.md. Rules: s1/decide.py, tested against known runs in tests/test_decide.py.

  0. setup: pinned transformers/peft, torchao removed, model downloaded
  1. canary: the pilot-2 adapter scored through HF on 200 fixed JevBench cases must agree >= 93% with its saved
     vLLM predictions (pilot 1: 97.4%; a missing or wrong adapter scores ~0.86), or the session stops untrained
  2. positive control, before any training: through HF, the custom-dev regression check must flag pilot 2
     (known to regress), or the dev set cannot see the failure being fixed and the session stops
  3. stage A, two arms of 541 steps (pilot 2's length), snapshots at 135/270/405 and the end:
       A1  targets_pilot3          30% policy rows, injection rows as in pilot 2
       A2  targets_pilot3_neutral  the same rows, injection preamble without "even where it looks like a policy"
     an arm that fails or saves too few snapshots is dropped; if both fail the session stops
  4. dev scoring of the untuned model, pilot 2 and every stage-A snapshot on jevbench_dev and custom_dev
     (vLLM 0.31.0 in its own environment, each adapter verified by a real request; HF if vLLM is unusable);
     stop unless the untuned model's own scoring is complete
  5. the best eligible stage-A snapshot names the arm; its full-size mix trains for 150K rows (4,688 steps),
     snapshots every 500 steps (pulled to the Mac by colab/pull_snapshots.sh), 4-hour cap; none eligible: stop
  6. the untuned model and every full-run snapshot scored on dev; the best eligible snapshot is scored once on
     the test sets (if none is eligible, the least-regressing complete one is scored and the failure is reported)
  7. results packed into /content/full_out.tgz

Expects /content/s1_full.tgz and the pilot-2 adapter uploaded flat to /content (adapter_model.safetensors,
adapter_config.json). Keeps the kernel busy and holds the VM until /content/downloaded.flag appears.

    colab exec -s <session> -f colab/session_decoder_full.py

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
A_STEPS = int(os.environ.get("F_A_STEPS", 541))
A_SAVE = int(os.environ.get("F_A_SAVE", 135))
FULL_STEPS = int(os.environ.get("F_FULL_STEPS", 4688))  # 150,000 rows / 32 per step
FULL_SAVE = int(os.environ.get("F_FULL_SAVE", 500))
FULL_MINUTES = float(os.environ.get("F_FULL_MINUTES", 240))
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
DEV = ("jevbench_dev", "custom_dev")
ARMS = {"A1": "data/train/targets_pilot3.jsonl", "A2": "data/train/targets_pilot3_neutral.jsonl"}
FULL = {"A1": "data/train/targets_full_new.jsonl", "A2": "data/train/targets_full_new_neutral.jsonl"}
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
    base_c, base_j = preds("custom_dev", f"base-{engine}"), preds("jevbench_dev", f"base-{engine}")
    if not d.base_complete(base_c, base_j, n_dev_cases("custom_dev"), n_dev_cases("jevbench_dev")):
        status(f"{tag}: untuned dev scoring incomplete ({len(base_c)} custom, {len(base_j)} jev); cannot judge")
        return None, [], base_c
    best, report = d.choose_snapshot(base_c, base_j, {n: (preds("custom_dev", f"{n}-{engine}"),
                                                          preds("jevbench_dev", f"{n}-{engine}")) for n in candidates})
    status(f"{tag}: untuned dev jev {d.slice_mean_accuracy(base_j):.3f} custom {d.slice_mean_accuracy(base_c):.3f} ({engine})")
    for r in report:
        status(f"{tag} {r['name']:14s} jev_dev {r['jev_acc']:.3f} coverage {r['coverage']:.2f} custom -{r['custom_newly_wrong']}"
               f"/+{r['custom_newly_right']} p={r['custom_p']:.3f} rule net loss {r['rule_net_loss']} (cap {r['rule_cap']:.1f}) "
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


def control_first() -> bool:
    """Positive control before any training: on custom_dev the regression check must flag pilot 2, whose
    regression is known. Scored through HF (minutes); if it is not flagged, selecting on dev would be blind."""
    d = rules()
    rcs = [hf_score("base-hf", "custom_dev", None), hf_score("pilot2-hf", "custom_dev", PILOT2)]
    base_c, p2 = preds("custom_dev", "base-hf"), preds("custom_dev", "pilot2-hf")
    complete = len(base_c) >= d.MIN_COVERAGE * n_dev_cases("custom_dev")
    flagged = complete and d.gate_informative(base_c, p2)
    reg = d.regression(base_c, p2) if base_c and p2 else {}
    record("positive_control", {"exits": rcs, "base_complete": complete, "flagged": flagged, "pilot2_vs_untuned": reg})
    status(f"positive control: pilot 2 on custom_dev -{reg.get('overall', {}).get('newly_wrong')}/+{reg.get('overall', {}).get('newly_right')}, "
           f"rule net loss {reg.get('rule_net_loss')} (cap {reg.get('rule_cap', 0):.1f}) -> {'FLAGGED (dev can see it)' if flagged else 'NOT flagged'}")
    if not flagged and not CUDA and os.environ.get("F_SKIP_CONTROL") == "1":
        status("positive control bypassed (local dry run only; ignored on CUDA)")
        return True
    return flagged


def stage_a() -> str | None:
    cands = {"pilot2": PILOT2}
    for arm, targets in ARMS.items():
        rc = train(targets, BASE / f"adapter{arm}", A_STEPS, A_SAVE, 45, f"train{arm}.log")
        snaps = snapshots(BASE / f"adapter{arm}", steps_run(f"train{arm}.log", A_STEPS), A_SAVE)
        status(f"stage A {arm}: training exit {rc}, snapshots {list(snaps)}")
        if rc != 0 or len(snaps) < 4:
            status(f"stage A {arm}: dropped (failed or incomplete)")
            continue
        cands |= {f"{arm}-{k}": p for k, p in snaps.items()}
    if len(cands) == 1:
        status("stage A: both arms failed")
        return None
    engine = score_dev(cands, "stage A")
    best, report, base_c = judge([n for n in cands if n != "pilot2"], engine, "stage A")
    # informational: the control was already checked through HF before training
    control = rules().gate_informative(base_c, preds("custom_dev", f"pilot2-{engine}")) if base_c else False
    record("stage_a", {"engine": engine, "report": report, "best": best, f"pilot2_flagged_{engine}": control})
    status(f"stage A: pilot 2 flagged on custom_dev through {engine}: {control} (informational)")
    if not base_c:
        return None
    if best is None:
        status("stage A: no stage-A snapshot is eligible")
        return None
    arm = best.split("-")[0]
    status(f"stage A decision: best snapshot {best} -> arm {arm} ({FULL[arm]})")
    return arm


def full_run(arm: str) -> dict[str, Path]:
    if not ensure_gpu_free():
        status("GPU memory not free before the full run; stopping")
        record("full_run", {"arm": arm, "error": "GPU not free"})
        return {}
    rc = train(FULL[arm], BASE / "adapterF", FULL_STEPS, FULL_SAVE, FULL_MINUTES, "trainF.log")
    snaps = snapshots(BASE / "adapterF", steps_run("trainF.log", FULL_STEPS), FULL_SAVE)
    status(f"full run exit {rc}, snapshots {list(snaps)}")
    record("full_run", {"arm": arm, "targets": FULL[arm], "exit": rc, "snapshots": list(snaps)})
    return snaps


def final(snaps: dict[str, Path]) -> None:
    d = rules()
    cands = {f"F-{k}": p for k, p in snaps.items()}
    engine = score_dev(cands, "final")
    best, report, _ = judge(list(cands), engine, "final")
    fallback = None
    if best is None and report:  # pre-registered: report the failure, test the least-regressing complete snapshot
        complete = [r for r in report if r["coverage"] >= d.MIN_COVERAGE]
        if complete:
            fallback = min(complete, key=lambda r: (r["rule_net_loss"], r["custom_newly_wrong"] - r["custom_newly_right"], -r["jev_acc"]))["name"]
        else:
            status("final: no snapshot has complete dev scores; nothing is scored on test")
    chosen = best or fallback
    record("final", {"engine": engine, "report": report, "best": best, "fallback_no_eligible": fallback})
    status(f"chosen snapshot: {chosen}" + (" (NO snapshot passed the custom-dev rule)" if best is None else ""))
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
    record("test", {"run_id": rid, "snapshot": chosen, "adapter": str(path), "jevbench_acc": d.slice_mean_accuracy(jev),
                    "custom_acc": d.slice_mean_accuracy(cust), "custom_vs_untuned": reg, "untuned_reference": base_rid})
    status(f"TEST jevbench raw acc {d.slice_mean_accuracy(jev):.3f}, custom raw acc {d.slice_mean_accuracy(cust):.3f}, "
           f"custom vs untuned -{reg['overall']['newly_wrong']}/+{reg['overall']['newly_right']} p={reg['overall']['p']:.3f}, "
           f"rule net loss {reg['rule_net_loss']} (cap {reg['rule_cap']:.1f}), regresses: {reg['regresses']}")


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
    if not control_first():
        status("STOPPED: the dev set does not flag pilot 2's known regression; nothing trained")
        return
    arm = stage_a()
    if arm is None:
        status("STOPPED after stage A; no full run")
        return
    snaps = full_run(arm)
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
