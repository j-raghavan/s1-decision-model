"""Gemma 4 prompts must start with <bos> on every path that tokenizes them itself (HF readout, vLLM).

Without it the model loses 4-13 points on knowledge questions; the tokenizer does not add it (add_bos_token=False).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval" / "jevbench")]


def test_vllm_fetcher_prepends_bos_once():
    import run_ollama

    sent = []

    class Client:
        def post(self, url, json):
            sent.append(json["prompt"])

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"logprobs": {"top_logprobs": [{" A": -0.1}]}}], "usage": {"prompt_tokens": 3}}
            return R()

    fetch = run_ollama.vllm_fetcher("http://x", bos="<bos>")
    fetch(Client(), "m", "<|turn>user\nhi")
    fetch(Client(), "m", "<bos><|turn>user\nhi")
    assert sent == ["<bos><|turn>user\nhi", "<bos><|turn>user\nhi"]
    run_ollama.vllm_fetcher("http://x")(Client(), "m", "plain")  # default: unchanged (gpt-oss, other templates)
    assert sent[-1] == "plain"


@pytest.mark.skipif(not (Path.home() / ".cache/huggingface/hub/models--google--gemma-4-E4B-it").exists(), reason="tokenizer not cached")
def test_hf_readout_tokens_start_with_bos():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
    assert getattr(tok, "add_bos_token", True) is False  # the reason the fix is needed
    bos = tok.bos_token
    prompts = [p if p.startswith(bos) else bos + p for p in ["<|turn>user\nhi", "<bos><|turn>user\nhi"]]
    ids = tok(prompts, add_special_tokens=False)["input_ids"]
    assert all(seq[0] == tok.bos_token_id and seq[1] != tok.bos_token_id for seq in ids)


def test_decoder_bos_switch_reaches_every_last_logits_call():
    import inspect

    from s1 import decoder
    src = inspect.getsource(decoder.option_log_probs)
    assert src.count("last_logits(") == 3 and src.count("add_bos)") == 3  # letter pass, first digit, second digit
    assert "add_bos: bool = True" in inspect.getsource(decoder.last_logits)


def test_ollama_fetcher_prepends_bos_once(monkeypatch):
    import run_ollama
    sent = []
    monkeypatch.setattr(run_ollama, "next_token_logprobs", lambda client, model, prompt: (sent.append(prompt) or ({}, 0)))
    f = run_ollama.ollama_fetcher("<bos>")
    f(None, "m", "<|turn>user\nhi")
    f(None, "m", "<bos><|turn>user\nhi")
    run_ollama.ollama_fetcher()(None, "m", "plain")  # library models add <bos> themselves: unchanged
    assert sent == ["<bos><|turn>user\nhi", "<bos><|turn>user\nhi", "plain"]
