"""scripts/ollama_configure.py: the Ollama build's chat path must send the training prompt and must not think."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "jevbench"))
from run_ollama import GEMMA4_PROMPT  # noqa: E402

spec = importlib.util.spec_from_file_location("ollama_configure", ROOT / "scripts" / "ollama_configure.py")
oc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oc)


def gemma4_large_chat(user: str) -> str:
    """What Ollama's gemma4-large renderer writes for one user message with thinking off
    (model/renderers/gemma4.go: <bos>, the user turn, then the generation prompt with an empty thought block)."""
    return "<bos><|turn>user\n" + user.strip() + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def test_chat_render_equals_training_prompt_without_prefill():
    user = "You are a decision model.\n\nQUESTION: Which team?\n\nOPTIONS:\nA) billing: Payments\n\nAnswer with the option letter only."
    assert gemma4_large_chat(user) == "<bos>" + GEMMA4_PROMPT.format(user=user)


def test_configuration_disables_thinking_and_uses_the_large_renderer():
    req = oc.request("jrlabs01/s1", "jrlabs01/s1")
    assert req["renderer"] == "gemma4-large"  # the small variant omits the empty thought block
    assert req["parser"] == "gemma4-no-thinking"  # the gemma4 parser would re-add thinking
    assert "thinking" not in req["info"]["capabilities"]
    assert req["parameters"]["temperature"] == 0 and req["from"] == "jrlabs01/s1" and req["stream"] is False
