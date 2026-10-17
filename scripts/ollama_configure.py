"""Set the serving configuration of the Ollama build so that a plain chat call sends the training prompt.

Ollama gives every Gemma 4 import its built-in `gemma4` renderer and parser, and a renderer replaces the Modelfile
TEMPLATE for chat calls. For s1 that default is wrong in two ways:

  - the name `jrlabs01/s1` carries no size, so `gemma4` resolves to the small variant, which leaves out the empty
    thought channel the model was trained with (`<|channel>thought\\n<channel|>`)
  - the `gemma4` parser declares thinking with default on, so `ollama run` and chat calls open a thought channel and
    reason for up to thousands of tokens instead of answering

This creates a new tag from an existing build (no weights are copied) with the `gemma4-large` renderer, the
`gemma4-no-thinking` parser and capabilities without thinking. A chat call then sends exactly the training prompt
minus the `Answer:` prefill; raw prompts (`"raw": true`, what the API and examples send) are unchanged. The
capability list can only be replaced through `/api/create`'s `info` field, not a Modelfile, hence this script.

    python3 scripts/ollama_configure.py                              # jrlabs01/s1 -> s1-configured (local)
    python3 scripts/ollama_configure.py --source jrlabs01/s1 --target jrlabs01/s1   # then: ollama push jrlabs01/s1
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CONFIG = {
    "renderer": "gemma4-large",
    "parser": "gemma4-no-thinking",
    "info": {"capabilities": ["completion", "vision", "tools"]},
    "parameters": {"temperature": 0, "stop": ["<turn|>"]},
}


def request(source: str, target: str) -> dict:
    return {"model": target, "from": source, "stream": False, **CONFIG}


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="jrlabs01/s1", help="existing build to configure")
    ap.add_argument("--target", default="s1-configured", help="tag to write (may equal --source)")
    args = ap.parse_args()
    print(post("/api/create", request(args.source, args.target)))
    show = post("/api/show", {"model": args.target})
    print("capabilities:", show.get("capabilities"))
    if "thinking" in (show.get("capabilities") or []):
        print("thinking is still enabled: this Ollama version did not apply the configuration")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
