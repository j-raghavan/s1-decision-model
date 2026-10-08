"""Stand-in for a vLLM OpenAI-compatible server, for dry runs of Colab cells on a machine without a GPU.

Serves /health, /tokenize and /v1/completions (next-token top logprobs over option letters, deterministic per
model and prompt). /tokenize treats a leading "<bos>" as token 2, as Gemma 4's tokenizer does. Unknown model names
get a 404, so adapter probes can fail as they would on a real server.

    python tests/mock_vllm.py --port 8000 --model google/gemma-4-26B-A4B-it --lora-names pilot2,A500
"""

import argparse
import hashlib
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def tokens(prompt: str) -> list[int]:
    body = prompt[len("<bos>"):] if prompt.startswith("<bos>") else prompt
    ids = [1000 + (int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % 50000) for w in body.split()]
    return ([2] if prompt.startswith("<bos>") else []) + ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--lora-names", default="")
    args, _ = ap.parse_known_args()  # ignores vLLM-only flags
    names = {args.model, *[n for n in args.lora_names.split(",") if n]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {}) if self.path == "/health" else self.reply(404, {"error": "not found"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if req.get("model") not in names:
                return self.reply(404, {"error": f"model {req.get('model')} not found"})
            prompt = req.get("prompt", "")
            if self.path == "/tokenize":
                return self.reply(200, {"tokens": tokens(prompt), "count": len(tokens(prompt))})
            if self.path == "/v1/completions":
                seed = int(hashlib.md5((req["model"] + prompt).encode()).hexdigest()[:8], 16)
                scores = {f" {L}": -((seed >> (i % 24)) % 7) - i * 0.01 for i, L in enumerate(LETTERS[:10])}
                z = math.log(sum(math.exp(v) for v in scores.values()))
                top = {k: v - z for k, v in scores.items()}
                return self.reply(200, {"choices": [{"text": max(top, key=top.get), "logprobs": {"top_logprobs": [top]}}],
                                        "usage": {"prompt_tokens": len(tokens(prompt))}})
            return self.reply(404, {"error": "not found"})

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
