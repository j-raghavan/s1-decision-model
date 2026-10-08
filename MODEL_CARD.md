---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
language:
- en
datasets:
- j-raghavan/s1-decision-data
tags:
- decision-making
- classification
- calibration
- structured-data
- prompt-injection-detection
- gemma4
- lora
---

# s1-gemma4-26b-decision

**s1** is an open, calibrated *System One* decision model. Given a `state` (text or JSON) and a typed question, it
returns a probability for every option in **one forward pass**, with no text generated:

| Question type | You give | You get |
| --- | --- | --- |
| `choice` | keyed options with descriptions | a probability per option |
| `noul` (yes/no) | a condition | the probability of yes |
| `score` | an ordered list of levels | a probability per level, and the expected level |

It is [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (mixture of experts, 26B total and
about 4B active parameters) fine-tuned with LoRA on about 100,000 typed decision rows, and merged. Code, data and
evaluation are all open: [GitHub](https://github.com/j-raghavan/s1-decision-model) ·
[training data](https://huggingface.co/datasets/j-raghavan/s1-decision-data) ·
[website and leaderboard](https://j-raghavan.github.io/s1-decision-model).

## Results

Held-out test sets. The released snapshot was chosen on separate dev splits by a rule fixed before training.

| System | JevBench subset (1,700 cases) raw / calibrated | Custom structured decisions raw / calibrated | Prompt injections caught / false alarms |
| --- | --- | --- | --- |
| **s1** (this model, bf16) | **0.808 / 0.817** | **0.971 / 0.981** | **45/53, 0** |
| Gemma 4 26B-A4B, untuned | 0.770 / 0.799 | 0.948 / 0.960 | 35/53, 0 |
| **s1, int4 through Ollama** (Apple-silicon laptop) | **0.801 / 0.804** | **0.966 / 0.981** | **46/53, 0** |
| Jev 1.13 (closed; predictions published with JevBench) | 0.835 / 0.856 | — | — |

- Mean accuracy over JevBench's 11 accuracy slices; ECE with 10 bins. Raw ECE is 0.083 (Jev: 0.081).
- Paired against the untuned model on the same JevBench cases: 103 answers fixed, 58 broken (McNemar p = 0.0005).
- "Calibrated" uses per-task calibrators fitted on the dev splits only (`eval/jevbench/calibrate_from_dev.py`). That
  study covers the cases with usable option log-probabilities, where s1's raw JevBench score is 0.804 rather than 0.808.
- Differences under about ±1.5 points are within noise at this test size.

Latency per decision on one NVIDIA RTX PRO 6000 through vLLM: median 28 ms (p95 95 ms) with FP8 weights, 40 ms
(133 ms) in bf16.

## How to use

The prompt format matters. Render the question with lettered options, **start the prompt with `<bos>`** (Gemma 4's
tokenizer does not add it, and accuracy drops by about 3 points without it), prefill the reply with `Answer:`, and
read the next-token logits of `" A"`, `" B"`, ... Load the model with **`Gemma4ForConditionalGeneration`**; a plain
causal-LM class silently skips the language-model weights.

```python
import json
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "j-raghavan/s1-gemma4-26b-decision"
TEMPLATE = "<|turn>user\n{user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()

def decide(state, instructions, options):  # options: {key: description}, 2 to 26 of them
    keys = list(options)
    lines = "\n".join(f"{chr(65 + i)}) {k}: {options[k]}" for i, k in enumerate(keys))
    user = ("You are a decision model. Read the state and answer the question by choosing one option.\n\n"
            f"STATE:\n{json.dumps(state, ensure_ascii=False, indent=1)}\n\nQUESTION: {instructions}\n\n"
            f"OPTIONS:\n{lines}\n\nAnswer with the option letter only.")
    enc = tok("<bos>" + TEMPLATE.format(user=user) + "Answer:", return_tensors="pt",
              add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**enc, logits_to_keep=1).logits[0, -1]
    ids = [tok.encode(" " + chr(65 + i), add_special_tokens=False)[0] for i in range(len(keys))]
    return dict(zip(keys, torch.softmax(logits[ids].float(), -1).tolist()))

print(decide({"ticket": "I was charged twice for order 4471."}, "Which team should handle this ticket?",
             {"billing": "Payments and refunds", "shipping": "Deliveries", "tech": "Bugs and outages"}))
```

The complete version (yes/no and ordinal questions, option formatting identical to training) is
[examples/quickstart.py](https://github.com/j-raghavan/s1-decision-model/blob/main/examples/quickstart.py). For a
calibrated HTTP endpoint (`POST /v1/decisions`), see `api/server.py` in the repository. Serving with vLLM: request one
token with `logprobs` and `temperature 0` on the same prompt.

### Local, quantised (Ollama int4)

```bash
hf download j-raghavan/s1-gemma4-26b-decision --local-dir s1-gemma4-26b
printf 'FROM %s\n' "$PWD/s1-gemma4-26b" > Modelfile
ollama create s1-gemma4-26b --quantize int4 -f Modelfile     # about 17 GB on disk
```

Send raw prompts that start with `<bos>`: Ollama adds it for its library Gemma models but not for an imported one. The
repository's API does this for you (`S1_MODEL=s1-gemma4-26b uv run --extra api uvicorn api.server:app`), with
calibration fitted on the int4 build's own dev predictions (`api/calibration_s1.json`).

Measured cost of int4 against bf16 on the test sets: JevBench 0.801 vs 0.808 (the same answer on 92% of cases; paired
36 better, 48 worse, p = 0.23) and custom 0.966 vs 0.971 (p = 0.18); neither difference is significant. On a 32 GB
Apple-silicon laptop it takes about 0.8 s per decision (median); Ollama holds about 26 GB while it is loaded, so a
32 GB machine swaps under other load.

Hardware: about 52 GB in bf16 (one 80–96 GB GPU, or `device_map="auto"` across several GPUs). If the model does not
fit, `device_map="auto"` offloads layers and the forward pass fails unless you also pass `offload_folder` (slow); use a
quantised build instead.

## Training

- **Data**: 101,470 rows from permissively licensed public datasets (intent, NLI, topic, toxicity, passage QA,
  science and commonsense QA, math), 199 audited Super-NaturalInstructions decision tasks, rule-generated structured
  decisions, synthetic policy tasks generated by `openai/gpt-oss-120b` (kept only where Gemma 4 independently agrees),
  and prompt-injection sets (SPML, Lakera Gandalf, LLMail-Inject). Full card:
  [j-raghavan/s1-decision-data](https://huggingface.co/datasets/j-raghavan/s1-decision-data).
- **Targets**: the base model, as teacher, scores every row; its distribution is calibrated per task family and blended
  with the gold label (gold weight 0.5–0.9 depending on how reliable the label is). Injection and held-out rows use gold
  labels only.
- **Fine-tune**: LoRA rank 64, alpha 128, dropout 0.05 on the language model's attention and dense MLP (experts frozen);
  AdamW, learning rate 1e-4, 32 rows per step; loss = soft cross-entropy + Brier (+ ranked probability score for
  ordinal questions); options reshuffled per example. One run on an RTX PRO 6000 in bf16; snapshot at step 1500
  (about 48,000 rows seen) chosen on dev, then merged into the base weights.

Methodology and decision rules:
[docs/methodology.md](https://github.com/j-raghavan/s1-decision-model/blob/main/docs/methodology.md).

## Evaluation data

- **JevBench subset**: 1,700 cases from [JevBench](https://huggingface.co/datasets/Leanmcp/jevbench) (100 task families
  per text slice, fixed seed): safety, agent-trace risk, prompt injection, jailbreaks, banking intent, medical, science and general knowledge, sentiment.
  Never used for training; dev cases are deduplicated against test and training.
- **Custom structured decisions**: 1,200 rule-labelled cases (960 choice or yes/no, 240 ordinal) with JSON states, generated by the repository with no model
  involved.

## Limitations

- **Knowledge questions** account for most of the gap to Jev (MMLU-Pro and medical slices).
- **Ordinal sentiment**: on SST-5 the MAE (0.611) is worse than the untuned model's (0.522).
- **Prompt injection**: 8 of 53 test injections are missed. Do not use s1 as the only security control.
- **English only.**
- **Decisions, not explanations**: s1 gives probabilities, not reasons. For high-stakes decisions, keep a human or a
  stronger system in the loop, and use the probability to route uncertain cases.

## Licence and attribution

Apache-2.0, like the base model. This model is a modified version of
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it): fine-tuned with LoRA and merged. It is
not endorsed by Google. Training data keep their source licences (Apache-2.0, MIT, BSD, CC0, CC-BY and CC-BY-SA);
see the dataset card. s1 is an independent project, not affiliated with TypeSafe AI (Jev) or the JevBench authors.

```bibtex
@software{raghavan2026s1,
  author = {Raghavan, Jayasimha},
  title  = {s1: an open, calibrated System One decision model},
  year   = {2026},
  url    = {https://github.com/j-raghavan/s1-decision-model}
}
```
