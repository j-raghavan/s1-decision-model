# s1: an open, calibrated System One decision model

[![CI](https://github.com/j-raghavan/s1-decision-model/actions/workflows/ci.yml/badge.svg)](https://github.com/j-raghavan/s1-decision-model/actions/workflows/ci.yml)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20model-s1--gemma4--26b--decision-yellow)](https://huggingface.co/j-raghavan/s1-gemma4-26b-decision)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-s1--decision--data-yellow)](https://huggingface.co/datasets/j-raghavan/s1-decision-data)
[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)

**s1** makes fast typed decisions over text or JSON: pick one of several options (`choice`), answer yes or no
(`noul`), or place something on an ordered scale (`score`). Every answer comes with a calibrated probability, from
**one forward pass**, with no text generated. It is Gemma 4 26B-A4B fine-tuned on about 100,000 decision rows built only
from permissively licensed sources, and everything is open: the weights, the training data, the code and the evaluation.

[Website and leaderboard](https://j-raghavan.github.io/s1-decision-model) ·
[Model card](MODEL_CARD.md) · [Data card](DATA_CARD.md) · [Methodology](docs/methodology.md) ·
[Lineage](LINEAGE.md)

## Results

Held-out test sets; the released snapshot was picked on separate dev splits by a rule fixed before training.

| System | JevBench (1,700 cases) raw / calibrated | Custom structured decisions raw / calibrated | Open weights |
| --- | --- | --- | --- |
| **s1** (this model, bf16) | **0.808 / 0.817** | **0.971 / 0.981** | yes |
| s1, int4 through Ollama | 0.801 / 0.804 | 0.966 / 0.981 | yes |
| Jev 1.13 (closed) | 0.835 / 0.856 | — | no |
| Gemma 4 26B-A4B, untuned | 0.770 / 0.799 | 0.948 / 0.960 | yes |
| Gemma 4 31B, untuned | 0.756 | 0.978 | yes |
| gpt-oss-120b, untuned | 0.670 | 0.853 | yes |
| Laya 421M | 0.485 | — | yes |
| Von 1.3 | 0.465 | — | yes |

- On the same JevBench cases, s1 turns 103 of the untuned model's wrong answers right and 58 right answers wrong
  (paired McNemar p = 0.0005). It catches 45 of 53 prompt injections with no false alarms.
- Raw probabilities are already well calibrated: mean ECE 0.083 on JevBench (Jev: 0.081). "Calibrated" columns use
  per-task calibrators fitted on the dev splits only ([results/calibration_from_dev.md](results/calibration_from_dev.md)).
- Differences under about ±1.5 points are within noise on the 1,700-case subset. Per-slice tables:
  [leaderboard](https://j-raghavan.github.io/s1-decision-model/leaderboard.html), `results/`.
- Latency per decision on one RTX PRO 6000 (vLLM): 28 ms median with FP8 weights, 40 ms in bf16. The int4 build runs
  on a 32 GB Apple-silicon laptop at about 0.8 s per decision; its accuracy cost against bf16 is not significant.

s1 is an independent project, not affiliated with TypeSafe AI (Jev) or the JevBench authors. Jev's numbers come from
the predictions published with JevBench.

## Use it

**Python (transformers).** A complete, tested example is in [examples/quickstart.py](examples/quickstart.py):

```python
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

tok = AutoTokenizer.from_pretrained("j-raghavan/s1-gemma4-26b-decision")
model = Gemma4ForConditionalGeneration.from_pretrained(
    "j-raghavan/s1-gemma4-26b-decision", dtype=torch.bfloat16, device_map="auto")
# Render the question with lettered options, start the prompt with <bos>, prefill "Answer:",
# and read the next-token logits of " A", " B", ... (see examples/quickstart.py).
```

**HTTP API.** `api/server.py` serves `POST /v1/decisions` with per-type calibration, backed by a local Ollama:

```bash
uv sync --extra api
uv run --extra api uvicorn api.server:app --port 8000
curl -s localhost:8000/v1/decisions -H 'content-type: application/json' -d '{
  "state": {"ticket": "I was charged twice for order 4471."},
  "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
    "criteria": {"billing": "Payments and refunds", "shipping": "Deliveries", "tech": "Bugs"}}}}'
```

Two things matter for correct results: prompts must start with `<bos>` (Gemma 4's tokenizer does not add it), and the
model must be loaded with `Gemma4ForConditionalGeneration` (a plain causal-LM class silently skips the weights).

## How it was built

1. **Data**: public decision datasets, 199 audited Super-NaturalInstructions tasks, rule-generated structured
   decisions, synthetic policy tasks and prompt-injection sets, all with permissive licences
   ([DATA_CARD.md](DATA_CARD.md)).
2. **Targets**: the base model, as teacher, scores every row; its calibrated distribution is blended with the gold
   label, weighted by how reliable the gold label is.
3. **Fine-tune**: LoRA (rank 64) on attention and dense MLP, options reshuffled per example, a proper-scoring-rule
   loss; merged into the base weights.
4. **Selection and evaluation**: a pre-registered rule on dev splits, then one scoring run on the test sets.

Full details: [docs/methodology.md](docs/methodology.md).

## Provenance

Every model and dataset that feeds the weights passes a CI gate ([scripts/check_lineage.py](scripts/check_lineage.py)):
models must come from an allowlisted organisation, checked up the whole `base_model` chain on the Hugging Face Hub,
and every training source must carry a licence that allows redistribution. Evaluation sets never train.
See [LINEAGE.md](LINEAGE.md).

## Repository

| Path | Contents |
| --- | --- |
| `s1/` | the decoder decision model (`decoder.py`), its trainer (`train_decoder.py`), the snapshot decision rules (`decide.py`), and the earlier encoder |
| `pipeline/` | training data: source conversion, Super-NaturalInstructions audit, rule and synthetic generators, teacher labelling, soft targets, mixing |
| `eval/` | JevBench subset and custom test set, runners (vLLM, Ollama, transformers, hosted APIs), scoring and calibration |
| `api/` | the `/v1/decisions` server and its calibration |
| `colab/` | the GPU session cells that trained and evaluated the model, with their watcher |
| `scripts/` | provenance gate, LINEAGE renderer, LoRA merge, site builder |
| `results/` | scores for every evaluated system, and the logs of the sessions behind the release |
| `site/` | the GitHub Pages website |

## Development

```bash
uv sync --extra train --extra api --group dev
uv run pytest -q tests/                                    # unit and property tests
uv run ruff check .
uv run --no-project python scripts/check_lineage.py        # provenance and licence gate
uv run --no-project python scripts/build_site.py           # website into _site/
```

## Citation

```bibtex
@software{raghavan2026s1,
  author = {Raghavan, Jayasimha},
  title  = {s1: an open, calibrated System One decision model},
  year   = {2026},
  url    = {https://github.com/j-raghavan/s1-decision-model}
}
```

## Licence

Code, weights and documentation: Apache-2.0. The model is a fine-tune of
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (Apache-2.0). Training rows keep their
source licences, listed per row and in [DATA_CARD.md](DATA_CARD.md).
