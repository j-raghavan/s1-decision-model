# Methodology

How s1 was built and measured: the task format, the training data and targets, the fine-tune, how the released
snapshot was chosen, and how it was evaluated.

## 1. The task

A request is a `state` (text or JSON) and one or more typed questions:

| Type | Criteria | Answer |
| --- | --- | --- |
| `choice` | keyed options with descriptions | the chosen key, with a probability for every key |
| `noul` | optional yes/no descriptions | the probability of yes |
| `score` | an ordered list of levels | the expected level, with a probability for every level |

s1 answers each question in **one forward pass**. The question is rendered with lettered options into a chat prompt
that ends with the model turn prefilled as `Answer:`; the next-token log-probabilities of the option letters
(A, B, C, ...) give the distribution over options. Nothing is generated. Questions with more than 26 options take
two passes. The prompt must start with `<bos>`: Gemma 4's tokenizer does not add it, and without it accuracy drops
by about 3 points (0.742 against 0.770 for the untuned model on JevBench). Code: `s1/decoder.py`, `eval/jevbench/run_ollama.py`.

## 2. Base model

`google/gemma-4-26B-A4B-it`: a mixture-of-experts decoder with 128 experts and about 4B active parameters per token,
Apache-2.0. It was chosen after measuring candidate teachers and bases zero-shot on the dev splits
(results in `results/`). An earlier approach, a 396M ModernBERT-large encoder trained on the same kind of data,
plateaued at about 0.49 mean accuracy on JevBench, far below what the decoder reaches untuned, and was dropped.

## 3. Training data

101,470 rows, every one from a permissively licensed public dataset, a rule generator in this repo, or a synthetic
generator. Full source list, row counts and licences: [DATA_CARD.md](../DATA_CARD.md) and [LINEAGE.md](../LINEAGE.md).

- **Public decision datasets** (intent, NLI, topic, toxicity, passage QA, science and commonsense QA, math), converted
  to typed questions by `pipeline/sources.py`.
- **199 Super-NaturalInstructions decision tasks** (`pipeline/sni.py`), audited task by task: English, 2–20 fixed
  answers, permissive instance licence, no source shared with an eval set (`pipeline/sni_manifest.json`).
- **Rule-generated structured decisions** (`pipeline/structured.py`): JSON states such as invoices, deploy gates,
  SLA checks and payload validation, whose gold answer is exact by construction.
- **Synthetic task families** (`pipeline/generate_synthetic.py`): task families written by `openai/gpt-oss-120b` from
  seed prompts; a generated answer is kept only where Gemma 4 26B-A4B independently agrees (265 rows dropped).
- **Prompt-injection detection** (`pipeline/injection.py`): attack prompts from SPML, Lakera Gandalf and the LLMail-Inject
  challenge, with benign intent-dataset questions as negatives; 9.6% of the mix.

The rule-generated and synthetic rows (30,000 together, 30% of the mix) are policy-reasoning questions: apply a
written policy or rubric to a structured state.

Whole task families are held out (`holdout_family`), so validation measures generalisation to instructions the
model has never seen. The dev splits are deduplicated against the test sets and the training rows
(`eval/jevbench/build_dev.py`); an exact-text check of every training row against every JevBench, dev and custom
test case found no overlap.

## 4. Soft targets

The teacher is the base model itself, Gemma 4 26B-A4B, served in bf16 through vLLM with `<bos>`. For every core row
it gives a distribution over the options, which is calibrated per task family against the gold answers (Platt
scaling for yes/no, a temperature for choice and score, 2-fold cross-fitted) and blended with the gold label:

    target = gold_weight * one_hot(gold) + (1 - gold_weight) * calibrated_teacher

`gold_weight` follows how reliable the gold label is: 0.9 for rule-generated rows, 0.7 for objective answer keys and
consensus-filtered synthetic rows, 0.6 for Super-NaturalInstructions, 0.5 for subjective or crowd-noisy labels
(`pipeline/targets.py`). Injection and held-out rows use gold targets only. The teacher can say how plausible the
wrong options are, but cannot outvote a known-correct answer.

## 5. Fine-tune

| Setting | Value |
| --- | --- |
| Method | LoRA on the language model's attention and dense MLP projections (experts frozen), then merged |
| Rank / alpha / dropout | 64 / 128 / 0.05 |
| Optimiser | AdamW, learning rate 1e-4, no weight decay; linear warm-up over max(10 steps, 5%), then linear decay |
| Batch | 32 rows per step (16 x 2 accumulation); rows over 2,048 tokens dropped; batches split into pieces of at most 24,576 tokens with the loss weighted so the gradient is unchanged |
| Loss | soft-target cross-entropy + 0.5 x Brier score, plus a ranked probability score term for ordinal questions (all proper scoring rules) |
| Order invariance | options reshuffled per example (except ordinal scales), so the model cannot learn positional preferences |
| Run | one session on an RTX PRO 6000 (96 GB), bf16, about 140 minutes, snapshots every 500 steps; the whole session (relabelling, training, dev and test scoring) used 25.92 Google Colab compute units |
| Released snapshot | step 1500 (about 48,000 rows seen) |

Code: `s1/train_decoder.py`, `colab/session_bos_train.py`.

## 6. Choosing the snapshot (pre-registered)

The decision rule was written and tested against earlier runs before training started (`s1/decide.py`,
`tests/test_decide.py`) and applied mechanically:

1. Score the untuned model and every snapshot on both dev splits (`jevbench_dev`, 765 cases; `custom_dev2`, 2,341 cases).
2. A snapshot is eligible if it covers at least 98% of every slice and is non-inferior to the untuned model on
   custom dev (accuracy at most 1.5 points lower; paired McNemar tests reported).
3. Pick the eligible snapshot with the best JevBench-dev accuracy, then score it **once** on the test sets.

Step 1500 won (JevBench dev 0.834, custom dev +53/−1 against the untuned model). Results:
`results/bostrain_session/decisions.json`.

## 7. Evaluation

- **JevBench test subset**: 1,700 cases, 100 task families per text slice, seed 20261004
  (`eval/jevbench/build_subset.py`, ids in `results/jevbench/subset_case_ids.txt`). Headline is mean accuracy over
  slices; score-type slices (SST-5) report MAE of the expected level and are excluded from the accuracy mean. ECE uses
  10 equal-width bins on the top probability. Differences under about ±1.5 points are within noise at this size.
  Jev 1.13, Laya and Von are scored from the predictions published with JevBench or from their public weights.
- **Custom structured-decision test set**: 1,200 rule-labelled cases, 960 choice or yes/no and 240 ordinal (JSON states and presentation decisions), generated
  by `eval/custom/generate.py` with no model involved.
- **Calibration** is fitted on the dev splits only and applied to test (`eval/jevbench/calibrate_from_dev.py`); the
  deployed API uses per-type calibrators fitted the same way.
- **Paired comparisons** use exact McNemar tests on the same cases.

## 8. Results

| System | JevBench raw / calibrated | Custom raw / calibrated | Injections caught / false alarms |
| --- | --- | --- | --- |
| **s1** (released, bf16) | **0.808 / 0.817** | **0.971 / 0.981** | **45/53, 0** |
| Gemma 4 26B-A4B, untuned (with `<bos>`) | 0.770 / 0.799 | 0.948 / 0.960 | 35/53, 0 |
| s1, int4 through Ollama | 0.801 / 0.804 | 0.966 / 0.981 | 46/53, 0 |
| Jev 1.13 (closed) | 0.835 / 0.856 | — | — |

Paired against the untuned model on JevBench test, s1 fixes 103 cases and breaks 58 (p = 0.0005); on the custom test
set it fixes 24 and breaks 2. Per-slice tables: `results/jevbench/summary.md`, `results/custom/summary.md`, and the
[leaderboard](https://j-raghavan.github.io/s1-decision-model/leaderboard.html).

Latency per decision on one RTX PRO 6000 through vLLM: median 28 ms (p95 95 ms) with FP8 weights, 40 ms (133 ms) in
bf16.

## 9. Limitations

- **Knowledge questions.** Most of the remaining gap to Jev is on knowledge-heavy slices (MMLU-Pro, medical
  questions). Gold-labelled knowledge rows did not move MMLU-Pro; a reasoning teacher helped modestly on dev.
- **Ordinal sentiment.** On SST-5, s1's MAE (0.611) is worse than the untuned model's (0.522).
- **Prompt injection.** 8 of 53 test injections are missed.
- **Quantisation.** The int4 Ollama build scores 0.801 on JevBench and 0.966 on custom against bf16's 0.808 and 0.971;
  neither difference is significant (paired p = 0.23 and 0.18).
- **Size.** 26B total parameters (about 52 GB in bf16): fast on a data-centre GPU, slow on a laptop.
- **English only.** Training and evaluation data are English.
