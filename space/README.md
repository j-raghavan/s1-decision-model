---
title: s1 decision model
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
models:
- j-raghavan/s1-gemma4-26b-decision
short_description: Typed decisions with calibrated probabilities, one pass
---

# s1 decision model demo

Try [s1-gemma4-26b-decision](https://huggingface.co/j-raghavan/s1-gemma4-26b-decision): give it a state (JSON or text)
and typed questions, get calibrated probabilities from one forward pass. Runs on ZeroGPU with the released bf16
weights. Source: [github.com/j-raghavan/s1-decision-model](https://github.com/j-raghavan/s1-decision-model) (`space/`).

API: `POST` through the Gradio client to the `/decide` endpoint with `state_text` and `questions_text` (JSON strings).
