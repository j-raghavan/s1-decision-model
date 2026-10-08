# Test fixtures

`predictions/<suite>/<run>.jsonl.gz`: saved model predictions reduced to the fields the decision rules read
(`case_id`, `slice`, `task_type`, `pred`, `gold`, `error`); no case text. They let `tests/test_decide.py` replay the
pilot verdicts and the snapshot choice behind the released model on every CI run.

- `custom`, `jevbench`: the untuned model (bf16, vLLM, without `<bos>`), pilot 1 and pilot 2, pilot 1 through
  transformers, and the untuned model through transformers.
- `jevbench_dev`, `custom_dev2`: the untuned model with `<bos>` as the training session scored it
  (`base-bos-vllm-2`) and every snapshot of the released training run
  (`bos-F-step-500` ... `bos-F-last`), as scored in that session.
