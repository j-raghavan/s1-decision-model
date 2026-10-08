# Colab session cells

Each cell runs one GPU session end to end on Google Colab (`colab exec -s <session> -f <cell>`): it keeps the kernel
busy, logs progress to `/content/session.log`, packs its results, and holds the VM until the results are downloaded.
Every cell can be dry-run locally with a small model or `tests/mock_vllm.py` through its environment variables.

## The release path

| Cell | What it did for the released model |
| --- | --- |
| `session_bos_train.py` | relabelled the training rows with the teacher, trained the LoRA, scored every snapshot on dev, chose step 1500 with `s1.decide.choose_snapshot_ni`, scored it once on test |
| `session_relabel.py` | regenerated the soft targets published in the dataset, with the training session's exact code bundle |
| `session_package2.py` | merged the adapter into the base weights, checked the merge on 200 fixed cases, uploaded the weights |
| `session_release_check.py` | downloaded the published model and checked `examples/quickstart.py`, the model-card snippet and agreement with the saved test answers |
| `watch_session.sh` | follows a running session's log, with a heartbeat, until it ends |

## Archive

`archive/` holds the cells of earlier, superseded experiments (encoder training, teacher evaluation, decoder pilots,
the first full runs, the first packaging attempt). They are kept unchanged as the record of what ran, not maintained:
several use an older snapshot rule (`s1.decide.choose_snapshot`) that the release replaced with the non-inferiority
rule, and none of them produced the released model.
