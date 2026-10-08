# Contributing

Issues and pull requests are welcome.

## Setup and checks

```bash
uv sync --extra train --extra api --group dev
uv run pytest -q tests/                                  # CI runs the same suite on CPU
uv run ruff check .
uv run --no-project python scripts/check_lineage.py      # provenance and licence gate
uv run --no-project python scripts/render_lineage.py     # after editing lineage.yaml
```

Every model or dataset that feeds training must be added to `lineage.yaml` and pass the provenance gate. Changes to
the snapshot rule (`s1/decide.py`) must keep `tests/test_decide.py` green: it replays the decision behind the released
model from committed predictions (`tests/fixtures/`).

## Known structural debt

- **Imports by path.** `eval/jevbench/` and `scripts/` are directories of scripts, not packages, and the project is
  not installed as a package (`[tool.uv] package = false`). `s1/decoder.py`, `api/server.py` and several pipeline
  modules add `eval/jevbench` to `sys.path` at import time to reach the shared prompt renderer and calibration code,
  and the Colab session bundles rely on this layout. Moving the shared pieces into the `s1` package is the planned fix.
- **Deliberate copies, pinned by tests.** `examples/quickstart.py` copies the prompt and option layout so it runs
  without the repository (`tests/test_quickstart.py` keeps it identical to the training renderer), and the earlier
  encoder keeps its own loss (`tests/test_losses.py` keeps it equal to the decoder's).
- **Archived session cells** (`colab/archive/`) are a record of earlier experiments and are not maintained.
