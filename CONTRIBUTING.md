# Contributing to LunaVLA

LunaVLA is a CPU-first teaching repository. Contributions should preserve the
small, reproducible imitation-learning loop and describe results within that
scope. It is not a real-robot or production VLA stack.

## Development setup

Use Python 3.10, 3.11, or 3.12 in an isolated environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Before opening a pull request, run:

```bash
ruff check .
mypy dataset model trainer eval_vla.py
pytest --cov --cov-report=term-missing
python scripts/validate_configs.py
python scripts/run_quickstart.py
```

## Change guidelines

- Add or update tests for behavior changes and negative paths.
- Keep seeds, data splits, configuration, and evaluation settings explicit.
- Do not commit checkpoints, complete run directories, credentials, private
  datasets, local absolute paths, or robot-specific secrets.
- Store only compact evidence snapshots in Git. Attach complete evidence packs
  and checksums to a release.
- Treat historical v1.0 reports as snapshots, not current evidence.
- Avoid causal or modality claims unless a controlled, multi-seed experiment
  supports them.

## Pull requests

Keep each pull request focused. Explain the teaching goal, compatibility impact,
tests run, and evidence affected. All required checks must pass, and at least one
maintainer review is required before merge.

By contributing, you agree that your contribution is licensed under Apache-2.0.
