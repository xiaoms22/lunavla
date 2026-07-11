## Summary

Describe the teaching or reproducibility problem and the change made.

## Verification

- [ ] `ruff check . && ruff check scripts/run_v2_release_profile.py`
- [ ] `mypy dataset model trainer lunavla scripts/run_v2_release_profile.py eval_vla.py`
- [ ] `pytest --cov --cov-report=term-missing`
- [ ] `python scripts/validate_configs.py`
- [ ] `python scripts/run_quickstart.py` (when the training/evaluation path changed)

## Evidence and compatibility

- [ ] I documented configuration/checkpoint compatibility changes.
- [ ] I kept claims within the CPU teaching-task boundary.
- [ ] I did not add credentials, private data, checkpoints, or local absolute paths.
- [ ] I updated compact evidence only when it is reproducible from a manifest.
