# v1.1 to v2 compatibility

The signed v1.1 release remains the stable NumPy/CPU teaching route. The v2 branch does not rewrite v1.1 history, tags, manifests, or result snapshots.

## Configuration migration

Migrate a resolved schema-v1 config without overwriting an existing file:

```bash
python scripts/migrate_v11_to_v2.py \
  configs/act_pusht_cpu_smoke.yaml \
  outputs/migrated/v2.yaml
```

Add `--overwrite` only when replacement is intentional. The migration preserves the policy type, dimensions, chunk size, dataset source/split, seeds, training budget, execution mode, output location, and legacy report path. The output is validated immediately against schema version 2. Unknown top-level or section fields remain errors.

## Policy and result compatibility

- v1.1 NumPy policies run through `NumpyPolicyAdapter` and the v2 registry.
- v1.1 JSON checkpoints remain readable through the NumPy loader; v2 never interprets a JSON file as a PyTorch pickle.
- Existing `results/v1.1/` manifests and the README renderer remain unchanged and readable.
- The frozen v1.1 requirements and quickstart remain in `requirements.txt` and `scripts/run_quickstart.py`.
- v2 PyTorch checkpoints use their own versioned format and do not replace v1.1 checkpoints.

## Dependency profiles

| Route | Python | NumPy | Heavy dependencies |
| --- | --- | --- | --- |
| signed v1.1 tag | 3.10–3.12 | 1.24–1.x | none |
| v2 base/dev | 3.12 | 2.0–2.2 | none |
| v2-core extra | 3.12 | 2.0–2.2 | PyTorch 2.11, torchvision 0.26 |
| v2 extra | 3.12 | 2.0–2.2 | v2-core plus LeRobot 0.6 dataset profile |

The cross-platform resolution is committed in `uv.lock`. Linux CPU CI additionally uses generated, hash-locked `requirements-v2-*-cpu.lock` profiles and rejects CUDA-only packages. GPU full training is a separate experimental manual workflow and does not block dependency-light or CPU checks.
