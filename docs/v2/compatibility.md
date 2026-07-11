# v1.1 to v2 compatibility

The signed v1.1 release remains the stable NumPy/CPU teaching route. The v2 branch does not rewrite v1.1 history, tags, manifests, or result snapshots. `v2.0.0-rc.1` freezes the migration and serialization behavior described here; see the complete [`contract_freeze.md`](contract_freeze.md) matrix.

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
- v2 PyTorch checkpoint schema 3 uses its own versioned format and does not replace v1.1 checkpoints.
- Transformer schema 2 state-only checkpoints are read-only inputs and are upgraded to schema 3 when re-saved. Schema 2 visual checkpoints are rejected because they predate the frozen CoordConv input shape.
- RunManifest schema 2 remains a read-only input. Current runs write schema 3.
- Unknown fields, boolean schema versions, non-finite values, and incompatible policy formats fail rather than falling back.

## Dependency profiles

| Route | Python | NumPy | Heavy dependencies |
| --- | --- | --- | --- |
| signed v1.1 tag | 3.10–3.12 | 1.24–1.x | none |
| v2 base/dev | 3.12 | 2.0–2.2 | none |
| v2-core extra | 3.12 | 2.0–2.2 | PyTorch 2.11, torchvision 0.26 |
| v2 extra | 3.12 | 2.0–2.2 | v2-core plus LeRobot 0.6 dataset profile |
| v2-integration extra | 3.12 | 2.0–2.2 | v2 plus Gym PushT and Pymunk 6.11.1 |

The cross-platform resolution is committed in `uv.lock`. Linux CPU CI additionally uses generated, hash-locked `requirements-v2-*-cpu.lock` profiles and rejects CUDA-only packages. GPU full training is a separate experimental manual workflow and does not block dependency-light or CPU checks.
