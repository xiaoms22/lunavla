# Data pipeline

```text
synthetic point-reach generator or JSONL -> VLARecord -> episode split
-> masked action chunks -> NumPy training -> checkpoint.json -> rollout evaluation
-> manifest and optional evidence snapshot
```

Each transition records the current `observation`, applied `action`, resulting `next_observation`, `terminated`, `success`, `episode_id`, contiguous `timestep`, optional instruction, and task context aligned with the current observation.

Train, validation, and test partitions are assigned by episode ID. JSONL validation rejects non-finite or dimensionally invalid values, duplicate transition keys, and discontinuous timesteps.

Supported sources are `mock_pusht` (a deprecated alias for generated data), `generated`, and `jsonl`. Neither source represents real PushT or robot data.

Useful commands:

```bash
python scripts/export_pusht_jsonl_dataset.py --config configs/act_pusht_jsonl_smoke.yaml
python scripts/validate_configs.py configs/act_pusht_jsonl_smoke.yaml
python scripts/run_controlled_experiments.py --suite data-quality
```

The controlled clean/noisy suite derives both datasets from the same base distribution and changes only the declared noise treatment. It does not compare the old example files when their generation settings differ.
