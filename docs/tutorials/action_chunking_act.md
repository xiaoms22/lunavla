# Action chunking with the NumPy linear policy

This file keeps its historical filename so old links continue to work. LunaVLA v1.1 teaches action-chunk target construction; it does not implement the ACT Transformer.

At timestep `t`, next-action behavior cloning predicts one action, while chunk prediction produces a fixed-shape sequence:

```text
observation_t + instruction_features -> actions[t:t+K], valid_mask[t:t+K]
```

Chunks never cross an episode boundary. Near the end of an episode the target is padded for batching, and `valid_mask` excludes every padded position from MSE.

## Execution modes

- `receding_horizon`: replan each step and execute the first valid action.
- `open_loop_chunk`: execute all valid actions in a predicted chunk before replanning.

These behaviors are not interchangeable: changing a later chunk action can change an open-loop trajectory while leaving the first receding-horizon step unchanged.

## Where to inspect

| concept | location |
| --- | --- |
| versioned transition record | `dataset/vla_dataset.py` |
| episode-safe chunk and mask construction | `dataset/pusht_dataset.py` |
| linear chunk predictor | `model/minivla_policy.py` |
| masked MSE training | `trainer/train_core.py` |
| both execution modes | `eval_vla.py` |
| controlled `1/2/4/8` sweep | `scripts/run_controlled_experiments.py` |

Training loss measures fit to demonstrations. Rollout success, final distance, and smoothness measure behavior after predictions affect later states. A chunking claim requires the controlled multi-seed sweep and paired intervals described in `docs/controlled_experiments.md`.
