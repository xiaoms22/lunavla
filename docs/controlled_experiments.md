# Controlled experiment design

## Shared rules

All experiment families use five training seeds and the same twenty evaluation seeds per treatment. Dataset split, evaluation initial states, success threshold, goal, action clip, training budget, and optimization scale remain fixed unless they are the declared treatment. The runner refuses configurations with undeclared differences.

## Action-chunk sweep

Treatment: `chunk_size ∈ {1, 2, 4, 8}` for `numpy_linear_chunk` with `open_loop_chunk` execution. This isolates the behavior of chunk length within one policy family; it is not a BC-versus-ACT comparison.

## BC capacity sweep

Treatment: the declared MLP hidden dimension for `numpy_bc_mlp`. Chunk size remains one. This experiment is reported independently from the action-chunk sweep.

## Clean/noisy data

Treatment: a declared deterministic data-noise transform applied to the same base episodes and split. Policy, training, and evaluation settings remain fixed.

## Statistics

- Success: aggregate successes and trials per treatment and report the 95% Wilson interval.
- Continuous metrics: align results by training/evaluation seed, bootstrap paired differences with a fixed analysis seed, and report the 2.5%/97.5% quantiles.
- Publish raw per-episode rows, aggregates, invariant-check results, and the analysis seed.

The default runner command is:

```bash
python scripts/run_controlled_experiments.py --suite all --seeds 11 22 33 44 55 --eval-episodes 20
```

Use `--dry-run` to inspect generated configs and commands without training.
