# Ablation Report Template

## Question

What variable changed, and why?

Start with an auto-generated comparison:

```bash
python scripts/compare_runs.py --runs outputs/act_pusht_baseline outputs/act_pusht_ablation_chunk_size --out outputs/run_comparison.md
```

Then inspect rollout behavior and add your own explanation.

## Runs

| run | config | changed variable |
| --- | --- | --- |
| baseline | | |
| ablation | | |

## Comparison

| metric | baseline | ablation | interpretation |
| --- | --- | --- | --- |
| final loss | | | |
| success rate | | | |
| mean final distance | | | |
| failure cases | | | |

## Qualitative Rollout Notes

Describe action smoothness, wrong direction, stuck behavior, or oscillation.

## Resume-Safe Claim

Write one claim that is fully supported by the results.
