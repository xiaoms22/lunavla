# Ablation Report Template

## Question

What variable changed, and why?

Start with an auto-generated comparison:

```bash
python scripts/run_ablation_evidence.py
```

This runs the baseline if needed, runs the chunk-size ablation, generates per-run project reports, and writes `outputs/run_comparison.md`. Then inspect rollout behavior and add your own explanation.

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
