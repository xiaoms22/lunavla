# Advanced Project Path

Use this path after the baseline, ablation, policy ladder, and report pack run successfully.

## Stronger Data

- Generate more episodes.
- Add noisier starts and harder goals.
- Save demonstrations as JSONL and reload them with `dataset.source: jsonl`.
- Compare clean demonstrations against noisy demonstrations with the same eval setting.
- Start with `python scripts/run_jsonl_data_smoke.py` to verify the local-file data path before designing a custom dataset.
- Then run `python scripts/run_data_quality_comparison.py` to produce a clean-vs-noisy report.

## Stronger Policy

- Tune hidden dimension, learning rate, and chunk size.
- Start with `python scripts/run_policy_tuning_comparison.py` for a BC hidden-size comparison with fixed data and eval settings.
- Compare tiny linear, BC, and ACT-style policies using the same summary fields.
- Track final loss, success rate, mean final distance, rollout length, and action smoothness together.
- Explain policy changes through rollout behavior, not only a metric table.

## Stronger Task Understanding

- Inspect phase/subtask labels in rollout traces.
- Count which subtask failed most often.
- Start with `python scripts/generate_task_understanding_report.py` after saved rollout JSON exists.
- Add one new first-pass failure label only after reading saved rollout JSON.
- Keep the Task Layer rule-based unless you have a verified no-API fallback.

## Stronger Action Analysis

- Generate action statistics for each run.
- Check action mean, standard deviation, min/max, and clipped fraction.
- Start with `python scripts/generate_action_analysis_report.py` to compare train-time targets with eval-time executable actions.
- Explain how train-time action targets relate to eval-time executable actions.
- Add an action-normalization ablation only after the baseline report is stable.

## Stronger Evaluation

- Increase evaluation episode count.
- Save more rollout demos.
- Start with `python scripts/run_extended_evaluation.py` after checkpoints exist.
- Compare success rate with mean final distance instead of relying on one metric.
- Inspect at least one success and one failure before writing a conclusion.

## Stronger Presentation

- Replace README assets with your own checked run.
- Add a short result table to your portfolio.
- Include the project card, experiment ledger, and submission pack.
- Keep resume claims tied to exact configs and metrics.

## Safe Extension Boundary

Do not publicly present unfinished adapters, LLM planners, world models, or real-robot claims as completed work. Add heavier VLA components only after they have a runnable command, generated report, and honest boundary statement.
