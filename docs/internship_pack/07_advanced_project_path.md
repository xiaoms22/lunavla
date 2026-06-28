# Advanced Project Path

Use this path after the baseline and ablation run successfully.

## Stronger Data

- Generate more episodes.
- Add noisier starts and harder goals.
- Save demonstrations as JSONL and reload them with `dataset.source: jsonl`.

## Stronger Policy

- Tune hidden dimension, learning rate, and chunk size.
- Compare action smoothness across runs.
- Track failure types before and after tuning.

## Stronger Evaluation

- Increase evaluation episode count.
- Save more rollout demos.
- Compare success rate with mean final distance instead of relying on one metric.

## Stronger Presentation

- Replace README assets with your own run.
- Add a short result table to your portfolio.
- Keep resume claims tied to exact configs and metrics.
