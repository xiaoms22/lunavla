# Model Card

## Model

MiniVLAPolicy is a tiny linear ACT-style action chunk predictor used for smoke tests and teaching. It predicts a short sequence of 2D actions from observation features plus optional language-instruction features.

## Intended Use

- Learn the VLA data-to-action loop.
- Verify training, evaluation, rollout logging, and demo generation.
- Produce internship-ready project evidence.

## Not Intended For

- Real robot deployment.
- Safety-critical control.
- General-purpose manipulation.
- State-of-the-art robotics claims.

## Inputs

- Observation: `[x, y, goal_x, goal_y]`.
- Optional language instruction: converted into small deterministic features.

## Outputs

- Action chunk: flattened sequence of 2D actions.

## Action Scale

Training runs save `action_statistics.json` beside the checkpoint. The stats record the demonstration action mean/std and explain the normalization boundary between train-time normalized actions and eval-time executable actions.

## Evaluation

Evaluation uses rollout success rate, mean final distance, rollout length, action smoothness, and failure cases. It is behavioral, not just loss-based.

## Limitations

The policy is deliberately small and NumPy-based. It is a teaching model, not a robotics foundation model.
