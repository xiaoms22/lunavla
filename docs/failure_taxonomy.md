# Failure Taxonomy

Use this document to label failed rollouts before writing a report or resume bullet.

| category | meaning | what to inspect |
| --- | --- | --- |
| `did_not_reach_goal` | final distance is above threshold | action magnitude, rollout horizon |
| `oscillation` | policy moves around the goal without settling | action chunk smoothness |
| `wrong_direction` | early actions increase distance | observation/action alignment |
| `stuck` | actions become too small too early | loss, clipping, training data coverage |
| `action_clipping` | predicted actions often hit the eval clip limit | action scaling and target range |
| `out_of_distribution_start` | initial state is unlike training records | dataset sampling range |

Each failure case should include:

- episode id;
- final distance;
- initial distance and minimum distance;
- failing subtask or phase;
- short label;
- one sentence explanation;
- next minimal fix.

`eval_vla.py` writes automatic labels for common rollout failures. Treat them as a first-pass teaching aid: inspect the saved rollout JSON or web demo before making a final project conclusion.
