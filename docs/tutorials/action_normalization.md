# Action Normalization In LunaVLA

Action learning is sensitive to scale. A policy that predicts `[0.01, -0.02]` and a policy that predicts `[1.0, -2.0]` may have the same direction, but they behave very differently in rollout.

LunaVLA keeps this concept small and inspectable:

```text
demonstration actions -> action statistics -> checkpoint metadata -> eval/report diagnostics
```

## What The Stats Mean

Run:

```bash
python scripts/generate_action_statistics.py
```

The command writes:

```text
outputs/action_statistics.json
outputs/action_statistics.md
outputs/act_pusht_baseline/action_statistics.json
```

The JSON records the demonstration action mean, standard deviation, min, max, percentiles, and clip fraction. For the mock PushT task, actions are 2D delta moves in the plane.

## Train-Time Vs Eval-Time Actions

Training systems often normalize actions:

```text
normalized_action = (action - mean) / std
```

At evaluation time, the model output has to become an executable action again:

```text
action = normalized_action * std + mean
```

LunaVLA currently stores stats for teaching and diagnostics while keeping the public baseline in original action units. That makes the rollout JSON easy to read: the action in a rollout frame is the action applied to the tiny PushT-style state update.

## Why This Matters

- Loss is affected by action scale.
- Clipping can hide bad scaling until rollout.
- A real robot dataset usually needs per-dataset action stats.
- Checkpoints should remember which stats were used.
- Reports should cite the stats source before comparing runs.

## Where To Look In Code

| concept | file |
| --- | --- |
| compute mean/std/min/max | `dataset/action_stats.py` |
| save stats during ACT training | `trainer/train_act_pusht.py` |
| save stats during BC training | `trainer/train_bc_pusht.py` |
| expose stats in eval summary | `eval_vla.py` |
| generate public stats report | `scripts/generate_action_statistics.py` |

## Interview-Safe Explanation

LunaVLA uses a tiny PushT-style action space, but it still records action statistics because the same idea appears in larger robotics datasets. The important distinction is that normalized actions are convenient for optimization, while executable actions are what the environment or robot actually receives.

## Boundary

These statistics describe a teaching-scale mock PushT dataset. They help explain scale, clipping, and checkpoint metadata. They do not calibrate a real robot controller.

