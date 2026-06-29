# Action Chunking With ACT

This tutorial is the static explanation behind `python scripts/generate_action_chunk_lesson.py`.

LunaVLA uses a tiny ACT-style policy because it gives beginners one clear VLA learning pattern:

```text
observation_t + instruction_features -> action_t:t+K
```

## The Core Idea

In next-action behavior cloning, the policy predicts one action:

```text
input_t -> action_t
```

In action chunking, the policy predicts a short sequence:

```text
input_t -> [action_t, action_t+1, action_t+2, ...]
```

The chunk is still trained with supervised learning. The difference is the shape of the target and the interpretation of the policy output.

## Where It Happens In Code

| concept | file | what to inspect |
| --- | --- | --- |
| record protocol | `dataset/vla_dataset.py` | `observation`, `action`, `episode_id`, `timestep`, `success`, `metadata` |
| mock PushT demonstrations | `dataset/pusht_dataset.py` | generated state, goal, expert action, and action chunk target |
| tiny ACT policy | `model/act_wrapper.py` | the ACT-style wrapper around the tiny policy |
| training loop | `trainer/train_act_pusht.py` | supervised loss over flattened action chunks |
| rollout eval | `eval_vla.py` | predicted actions are fed back into a closed-loop rollout |
| ablation | `configs/act_pusht_ablation_chunk_size.yaml` | one-variable chunk-size comparison |

## How The Dataset Builds A Chunk

For a sample at timestep `t`, LunaVLA takes future expert actions from the same episode:

```text
target_t = concat(action_t, action_t+1, ..., action_t+K-1)
```

If the sample is close to the end of the episode, the last action is repeated so every target has the same dimension. That keeps batching simple and makes the first implementation readable.

## Why Rollout Still Matters

Training loss measures how well the model fits demonstration labels. Rollout evaluation asks a harder question: if the model's own predicted action changes the next state, does the behavior still reach the goal?

That is why LunaVLA reports success rate, final distance, action smoothness, and failure cases instead of only reporting loss.

## What To Say In An Interview

LunaVLA implements a teaching-scale ACT-style imitation-learning loop. Demonstrations are converted into observation inputs and flattened future-action chunks. The policy is trained with supervised loss, then evaluated in rollout because closed-loop errors can compound. The chunk-size ablation shows how the action horizon changes behavior, while the repo stays honest that this is a small PushT-style teaching setup rather than a real-robot deployment claim.

## Run The Data-Backed Lesson

```bash
python scripts/run_baseline_evidence.py
python scripts/generate_action_chunk_lesson.py
```
