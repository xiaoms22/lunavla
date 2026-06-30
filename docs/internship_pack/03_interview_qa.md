# Interview Q&A

## Is LunaVLA really VLA?

LunaVLA is VLA-shaped, but the current public core is more accurately an IL/VA starter. It teaches the observation-to-action and rollout-evaluation loop first. The language field is present as task context, but the public baseline does not claim to be an OpenVLA/openpi-style language-conditioned robot foundation model.

## Why keep the name LunaVLA?

The long-term learning path points toward VLA systems, but the repo starts with the smallest teachable core: demonstrations, action chunks, behavior cloning, ACT-style policies, rollout metrics, and reports. This makes the project useful before adding heavier language or world-model components.

## What is behavior cloning?

Behavior cloning is supervised imitation learning. A demonstration provides an observation and expert action; the model learns to predict the action from the observation.

## Why can low training loss still fail?

Training loss measures one-step or chunk-level fit to demonstrations. Rollout evaluation feeds the policy's own predictions back into the environment, so small errors can accumulate and push the state away from the goal.

## Why use action chunks?

ACT predicts a short sequence of actions instead of only the next action. In LunaVLA, `chunk_size` is an interpretable variable: changing it can affect final loss, success rate, smoothness, and failure modes.

## What is the Task Layer?

The Task Layer adds structured context such as phase, subtask, and instruction metadata. It helps explain where a rollout failed, for example `approach_block`, `align_push`, `push_to_goal`, or `settle`, without requiring an LLM dependency.

## How do you analyze failed subtasks?

Run `python scripts/generate_task_understanding_report.py` after saving rollout JSON. It reads frame-level phase/subtask labels, counts which final phase failed most often, and adds a rule-based `phase_regression` label when a failed episode gets closer or reaches a later phase before drifting away.

## What are action statistics?

Action statistics summarize the scale and range of demonstration actions. Mean, standard deviation, min/max, and clipping information help explain why action normalization matters and why train-time targets differ from executable rollout actions.

## How do you connect train-time and eval-time actions?

Run `python scripts/generate_action_analysis_report.py`. It compares demonstration action targets from training records with executable rollout actions after eval clipping, so you can explain whether a policy is saturating, producing tiny actions, or staying within the teaching-scale action range.

## What does the clean vs noisy JSONL comparison show?

It keeps the policy, chunk size, training steps, and evaluation setting fixed, then changes the local JSONL demonstrations. This makes the result easier to explain as a data-quality experiment: noisier actions or harder starts can increase loss, reduce rollout success, and create more failure cases.

## What metrics do you report?

The main metrics are final training loss, success rate, mean final distance, mean rollout length, mean action smoothness, failure-case count, failure categories, failure subtasks, and action statistics.

## Why run extended evaluation?

Run `python scripts/run_extended_evaluation.py` after checkpoints exist. It increases the number of rollout episodes, saves more rollout JSON demos, and forces the conclusion to cite both success rate and mean final distance plus at least one success or failure example.

## How do you keep homepage claims honest?

Run `python scripts/generate_homepage_summary.py` after extended evaluation. It produces `outputs/homepage_summary.md` and `images/homepage_results.svg`, so the README result card, portfolio table, and interview claims point back to generated metrics instead of hand-written numbers.

## How do you explain BC vs ACT?

BC is the simplest next-action imitation baseline. ACT predicts action chunks, which can better represent temporally correlated behavior. LunaVLA compares them as a learning ladder, not as a claim of state-of-the-art performance.

## What does the BC hidden-size comparison teach?

It changes only the MLP hidden dimension while keeping the PushT-style data and eval path fixed. The point is to show that a larger supervised policy should still be judged by rollout behavior, final distance, smoothness, and failure cases rather than by training loss alone.

## What would you improve next?

Good extensions include better demonstrations, harder starts, stronger observation features, BC/ACT tuning, action normalization experiments, richer failure labels, local JSONL data loading, and clearer report assets.

## What should you not claim?

Do not claim real-robot deployment, foundation-model reproduction, OpenVLA/openpi/pi0 equivalence, or production robotics performance. Tie every claim to a command, config, metric, and artifact.
