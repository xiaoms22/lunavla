# Interview Q&A

## What is VLA in this project?

VLA means Vision-Language-Action. In this tiny project, the interface is VLA-shaped: observations describe the environment state, an optional text instruction describes task intent, and actions are evaluated through rollout behavior.

## What is behavior cloning?

Behavior cloning trains a policy to imitate expert demonstrations. MiniMind-VLA generates simple PushT-style demonstrations and trains a policy to predict the expert action chunk from the current observation.

## Why use action chunks?

ACT predicts a short sequence of actions instead of only one action. This is useful because actions are temporally correlated, and chunk size becomes an interpretable ablation variable.

## Why is rollout evaluation important?

Low training loss can still fail when predictions are fed back into the environment. Rollout evaluation checks whether repeated predicted actions actually move the object toward the goal.

## What metrics do you report?

The main metrics are success rate, mean final distance, mean rollout length, action smoothness, final training loss, and failure-case count.

## What can you improve after the baseline?

Good extensions include better observation features, stronger policy capacity, cleaner demonstrations, chunk-size tuning, richer failure labels, and a more polished project report.
