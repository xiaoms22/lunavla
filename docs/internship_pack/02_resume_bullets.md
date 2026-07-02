# Resume Bullets

Use only the strongest bullet that matches work you actually completed. Replace bracketed values with metrics from your generated reports.

## CPU Smoke Completed

> Verified LunaVLA, a lightweight IL/VA learning scaffold, with a one-command CPU smoke test covering training, rollout evaluation, summary generation, claim-safety diagnostics, and rollout browser export.

## Dataset And Task Layer Completed

> Built and inspected a PushT-style demonstration pipeline with structured observation/action records, optional task instruction fields, phase/subtask labels, and rollout metadata for failure analysis.

## Task Understanding Completed

> Analyzed saved rollout traces by phase and subtask, counted where failures ended, and added a rule-based first-pass `phase_regression` label for episodes that got closer before drifting away.

## BC Baseline Completed

> Implemented a from-scratch behavior-cloning baseline for observation-to-action prediction, including config-driven training, checkpoint export, rollout evaluation, and a generated project report.

## ACT Baseline Completed

> Implemented an ACT-style imitation-learning baseline on a PushT-style task, including action-chunk prediction, checkpoint export, rollout success evaluation, action smoothness metrics, and failure-case logging.

## Ablation Completed

> Ran a chunk-size ablation for an ACT-style action policy and compared final loss, success rate, mean final distance, rollout length, action smoothness, and failure labels to explain policy behavior.

## Policy Ladder Completed

> Compared tiny linear, BC, and ACT-style policies in a small policy ladder, using shared train/eval/report artifacts to explain how policy capacity and action chunks affect rollout behavior.

## Policy Tuning Comparison Completed

> Ran a controlled BC hidden-size comparison on the same PushT-style demonstration loop, using rollout success, final distance, action smoothness, and failure cases to explain why tuning should be judged with behavior evidence, not loss alone.

## Action Statistics Completed

> Added action-statistics reporting for a tiny IL/VA training loop, connecting action mean/std/min/max, clipping behavior, normalization notes, and rollout diagnostics to evidence-backed project claims.

## Action Analysis Completed

> Compared train-time demonstration action targets with eval-time executable rollout actions across runs, using clipped fraction, max action magnitude, and rollout metrics to explain action-scale risks before policy tuning.

## Extended Evaluation Completed

> Reran rollout evaluation with more episodes, saved additional rollout traces, and compared success rate with mean final distance plus success/failure examples before writing evaluation conclusions.

## Presentation Pack Completed

> Built a README-facing result card and submission pack that tie homepage metrics to generated reports, experiment ledgers, rollout evidence, and claim-safety boundaries.

## Reviewer Readiness Completed

> Added a reviewer-readiness checklist that verifies public commands, generated artifacts, boundary statements, and claim-safe project evidence before sharing the repo or using it in applications.

## Data Quality Comparison Completed

> Compared clean and noisier local JSONL demonstrations under the same ACT-style train/eval setup, then reported changes in rollout success rate, final distance, action smoothness, and failure cases to analyze data quality effects.

## Full Report Completed

> Produced a reproducible LunaVLA project report connecting dataset design, BC/ACT policy learning, action-chunk ablations, rollout metrics, failure taxonomy, action statistics, and interview-ready conclusions.

## Avoid These Claims

- "deployed on a real robot"
- "state-of-the-art robotics model"
- "OpenVLA/openpi/pi0 reproduction"
- "large-scale VLA foundation model"
- "production-ready manipulation policy"
