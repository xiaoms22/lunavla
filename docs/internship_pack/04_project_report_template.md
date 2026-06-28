# Project Report Template

## Title

MiniMind-VLA: A Tiny ACT-Style Observation-to-Action Learning Loop

Start with an auto-generated draft:

```bash
python scripts/generate_project_report.py --run-dir outputs/act_pusht_baseline
```

Then edit the generated report with your own rollout observations and ablation notes.

## Abstract

Summarize the task, policy, dataset source, training setup, evaluation metrics, and the strongest result in 3-5 sentences.

## Motivation

Explain why a small runnable VLA-style project is useful for learning embodied AI: it connects demonstration data, policy learning, rollout behavior, and failure analysis.

## Method

Describe:

- the VLA record schema;
- how PushT-style demonstrations are generated;
- the ACT-style action chunk target;
- the train/eval/demo workflow.

## Experiment Setup

Record:

- config file;
- number of episodes and steps;
- chunk size;
- training steps and learning rate;
- evaluation episode count;
- checkpoint path.

## Results

| Run | Chunk Size | Final Loss | Success Rate | Mean Final Distance | Action Smoothness |
| --- | --- | --- | --- | --- | --- |
| baseline | | | | | |
| ablation | | | | | |

## Failure Analysis

List 2-3 failed episodes and classify them with `docs/failure_taxonomy.md`.

## Discussion

Explain what changed behavior, what remained unstable, and what you would tune next.

## Resume Bullet

Paste the matching bullet from `docs/internship_pack/02_resume_bullets.md` and adjust the numbers to your run.
