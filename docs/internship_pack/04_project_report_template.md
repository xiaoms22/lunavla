# Project Report Template

## Title

LunaVLA: A Tiny IL/VA Learning Loop For Observation-To-Action Policy Evaluation

Start with an auto-generated draft:

```bash
python scripts/generate_project_report.py --run-dir outputs/act_pusht_baseline
```

Then edit the generated report with your own rollout observations, ablation notes, and claim-safe conclusions.

## Abstract

Summarize the task, dataset source, policy, training setup, rollout metrics, and strongest result in 3-5 sentences. State that this is a teaching-scale IL/VA loop, not real-robot deployment.

## Motivation

Explain why a small runnable embodied-AI project is useful: it connects demonstration data, policy learning, action chunks, rollout behavior, failure analysis, and reportable evidence.

## Method

Describe:

- the LunaVLA record schema;
- how PushT-style demonstrations are generated or loaded;
- how the Task Layer labels phase/subtask context;
- the BC or ACT-style policy input/output;
- the action chunk target;
- the train/eval/report/demo workflow;
- how action statistics are recorded.

## Experiment Setup

Record:

- config file;
- dataset source;
- number of records, episodes, and rollout steps;
- policy type and chunk size;
- training steps and learning rate;
- evaluation episode count;
- checkpoint path;
- action statistics path.

## Results

| Run | Policy | Chunk Size | Final Loss | Success Rate | Mean Final Distance | Action Smoothness |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | ACT | | | | | |
| ablation | ACT | | | | | |
| optional | BC | | | | | |

## Action Statistics

Record mean, standard deviation, min/max, clipping fraction, and the explanation you would give in an interview.

## Failure Analysis

List 2-3 failed episodes and classify them with `docs/failure_taxonomy.md`. Include the failed subtask or phase when available.

## Discussion

Explain what changed behavior, what remained unstable, and what you would tune next. Separate training-loss observations from rollout-behavior observations.

## Resume Bullet

Paste the matching bullet from `docs/internship_pack/02_resume_bullets.md` and adjust it only with metrics that appear in your generated reports.

## Honest Boundary

End with one sentence that makes the scope clear:

> This project demonstrates a teaching-scale IL/VA learning and evaluation loop; it does not claim real-robot deployment or frontier VLA performance.
