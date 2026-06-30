# JD To Project Mapping

Use this table to connect job-description keywords to concrete repo evidence. Do not list keywords alone; point to commands, files, metrics, and artifacts.

| JD Keyword | What To Show | Repo Evidence |
| --- | --- | --- |
| VLA / embodied AI | You understand the observation-to-action loop and its limits | README, `docs/internship_pack/01_vla_internship_skill_map.md` |
| imitation learning | Demonstration-to-policy training | `trainer/train_act_pusht.py`, `trainer/train_bc_pusht.py` |
| behavior cloning | Next-action supervised imitation baseline | `python scripts/run_bc_smoke.py` |
| model tuning | Controlled BC hidden-size comparison | `python scripts/run_policy_tuning_comparison.py`, `outputs/policy_tuning_comparison.md` |
| ACT | Action chunk prediction and chunk-size ablation | `configs/act_pusht_baseline.yaml`, `configs/act_pusht_ablation_chunk_size.yaml` |
| rollout evaluation | Behavior over time, not just training loss | `eval_vla.py`, saved rollouts, `web_demo.html` |
| evaluation robustness | More episodes, saved demos, success/failure examples | `python scripts/run_extended_evaluation.py`, `outputs/extended_evaluation_report.md` |
| failure analysis | Debugging wrong direction, stuck behavior, and subtask failures | `docs/failure_taxonomy.md`, `python scripts/generate_failure_review.py` |
| task decomposition | Phase/subtask context for explaining behavior | `dataset/task_context.py`, `python scripts/check_task_layer.py` |
| task understanding | Rollout trace phase analysis and failed-subtask counts | `python scripts/generate_task_understanding_report.py`, `outputs/task_understanding_report.md` |
| action representation | Action scale, smoothness, chunks, and normalization | `docs/tutorials/action_normalization.md`, action statistics reports |
| action diagnostics | Train-target vs eval-executable action analysis | `python scripts/generate_action_analysis_report.py`, `outputs/action_analysis_report.md` |
| data quality | Clean/noisy demonstration comparison | `python scripts/run_data_quality_comparison.py`, `outputs/data_quality_comparison.md` |
| experiment reporting | Reproducible project communication | `scripts/generate_project_report.py`, `scripts/build_submission_pack.py` |
| GitHub project evidence | A reviewer can rerun, inspect, and verify your work | README commands, configs, generated report paths |

## Interview Framing

When a JD says "VLA", "imitation learning", "ACT", or "robot learning", answer with evidence:

> I built a small IL/VA project loop. It loads demonstration records, trains BC and ACT-style policies, evaluates saved rollouts, logs failure cases, and generates report-ready evidence. I treat it as a teaching-scale bridge toward VLA systems, not as a real-robot foundation model.

## Claim Hygiene

Good:

> I implemented a tiny observation-to-action training and rollout-evaluation loop with ACT-style action chunks and failure analysis.

Bad:

> I built a production robotics policy.

Good:

> I used optional instruction/task-context fields and phase labels to make the run easier to analyze.

Bad:

> I reproduced OpenVLA or openpi.
