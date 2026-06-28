# JD To Project Mapping

Use this table to connect job-description keywords to concrete repo evidence.

| JD Keyword | What To Show | Repo Evidence |
| --- | --- | --- |
| VLA / embodied AI | Observation-to-action system understanding | README architecture and data schema |
| imitation learning | Demonstration-to-policy training | `trainer/train_act_pusht.py` |
| ACT | Action chunk prediction | baseline and ablation configs |
| rollout | Behavior evaluation over time | `eval_vla.py` and rollout JSON |
| failure analysis | Debugging behavior, not just loss | `docs/failure_taxonomy.md` |
| experiment reporting | Reproducible project communication | report template and result table |

## Interview Framing

When a JD says "熟悉 VLA / 模仿学习 / ACT", do not simply list keywords. Point to evidence:

> I built a small ACT-style imitation-learning loop, evaluated rollout success, and analyzed failure cases with reproducible configs and generated demo assets.

## Claim Hygiene

Good:

> I implemented a tiny VLA-style training and rollout-evaluation loop.

Bad:

> I built a production robotics policy.
