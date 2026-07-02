# LunaVLA Internship Skill Map

LunaVLA is for learners who know VLA terms but have not yet built a runnable embodied-AI project. The public repo teaches the IL/VA core first: demonstrations become action targets, a policy predicts actions, rollouts test behavior, and reports turn results into evidence.

The honest framing is:

```text
IL/VA core now: demonstration -> action chunk dataset -> BC/ACT policy -> rollout evaluation -> report
VLA bridge later: language/task context and heavier robot-learning stacks can be added after the core loop is clear
```

## Core Skills

| Skill | What It Means | Repo Evidence |
| --- | --- | --- |
| Data schema | Read one sample as observation, action, episode, timestep, success, metadata, and optional instruction | `dataset/pusht_dataset.py`, `python scripts/inspect_dataset.py` |
| Imitation learning | Train a policy from demonstration records | `trainer/train_act_pusht.py`, `trainer/train_bc_pusht.py` |
| BC baseline | Understand next-action behavior cloning as supervised learning | `configs/bc_pusht_smoke.yaml`, `python scripts/run_bc_smoke.py` |
| ACT action chunks | Predict a short action sequence and study chunk size | `configs/act_pusht_baseline.yaml`, `configs/act_pusht_ablation_chunk_size.yaml` |
| Task Layer | Track phase/subtask context without requiring an LLM | `dataset/task_context.py`, `python scripts/check_task_layer.py` |
| Rollout evaluation | Judge behavior by repeated policy actions, not loss alone | `eval_vla.py`, saved rollout JSON, `web_demo.html` |
| Failure analysis | Explain where and why rollout behavior fails | `docs/failure_taxonomy.md`, `python scripts/generate_failure_review.py` |
| Action statistics | Explain action scale, clipping, and normalization | `python scripts/generate_action_statistics.py`, `docs/tutorials/action_normalization.md` |
| Project communication | Convert code and metrics into a report, resume bullet, and interview pitch | `scripts/generate_project_report.py`, `scripts/generate_resume_pack.py` |

## What A Beginner Should Be Able To Say

> I built a small IL/VA project loop. The dataset records observations and expert actions, the policy predicts an action chunk, evaluation rolls the policy forward, and the report compares training loss with rollout metrics and failure cases.

## What This Proves

- You can run an end-to-end observation-to-action learning loop.
- You can explain why low training loss does not guarantee successful rollout behavior.
- You can compare BC and ACT-style policies with metrics and saved artifacts.
- You can connect failure cases to subtask/phase context instead of only saying "the model failed".
- You can keep claims tied to exact commands, configs, metrics, and generated files.

## What It Does Not Claim

- Real-robot deployment.
- OpenVLA, openpi, or pi0 reproduction.
- A general robot foundation model.
- State-of-the-art robotics performance.
