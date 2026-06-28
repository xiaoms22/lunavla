# MiniMind-VLA: A Tiny VLA Internship Project You Can Run

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-green)
![CPU Smoke](https://img.shields.io/badge/CPU%20Smoke-passing-brightgreen)
![Task](https://img.shields.io/badge/Task-ACT%20%2B%20PushT--style-orange)

学过 VLA 但缺少一个能写进简历、能在面试里讲清楚的项目？MiniMind-VLA 用一个轻量 imitation-learning 闭环，把 `observation -> action -> rollout -> evaluation` 跑起来。

它参考 MiniMind 的思路：低成本、可复现、从代码理解完整链路。这里不追求大模型规模和真实机器人部署，而是提供一个普通电脑也能启动、普通学生也能改懂的具身智能项目起点。

![MiniMind-VLA architecture](images/minimind-vla-architecture.svg)

## Quick Results

The images below are generated from a checked local run:

```bash
python scripts/export_readme_assets.py --run-dir outputs/act_pusht_baseline --out-dir images
```

![Rollout demo](images/rollout_demo.png)

![Loss curve](images/loss_curve_baseline.png)

![Result table](images/result_table.svg)

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_cpu_smoke.py
```

This single command runs training, evaluation, summary generation, and a static HTML rollout demo. Local artifacts are written to `outputs/cpu_smoke/` and ignored by Git.

Run the baseline manually:

```bash
python trainer/train_act_pusht.py --config configs/act_pusht_baseline.yaml
python eval_vla.py --checkpoint outputs/act_pusht_baseline/checkpoint.pt --episodes 50 --save-rollouts
python scripts/summarize_results.py --run-dir outputs/act_pusht_baseline
python scripts/export_readme_assets.py --run-dir outputs/act_pusht_baseline --out-dir images
```

Run the chunk-size ablation:

```bash
python trainer/train_act_pusht.py --config configs/act_pusht_ablation_chunk_size.yaml
python scripts/compare_runs.py --runs outputs/act_pusht_baseline outputs/act_pusht_ablation_chunk_size --out outputs/run_comparison.md
```

## What You Build

MiniMind-VLA is intentionally small, but it includes the pieces a VLA internship project should be able to explain:

- data records with `observation`, `action`, `episode_id`, `timestep`, `success`, and `metadata`;
- a PushT-style demonstration generator;
- an ACT-style action chunk policy;
- config-driven training and checkpoint export;
- rollout evaluation with success rate, final distance, rollout length, and action smoothness;
- failure-case logging, result summaries, README assets, and a static web demo.

## Internship Pack

If your goal is to turn this into project evidence, start here:

- `docs/internship_pack/01_vla_internship_skill_map.md`: what the project teaches.
- `docs/internship_pack/02_resume_bullets.md`: resume bullets matched to completed work.
- `docs/internship_pack/03_interview_qa.md`: interview answers for VLA, BC, ACT, rollout, and failure analysis.
- `docs/internship_pack/04_project_report_template.md`: experiment report template.
- `docs/internship_pack/05_jd_to_project_mapping.md`: map JD keywords to code evidence.
- `docs/internship_pack/06_4_week_project_path.md`: four-week learning path.
- `docs/internship_pack/07_advanced_project_path.md`: stronger project path after the baseline works.

## Repository Layout

```text
minimind-vla/
  configs/              # CPU smoke, baseline, and ablation configs
  dataset/              # VLA record schema and PushT-style data generator
  docs/                 # learning notes, evaluation guide, and internship pack
  images/               # README-visible rollout, loss, architecture, and result assets
  model/                # tiny policy and ACT-style wrapper
  scripts/              # smoke test, summaries, asset export, and web demo generator
  trainer/              # training entrypoints and shared utilities
  eval_vla.py           # rollout evaluation entrypoint
```

## Data Schema

Each training record follows this shape:

```json
{
  "observation": [0.12, 0.33, 0.80, 0.20],
  "action": [0.05, -0.02],
  "episode_id": 0,
  "timestep": 3,
  "success": false,
  "language_instruction": "push the T block to the goal",
  "metadata": {"task": "pusht_mock"}
}
```

## Honest Claim

MiniMind-VLA is a tiny, readable, reproducible project starter for learning observation-to-action training. It is not a real-robot deployment benchmark and does not claim state-of-the-art robotics performance.

## License

Apache-2.0. This repository is built as an educational and internship-oriented VLA scaffold.
