# Four-Week Project Path

This path turns MiniMind-VLA from a runnable repo into a project a VLA beginner can understand, report, and explain clearly.

## Week 1: Run And Read

- Run `python scripts/run_cpu_smoke.py`.
- Read `dataset/pusht_dataset.py`, `model/minivla_policy.py`, and `trainer/train_act_pusht.py`.
- Write down the shape of one training sample.

Deliverable: one paragraph explaining `observation -> action`.

## Week 2: Baseline

- Run `configs/act_pusht_baseline.yaml`.
- Evaluate with `eval_vla.py`.
- Generate a summary and demo page.

Deliverable: baseline result table and rollout screenshot.

## Week 3: Ablation

- Run `configs/act_pusht_ablation_chunk_size.yaml`.
- Compare the baseline and ablation with `scripts/compare_runs.py`.
- Read `outputs/run_comparison.md` and inspect at least one rollout from each run.
- Explain how chunk size changes behavior.

Deliverable: one ablation report with metric deltas and a short conclusion.

## Week 4: Report And Interview Pack

- Fill in the project report template.
- Generate a first draft with `python scripts/generate_project_report.py --run-dir outputs/act_pusht_baseline`.
- Choose a matching resume bullet.
- Practice the two-minute interview pitch.

Deliverable: public project README, report, resume bullet, and interview script.
