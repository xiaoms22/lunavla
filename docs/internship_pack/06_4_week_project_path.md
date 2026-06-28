# Four-Week Project Path

This path turns MiniMind-VLA from a runnable repo into a project a VLA beginner can understand, report, and explain clearly.

## Week 1: Run And Read

- Run `python scripts/inspect_dataset.py`.
- Run `python scripts/run_cpu_smoke.py`.
- Read `dataset/pusht_dataset.py`, `model/minivla_policy.py`, and `trainer/train_act_pusht.py`.
- Write down the shape of one training sample.

Deliverable: one paragraph explaining `observation -> action chunk`.

## Week 2: Baseline

- Run `python scripts/run_baseline_evidence.py`.
- Inspect the summary, project report, web demo, and README assets.
- Optionally rerun `python scripts/run_baseline_evidence.py --episodes 50` for a stronger report.

Deliverable: baseline result table and rollout screenshot.

## Week 3: Ablation

- Run `python scripts/run_ablation_evidence.py`.
- Read `outputs/run_comparison.md` and inspect at least one rollout from each run.
- Explain how chunk size changes behavior.

Deliverable: one ablation report with metric deltas and a short conclusion.

## Week 4: Report And Interview Pack

- Run `python scripts/build_evidence_pack.py --skip-runs`.
- Fill in the project report template.
- Generate a first draft with `python scripts/generate_project_report.py --run-dir outputs/act_pusht_baseline`.
- Generate interview material with `python scripts/generate_resume_pack.py --run-dir outputs/act_pusht_baseline`.
- Choose a matching resume bullet.
- Practice the two-minute interview pitch.

Deliverable: evidence index, public project README, report, resume/interview pack, and interview script.
