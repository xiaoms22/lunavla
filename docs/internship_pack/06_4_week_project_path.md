# Four-Week Project Path

This path turns MiniMind-VLA from a runnable repo into a project a VLA beginner can understand, report, and explain clearly.

## Week 1: Run And Read

- Run `python scripts/check_environment.py`.
- Run `python scripts/run_quickstart.py`.
- Run `python scripts/inspect_dataset.py`.
- Run `python scripts/run_cpu_smoke.py`.
- Run `python scripts/generate_first_run_checklist.py`.
- Run `python scripts/generate_troubleshooting_guide.py`.
- Run `python scripts/generate_command_reference.py`.
- Run `python scripts/generate_code_walkthrough.py`.
- Read `dataset/pusht_dataset.py`, `model/minivla_policy.py`, and `trainer/train_act_pusht.py`.
- Write down the shape of one training sample.

Deliverable: environment check, first-run checklist, and one paragraph explaining `observation -> action chunk`.

## Week 2: Baseline

- Run `python scripts/run_baseline_evidence.py`.
- Inspect the summary, run diagnostic, project report, web demo, and README assets.
- Run `python scripts/check_readme_assets.py` after exporting README assets.
- Optionally rerun `python scripts/run_baseline_evidence.py --episodes 50` for a stronger report.

Deliverable: baseline result table, run diagnostic, README asset check, and rollout screenshot.

## Week 3: Ablation

- Run `python scripts/run_ablation_evidence.py`.
- Read `outputs/run_comparison.md` and inspect at least one rollout from each run.
- Read each run diagnostic before writing the ablation conclusion.
- Run `python scripts/generate_failure_review.py`.
- Explain how chunk size changes behavior.

Deliverable: one ablation report with metric deltas, failure review, and a short conclusion.

## Week 4: Report And Interview Pack

- Run `python scripts/build_evidence_pack.py --skip-runs`.
- Run `python scripts/build_submission_pack.py`.
- Run `python scripts/check_project_progress.py`.
- Run `python scripts/generate_learning_checkpoint.py`.
- Run `python scripts/generate_interview_flashcards.py`.
- Run `python scripts/generate_skill_evidence_map.py`.
- Run `python scripts/generate_project_card.py`.
- Run `python scripts/generate_showcase_issue.py`.
- Include `outputs/quickstart_summary.md` as the one-command starter summary.
- Include `outputs/environment_check.md` as reproducibility context.
- Include `outputs/first_run_checklist.md` as the smallest-loop readiness check.
- Include `outputs/troubleshooting_guide.md` as the recovery and debugging guide.
- Include `outputs/command_reference.md` as the public command map.
- Include `outputs/code_walkthrough.md` as the code reading guide.
- Include `outputs/readme_asset_check.md` as visual evidence context.
- Include `outputs/project_progress.md` as the evidence coverage checklist.
- Include `outputs/project_card.md` as the one-page overview.
- Include `outputs/learning_checkpoint.md` as the concept review checklist.
- Include `outputs/interview_flashcards.md` as the evidence-backed interview practice sheet.
- Include `outputs/skill_evidence_map.md` as the ability-to-evidence map.
- Include `outputs/learner_showcase.md` as the public sharing draft.
- Include `outputs/failure_review.md` as failure-analysis evidence.
- Fill in the project report template.
- Generate a first draft with `python scripts/generate_project_report.py --run-dir outputs/act_pusht_baseline`.
- Generate interview material with `python scripts/generate_resume_pack.py --run-dir outputs/act_pusht_baseline`.
- Check claim safety with `python scripts/diagnose_run.py --run-dir outputs/act_pusht_baseline`.
- Choose a matching resume bullet.
- Practice the two-minute interview pitch.

Deliverable: evidence index, submission pack, public project README, report, resume/interview pack, and interview script.
