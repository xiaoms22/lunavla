# Initial Public Release

LunaVLA is released as a tiny, reproducible VLA internship project starter.

## Included

- Environment check command for Python, dependencies, repo files, and output write access.
- One-command quickstart for environment check, dataset inspection, CPU smoke, and first-run guidance.
- CPU smoke command for the full train/eval/demo loop.
- Public repo quality check for clean docs, safe wording, and beginner-friendly commands.
- First-run checklist for reviewing CPU smoke artifacts before moving to the baseline path.
- Troubleshooting guide for common missing-artifact, weak-run, and report-prep symptoms.
- Command reference generator for mapping public commands to generated artifacts.
- Code walkthrough generator for reading the runnable implementation in order.
- Action statistics generator for recording mean/std, clipping, and normalization formulas.
- Optional JSONL data smoke path for exporting local PushT-style demonstrations and reloading them with `dataset.source: jsonl`.
- Clean-vs-noisy JSONL data-quality comparison for learning how demonstration quality affects rollout behavior.
- BC hidden-size tuning comparison for learning why policy tuning should be judged with rollout evidence.
- Task understanding report for reading saved rollout traces, counting failed phases/subtasks, and adding a first-pass `phase_regression` label.
- Action analysis report for comparing train-time demonstration targets with eval-time executable rollout actions.
- Extended evaluation report for rerunning more rollout episodes, saving demos, and comparing success rate with mean final distance.
- Homepage summary generator for tying README-visible result claims to checked metrics and generated reports.
- Negative-path release checks for malformed configs and missing submission-pack sources.
- Release readiness check that keeps README-visible commands covered by the generated command reference.
- Task Layer check for generated record labels, rollout frame context, eval summaries, reports, and rollout browser details.
- ACT-style PushT baseline and chunk-size ablation config.
- Config diff generator for auditing the baseline vs ablation setup.
- Rollout evaluation with success rate, final distance, rollout length, and action smoothness.
- Static README assets generated from local run artifacts.
- Homepage ecosystem media from official LeRobot and LIBERO sources with attribution.
- README asset check for ensuring GIFs, PNGs, SVGs, and the asset manifest are usable.
- Project progress check for mapping generated artifacts to report-ready stages.
- Project card generator for a one-page command, metric, evidence, and boundary summary.
- Experiment ledger generator for linking commands, config hashes, metrics, and artifacts.
- Learning checkpoint generator for concept-to-evidence self-check questions.
- Interview flashcards generator for evidence-backed answers tied to code and run artifacts.
- Skill evidence map generator for connecting VLA abilities to code, commands, and artifacts.
- Learner showcase generator for copyable, evidence-backed public sharing.
- Internship pack with skill map, resume bullets, interview Q&A, JD mapping, and project paths.
- Submission pack generator for collecting reports, rollout browser, comparison results, and README assets.
- Run diagnostic generator for checking artifact completeness, metric strength, failure cases, and safe reporting claims.
- Failure review generator for summarizing rollout failure cases across smoke, baseline, and ablation runs.
- Model card and data card for honest, reproducible project reporting.

## Boundary

This release focuses on a lightweight observation-to-action learning loop. It is an educational project scaffold, not a real-robot deployment benchmark.
