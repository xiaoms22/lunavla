# Final Package Checklist

Use this checklist when you are ready to turn a LunaVLA run into a GitHub project, report, resume bullet, and interview story.

## Build The Pack

Run the project evidence commands first:

```bash
python scripts/run_quickstart.py
python scripts/run_baseline_evidence.py
python scripts/run_ablation_evidence.py
python scripts/build_evidence_pack.py --skip-runs
python scripts/build_submission_pack.py
python scripts/check_final_package.py
python scripts/check_reviewer_readiness.py
```

The main review folder is:

```text
outputs/submission_pack/
```

After `python scripts/build_submission_pack.py`, this checklist is copied into:

```text
outputs/submission_pack/final_package_checklist.md
```

The final package checker writes:

```text
outputs/final_package_check.md
outputs/submission_pack/final_package_check.md
```

## Required Deliverables

| deliverable | artifact to open | what it should prove |
| --- | --- | --- |
| GitHub repo | `README.md` | A beginner can understand the project promise, run commands, and see checked visuals. |
| Baseline report | `outputs/act_pusht_baseline/project_report.md` | The ACT PushT-style baseline has config, metrics, rollout evidence, and honest boundaries. |
| Ablation report | `outputs/run_comparison.md` and `outputs/config_diff.md` | The chunk-size ablation changed one variable and has metric evidence. |
| Rollout demo | `outputs/act_pusht_baseline/web_demo.html` | A reviewer can inspect saved rollout behavior instead of reading only loss. |
| Failure taxonomy | `outputs/failure_review.md` and `docs/failure_taxonomy.md` | Failures are categorized with next minimal checks. |
| Resume bullet | `outputs/act_pusht_baseline/resume_pack.md` | The resume claim is tied to commands, metrics, and project boundaries. |
| Two-minute explanation | `outputs/interview_flashcards.md` and `docs/interview_pitch.md` | You can explain data, policy, rollout, metrics, failure analysis, and limits. |

## Reviewer Pass

Before sharing the repo or using it in applications, check:

- The README starts from a runnable command, not a vague promise.
- Every metric you cite appears in a generated artifact.
- Every image or GIF in the README has attribution or local provenance.
- The ablation claim is supported by `outputs/config_diff.md`.
- The rollout claim is supported by saved rollout JSON or `web_demo.html`.
- The resume bullet does not claim real-robot deployment, OpenVLA/openpi reproduction, or state-of-the-art robotics performance.

## Interview Story

A safe two-minute structure:

1. I built LunaVLA as a tiny IL/VA learning loop for VLA beginners.
2. The data record contains observation, optional instruction text, action, episode id, timestep, task phase, and metadata.
3. The policy learns from demonstrations and predicts actions or short action chunks.
4. I evaluate with rollout success rate, final distance, action smoothness, and failure cases.
5. I ran a chunk-size ablation and used config diff plus rollout metrics to avoid overclaiming.
6. The honest boundary is that this is a teaching-scale PushT-style project, not a real-robot benchmark or frontier VLA reproduction.

## Completion Rule

Call the project complete only when a reviewer can:

- clone the repo;
- run the quickstart;
- inspect the baseline and ablation reports;
- open a rollout demo;
- read one failure analysis;
- understand one resume bullet;
- hear a two-minute explanation without any unsupported claim.
