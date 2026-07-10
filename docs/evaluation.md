# Evaluation

Training loss alone does not establish rollout behavior. `eval_vla.py` evaluates the synthetic `pusht_style_point_reach` task using configuration-driven goal, start range, action clip, threshold, seeds, horizon, and execution mode.

It writes `eval_summary.json`, optional `rollouts/episode_*.json`, and categorized `failure_cases.jsonl`. Before saving rollouts it removes stale files from the run being evaluated.

## Metrics

- success count/rate and 95% Wilson interval in controlled summaries;
- final distance, rollout length, and action smoothness;
- failure categories and final task stage;
- paired-bootstrap intervals for continuous treatment differences.

Automatic failure labels are inspection aids, not ground truth. Review the saved rollout before reporting a conclusion.

## Execution semantics

`receding_horizon` executes the first valid action and replans. `open_loop_chunk` executes every valid action before replanning. v1.1 chunk-size experiments use open-loop execution; chunk size one has the same single-action behavior.

Use `scripts/run_controlled_experiments.py` for publishable comparisons and `scripts/run_cpu_smoke.py` only for a local functional check.
