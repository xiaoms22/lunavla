from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether a LunaVLA run is ready to report.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/ or an absolute path.")
    parser.add_argument("--out", default=None, help="Markdown output path. Defaults to <run-dir>/run_diagnostic.md.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def number(value: Any) -> float | None:
    if value in (None, "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def status_rank(status: str) -> int:
    return {"pass": 0, "warn": 1, "fail": 2}[status]


def add_check(checks: list[dict[str, str]], name: str, status: str, detail: str, next_action: str) -> None:
    checks.append(
        {
            "check": name,
            "status": status,
            "detail": detail,
            "next_action": next_action,
        }
    )


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{key}:{value}" for key, value in sorted(value.items()))
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def artifact_checks(run_dir: Path) -> tuple[list[dict[str, str]], list[Path]]:
    required = [
        run_dir / "checkpoint.pt",
        run_dir / "training_summary.json",
        run_dir / "eval_summary.json",
        run_dir / "action_statistics.json",
        run_dir / "summary_report.md",
        run_dir / "project_report.md",
        run_dir / "resume_pack.md",
        run_dir / "web_demo.html",
    ]
    rollout_dir = run_dir / "rollouts"
    missing = [path for path in required if not path.exists()]
    if not rollout_dir.exists() or not any(rollout_dir.glob("*.json")):
        missing.append(rollout_dir)
    rows = [
        {
            "artifact": relative(path),
            "exists": "yes" if path.exists() and (not path.is_dir() or any(path.glob("*.json"))) else "no",
        }
        for path in [*required, rollout_dir]
    ]
    return rows, missing


def build_checks(run_dir: Path, training: dict[str, Any], evaluation: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    artifacts, missing = artifact_checks(run_dir)
    if missing:
        add_check(
            checks,
            "artifact completeness",
            "fail",
            "Missing " + ", ".join(relative(path) for path in missing),
            "Rerun the training, eval, summary, project report, resume pack, and web demo commands.",
        )
    else:
        add_check(
            checks,
            "artifact completeness",
            "pass",
            f"{len(artifacts)} expected run artifacts are present.",
            "Use this run directory as a complete evidence folder.",
        )

    success_rate = number(evaluation.get("success_rate"))
    if success_rate is None:
        add_check(
            checks,
            "success rate",
            "fail",
            "No success_rate was found in eval_summary.json.",
            "Run eval_vla.py before reporting rollout metrics.",
        )
    elif success_rate >= 0.8:
        add_check(
            checks,
            "success rate",
            "pass",
            f"success_rate={success_rate:.4g}",
            "This can be cited as the headline rollout metric for this teaching task.",
        )
    elif success_rate > 0:
        add_check(
            checks,
            "success rate",
            "warn",
            f"success_rate={success_rate:.4g}",
            "Cite it with the small-run boundary and inspect failures before writing conclusions.",
        )
    else:
        add_check(
            checks,
            "success rate",
            "fail",
            "success_rate=0",
            "Treat this as a runnable debugging run, not a positive baseline result.",
        )

    episodes = number(evaluation.get("episodes"))
    if episodes is None:
        add_check(
            checks,
            "episode count",
            "warn",
            "No episode count was found.",
            "Rerun evaluation with an explicit --episodes value.",
        )
    elif episodes < 5:
        add_check(
            checks,
            "episode count",
            "warn",
            f"episodes={int(episodes)}",
            "Good for smoke testing; rerun with more episodes before making stronger claims.",
        )
    else:
        add_check(
            checks,
            "episode count",
            "pass",
            f"episodes={int(episodes)}",
            "Enough for a small teaching report. Larger reports should still use more episodes.",
        )

    mean_final_distance = number(evaluation.get("mean_final_distance"))
    success_distance = number(evaluation.get("success_distance")) or 0.10
    if mean_final_distance is None:
        add_check(
            checks,
            "final distance",
            "fail",
            "No mean_final_distance was found.",
            "Run eval_vla.py and summarize the run again.",
        )
    elif mean_final_distance <= success_distance:
        add_check(
            checks,
            "final distance",
            "pass",
            f"mean_final_distance={mean_final_distance:.4g}, success_distance={success_distance:.4g}",
            "The average final distance is inside the success threshold.",
        )
    elif mean_final_distance <= success_distance * 2:
        add_check(
            checks,
            "final distance",
            "warn",
            f"mean_final_distance={mean_final_distance:.4g}, success_distance={success_distance:.4g}",
            "The policy often gets near the goal; inspect rollout behavior before claiming success.",
        )
    else:
        add_check(
            checks,
            "final distance",
            "fail",
            f"mean_final_distance={mean_final_distance:.4g}, success_distance={success_distance:.4g}",
            "Use failure analysis first, then tune data coverage, chunk size, or model capacity.",
        )

    failure_count = int(number(evaluation.get("failure_count")) or len(failures))
    if failure_count == 0:
        add_check(
            checks,
            "failure cases",
            "pass",
            "No failures were logged for this eval run.",
            "Still inspect at least one rollout before writing the report.",
        )
    else:
        add_check(
            checks,
            "failure cases",
            "warn",
            f"failure_count={failure_count}",
            "Use failure_cases.jsonl to name concrete failure modes and avoid overclaiming.",
        )

    final_loss = number(training.get("final_loss"))
    if final_loss is None:
        add_check(
            checks,
            "training summary",
            "warn",
            "No final_loss was found.",
            "Rerun training or check training_summary.json before comparing runs.",
        )
    else:
        add_check(
            checks,
            "training summary",
            "pass",
            f"final_loss={final_loss:.6g}",
            "Use loss as an optimization signal, not as the only project result.",
        )
    action_stats_path = training.get("action_stats_path") or evaluation.get("action_stats_path")
    if action_stats_path and action_stats_path != "n/a" and (run_dir / "action_statistics.json").exists():
        add_check(
            checks,
            "action statistics",
            "pass",
            f"stats={action_stats_path}",
            "Use action mean/std to explain scale, clipping, and normalization boundaries.",
        )
    else:
        add_check(
            checks,
            "action statistics",
            "warn",
            "No action_statistics.json was found for this run.",
            "Rerun training or run scripts/generate_action_statistics.py before discussing action scale.",
        )
    return checks


def verdict(checks: list[dict[str, str]]) -> str:
    if not checks:
        return "fail"
    return max((check["status"] for check in checks), key=status_rank)


def safe_claims(overall: str) -> list[str]:
    if overall == "pass":
        return [
            "Safe: this run completed train, eval, reporting, rollout browser, and generated artifacts.",
            "Safe: cite success_rate, mean_final_distance, rollout length, and action smoothness for this PushT-style teaching task.",
            "Still avoid: claims about real-robot deployment or broad robot foundation model capability.",
        ]
    if overall == "warn":
        return [
            "Safe: this run is useful as learning evidence if you include the warning checks.",
            "Safe: cite the metric values exactly and include at least one failure observation.",
            "Avoid: presenting the run as a fully successful baseline without showing the boundary.",
        ]
    return [
        "Safe: this run proves that some parts of the pipeline executed, if the artifact check passed.",
        "Use it as a debugging note, not as a positive baseline result.",
        "Avoid: resume or report claims that imply the policy solved the task.",
    ]


def file_rows(run_dir: Path) -> list[dict[str, str]]:
    return [
        {"file": relative(run_dir / "summary_report.md"), "why": "metric summary"},
        {"file": relative(run_dir / "project_report.md"), "why": "technical report draft"},
        {"file": relative(run_dir / "action_statistics.json"), "why": "action scale and normalization stats"},
        {"file": relative(run_dir / "resume_pack.md"), "why": "resume and interview wording"},
        {"file": relative(run_dir / "web_demo.html"), "why": "rollout browser"},
        {"file": relative(run_dir / "failure_cases.jsonl"), "why": "failure examples"},
        {"file": relative(run_dir / "rollouts"), "why": "saved episode trajectories"},
    ]


def build_markdown(run_dir: Path, training: dict[str, Any], evaluation: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    checks = build_checks(run_dir, training, evaluation, failures)
    overall = verdict(checks)
    metrics = [
        {"metric": "project_name", "value": training.get("project_name", run_dir.name)},
        {"metric": "records", "value": training.get("records", "n/a")},
        {"metric": "chunk_size", "value": training.get("chunk_size", "n/a")},
        {"metric": "action_stats_path", "value": training.get("action_stats_path", evaluation.get("action_stats_path", "n/a"))},
        {"metric": "action_mean", "value": training.get("action_mean", evaluation.get("action_mean", "n/a"))},
        {"metric": "action_std", "value": training.get("action_std", evaluation.get("action_std", "n/a"))},
        {"metric": "final_loss", "value": training.get("final_loss", "n/a")},
        {"metric": "episodes", "value": evaluation.get("episodes", "n/a")},
        {"metric": "success_rate", "value": evaluation.get("success_rate", "n/a")},
        {"metric": "mean_final_distance", "value": evaluation.get("mean_final_distance", "n/a")},
        {"metric": "mean_rollout_length", "value": evaluation.get("mean_rollout_length", "n/a")},
        {"metric": "mean_action_smoothness", "value": evaluation.get("mean_action_smoothness", "n/a")},
        {"metric": "failure_count", "value": evaluation.get("failure_count", len(failures))},
        {"metric": "failure_categories", "value": evaluation.get("failure_category_counts", {})},
    ]
    artifact_rows, _ = artifact_checks(run_dir)

    lines: list[str] = [
        "# LunaVLA Run Diagnostic",
        "",
        f"Run directory: `{relative(run_dir)}`",
        "",
        "## Verdict",
        "",
        f"Overall: `{overall}`",
        "",
        "A `pass` run is ready for a small teaching report. A `warn` run is still useful, but the warning should be visible in the report. A `fail` run should be treated as a debugging run.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(markdown_table(checks))
    lines.extend(["", "## Metrics", ""])
    lines.extend(markdown_table(metrics))
    lines.extend(["", "## Artifact Presence", ""])
    lines.extend(markdown_table(artifact_rows))
    lines.extend(["", "## Safe Claim Guidance", ""])
    lines.extend(f"- {claim}" for claim in safe_claims(overall))
    lines.extend(["", "## Files To Inspect", ""])
    lines.extend(markdown_table(file_rows(run_dir)))
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "```bash",
            f"python scripts/diagnose_run.py --run-dir {relative(run_dir)}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    out_path = resolve(args.out) if args.out else run_dir / "run_diagnostic.md"
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    failures = read_jsonl(run_dir / "failure_cases.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(run_dir, training, evaluation, failures), encoding="utf-8")
    print(f"run diagnostic: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
