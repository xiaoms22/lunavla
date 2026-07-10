from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
START = "<!-- VERIFIED_RESULTS_START -->"
END = "<!-- VERIFIED_RESULTS_END -->"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return (math.nan, math.nan)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
    return ((centre - spread) / denominator, (centre + spread) / denominator)


def metric(metrics: dict[str, Any], *names: str, default: Any = "n/a") -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return default


def format_number(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return "n/a"


def relative_link(target: Path, readme_path: Path) -> str:
    return Path(os.path.relpath(target, start=readme_path.parent)).as_posix()


def render_aggregate(index_path: Path, readme_path: Path, analysis: dict[str, Any]) -> str:
    seed_count = int(analysis.get("train_seed_count", 0))
    eval_episode_count = int(analysis.get("eval_episode_count", 0))
    status = "controlled" if analysis.get("controlled") else "observational"
    summaries = analysis.get("summaries", {})
    if not isinstance(summaries, dict) or not summaries:
        raise ValueError("evidence index analysis.summaries is empty")
    lines = [
        "| Experiment | Treatment | Train seeds | Eval trials | Success (95% Wilson CI) | Mean final distance | Mean smoothness | Evidence |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    contrast_rows: list[str] = []
    for family, summary_rel in sorted(summaries.items()):
        summary_path = index_path.parent / str(summary_rel)
        summary = read_json(summary_path)
        summary_link = relative_link(summary_path, readme_path)
        for aggregate in summary.get("aggregates", []):
            successes = int(aggregate.get("successes", 0))
            trials = int(aggregate.get("trials", 0))
            interval = aggregate.get("success_wilson_95", [])
            if trials > 0 and isinstance(interval, list) and len(interval) == 2:
                success_text = (
                    f"{successes / trials:.1%} "
                    f"({float(interval[0]):.1%}–{float(interval[1]):.1%})"
                )
            elif trials > 0:
                low, high = wilson(successes, trials)
                success_text = f"{successes / trials:.1%} ({low:.1%}–{high:.1%})"
            else:
                success_text = "n/a"
            lines.append(
                f"| `{family}` | `{aggregate.get('treatment', 'unknown')}` | {seed_count} | "
                f"{trials} | {success_text} | "
                f"{format_number(aggregate.get('mean_final_distance'))} | "
                f"{format_number(aggregate.get('mean_action_smoothness'))} | "
                f"[{status}]({summary_link}) |"
            )
        for contrast in summary.get("contrasts", []):
            interval = contrast.get("paired_bootstrap_95", [])
            interval_text = "n/a"
            if isinstance(interval, list) and len(interval) == 2:
                interval_text = (
                    f"[{format_number(interval[0])}, {format_number(interval[1])}]"
                )
            contrast_rows.append(
                f"| `{family}` | `{contrast.get('treatment')}` − `{contrast.get('reference')}` | "
                f"`{contrast.get('metric')}` | {int(contrast.get('paired_n', 0))} | "
                f"{format_number(contrast.get('mean_difference'))} | {interval_text} |"
            )
    lines.extend(
        [
            "",
            f"Each aggregate combines {seed_count} training seeds × {eval_episode_count} fixed evaluation episodes. Rows are rendered from validated manifests and predeclared summaries.",
        ]
    )
    if contrast_rows:
        lines.extend(
            [
                "",
                "Continuous paired contrasts (treatment minus reference):",
                "",
                "| Experiment | Contrast | Metric | Paired n | Mean difference | Paired bootstrap 95% CI |",
                "| --- | --- | --- | ---: | ---: | --- |",
                *contrast_rows,
                "",
                "A controlled label describes the design. A directional claim is allowed only when the relevant paired interval excludes zero in the declared direction.",
            ]
        )
    return "\n".join(lines)


def render_runs(index_path: Path, readme_path: Path, runs: list[dict[str, Any]]) -> str:
    lines = [
        "| Run | Policy | Task | Chunk | Eval trials | Success (95% Wilson CI) | Mean final distance | Evidence |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for item in sorted(runs, key=lambda value: str(value.get("run_id", ""))):
        manifest_path = index_path.parent / str(item["manifest"])
        manifest = read_json(manifest_path)
        metrics = manifest.get("metrics", {}).get("evaluation", {})
        successes = int(metric(metrics, "success_count", "successes", default=0))
        trials = int(metric(metrics, "episodes", "trials", default=0))
        low, high = wilson(successes, trials)
        success_text = (
            f"{successes / trials:.1%} ({low:.1%}–{high:.1%})" if trials else "n/a"
        )
        config_path = manifest_path.parent / "resolved_config.json"
        config = read_json(config_path) if config_path.is_file() else {}
        policy_config = config.get("policy", {}) if isinstance(config.get("policy"), dict) else {}
        status = "controlled" if item.get("controlled") else "observational"
        link = relative_link(manifest_path, readme_path)
        lines.append(
            f"| `{item.get('run_id', 'unknown')}` | `{manifest.get('policy_id', 'unknown')}` | "
            f"`{manifest.get('task_id', 'unknown')}` | {policy_config.get('chunk_size', 'n/a')} | "
            f"{trials} | {success_text} | {format_number(metric(metrics, 'mean_final_distance'))} | "
            f"[{status}]({link}) |"
        )
    lines.extend(
        [
            "",
            "These are individual-run rows. They are diagnostic unless the index also contains a validated multi-seed analysis.",
        ]
    )
    return "\n".join(lines)


def render(index_path: Path, readme_path: Path) -> str:
    index = read_json(index_path)
    runs = index.get("runs", [])
    analysis = index.get("analysis")
    if isinstance(analysis, dict):
        return render_aggregate(index_path, readme_path, analysis)
    if isinstance(runs, list) and runs:
        return render_runs(index_path, readme_path, runs)
    return (
        "No controlled v1.1 result snapshot has been published yet. Historical v1.0 numbers were produced "
        "by non-controlled runs and are not evidence that action chunking caused an improvement."
    )


def replace_section(readme: str, body: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), flags=re.DOTALL)
    replacement = f"{START}\n{body.rstrip()}\n{END}"
    updated, count = pattern.subn(replacement, readme)
    if count != 1:
        raise ValueError(f"Expected exactly one generated result section, found {count}")
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render README verified results from results/v1.1 manifests.")
    parser.add_argument("--index", default="results/v1.1/index.json")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    index_path = resolve(args.index)
    readme_path = resolve(args.readme)
    current = readme_path.read_text(encoding="utf-8")
    expected = replace_section(current, render(index_path, readme_path))
    if args.check:
        if expected != current:
            print("README verified-results section is stale; run scripts/render_readme_results.py", file=sys.stderr)
            return 1
        print("README verified-results section is current")
        return 0
    readme_path.write_text(expected, encoding="utf-8")
    print(f"updated: {readme_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
