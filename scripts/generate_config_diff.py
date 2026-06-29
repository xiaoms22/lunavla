from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BOOKKEEPING_PATHS = {
    "project_name",
    "artifacts.output_dir",
    "artifacts.report_path",
}
EXPECTED_EXPERIMENT_PATHS = {
    "model.chunk_size",
    "policy.chunk_size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LunaVLA baseline and ablation configs.")
    parser.add_argument("--baseline", default="configs/act_pusht_baseline.yaml", help="Baseline config path.")
    parser.add_argument("--candidate", default="configs/act_pusht_ablation_chunk_size.yaml", help="Ablation config path.")
    parser.add_argument("--out", default="outputs/config_diff.md", help="Markdown config diff path.")
    parser.add_argument("--json-out", default="outputs/config_diff.json", help="Machine-readable diff path.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{relative(path)} must be a mapping")
    return data


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        rows: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.update(flatten(item, child))
        return rows
    return {prefix: value}


def classify(path: str) -> str:
    if path in BOOKKEEPING_PATHS:
        return "bookkeeping"
    if path in EXPECTED_EXPERIMENT_PATHS:
        return "expected experiment variable"
    return "extra experimental change"


def diff_configs(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_flat = flatten(baseline)
    candidate_flat = flatten(candidate)
    rows: list[dict[str, Any]] = []
    for path in sorted(set(baseline_flat) | set(candidate_flat)):
        baseline_value = baseline_flat.get(path, "<missing>")
        candidate_value = candidate_flat.get(path, "<missing>")
        if baseline_value != candidate_value:
            rows.append(
                {
                    "path": path,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "kind": classify(path),
                }
            )
    return rows


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No differences."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def clean_ablation(rows: list[dict[str, Any]]) -> bool:
    extra = [row for row in rows if row["kind"] == "extra experimental change"]
    expected = {row["path"] for row in rows if row["kind"] == "expected experiment variable"}
    return not extra and expected == EXPECTED_EXPERIMENT_PATHS


def build_payload(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    rows = diff_configs(read_yaml(baseline_path), read_yaml(candidate_path))
    return {
        "baseline": relative(baseline_path),
        "candidate": relative(candidate_path),
        "expected_experiment_paths": sorted(EXPECTED_EXPERIMENT_PATHS),
        "bookkeeping_paths": sorted(BOOKKEEPING_PATHS),
        "clean_one_variable_ablation": clean_ablation(rows),
        "differences": rows,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    rows = payload["differences"]
    extra = [row for row in rows if row["kind"] == "extra experimental change"]
    lines: list[str] = [
        "# LunaVLA Config Diff",
        "",
        "This file checks whether the public ablation changes only the intended experiment variable.",
        "",
        f"- Baseline: `{payload['baseline']}`",
        f"- Candidate: `{payload['candidate']}`",
        f"- Clean one-variable ablation: `{str(payload['clean_one_variable_ablation']).lower()}`",
        "",
        "## Differences",
        "",
    ]
    lines.extend(markdown_table(rows))
    lines.extend(["", "## Interpretation", ""])
    if payload["clean_one_variable_ablation"]:
        lines.append("- The experimental change is limited to `model.chunk_size` and `policy.chunk_size`.")
        lines.append("- Project name and artifact paths are bookkeeping differences needed to keep outputs separate.")
    else:
        lines.append("- Extra experimental changes were found. Treat the ablation conclusion as weaker until these are explained.")
        for row in extra:
            lines.append(f"- Extra change: `{row['path']}` changed from `{row['baseline']}` to `{row['candidate']}`.")
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_config_diff.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    payload = build_payload(resolve(args.baseline), resolve(args.candidate))
    out_path = resolve(args.out)
    json_path = resolve(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"config diff: {out_path}")
    print(f"config diff json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
