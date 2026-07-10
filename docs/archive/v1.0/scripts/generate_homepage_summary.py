from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTENDED_CSV = "outputs/extended_evaluation_report.csv"
DEFAULT_TASK_REPORT = "outputs/task_understanding_report.md"
DEFAULT_ACTION_REPORT = "outputs/action_analysis_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README-facing LunaVLA presentation assets.")
    parser.add_argument("--extended-csv", default=DEFAULT_EXTENDED_CSV, help="Extended evaluation CSV path.")
    parser.add_argument("--out", default="outputs/homepage_summary.md", help="Markdown homepage summary path.")
    parser.add_argument("--image", default="images/homepage_results.svg", help="README-visible SVG result card.")
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
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_extended_rows(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    fallback_runs = [
        ("cpu_smoke", "outputs/cpu_smoke"),
        ("bc_pusht_cpu_smoke", "outputs/bc_pusht_cpu_smoke"),
        ("act_pusht_baseline", "outputs/act_pusht_baseline"),
        ("act_pusht_jsonl_noisy_smoke", "outputs/act_pusht_jsonl_noisy_smoke"),
    ]
    rows: list[dict[str, Any]] = []
    for name, run_dir_text in fallback_runs:
        run_dir = resolve(run_dir_text)
        summary = read_json(run_dir / "eval_summary.json")
        if not summary:
            continue
        rows.append(
            {
                "run": name,
                "episodes": summary.get("episodes", "n/a"),
                "success_rate": summary.get("success_rate", "n/a"),
                "mean_final_distance": summary.get("mean_final_distance", "n/a"),
                "failure_count": summary.get("failure_count", "n/a"),
                "conclusion": "fallback from run eval_summary.json",
            }
        )
    return rows


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_number(value: Any) -> str:
    number = as_float(value, default=float("nan"))
    if number != number:
        return str(value)
    if abs(number) >= 1:
        return f"{number:.3g}"
    return f"{number:.4g}"


def display_name(run: str) -> str:
    names = {
        "cpu_smoke": "CPU smoke",
        "bc_pusht_cpu_smoke": "BC MLP",
        "act_pusht_baseline": "ACT baseline",
        "act_pusht_jsonl_noisy_smoke": "Noisy JSONL",
    }
    return names.get(run, run.replace("_", " "))


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = ["act_pusht_baseline", "bc_pusht_cpu_smoke", "act_pusht_jsonl_noisy_smoke", "cpu_smoke"]
    ordered = sorted(rows, key=lambda row: order.index(str(row.get("run"))) if str(row.get("run")) in order else len(order))
    return [
        {
            "run": display_name(str(row.get("run", "unknown"))),
            "episodes": row.get("episodes", "n/a"),
            "success_rate": row.get("success_rate", "n/a"),
            "mean_final_distance": row.get("mean_final_distance", "n/a"),
            "failure_count": row.get("failure_count", "n/a"),
            "lesson": row.get("conclusion", "inspect saved rollouts before making claims"),
        }
        for row in ordered
    ]


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines


def bar_width(value: Any, max_width: int = 260) -> int:
    return max(0, min(max_width, int(round(as_float(value) * max_width))))


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1040
    row_height = 82
    height = 132 + row_height * max(1, len(rows)) + 34
    colors = ["#1d6f42", "#305cde", "#d36b21", "#7a4cc2", "#5a6b7d"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<rect x="24" y="24" width="992" height="88" rx="12" fill="#ffffff" stroke="#d7deea"/>',
        '<text x="48" y="60" font-family="Arial, sans-serif" font-size="28" font-weight="700" fill="#172033">LunaVLA checked results</text>',
        '<text x="48" y="88" font-family="Arial, sans-serif" font-size="15" fill="#536071">Extended rollout evaluation: success rate + final distance + evidence boundary</text>',
    ]
    y = 150
    for index, row in enumerate(rows):
        color = colors[index % len(colors)]
        run = html.escape(str(row.get("run", "unknown")))
        lesson = html.escape(str(row.get("lesson", ""))[:94])
        success = format_number(row.get("success_rate", "n/a"))
        distance = format_number(row.get("mean_final_distance", "n/a"))
        episodes = html.escape(str(row.get("episodes", "n/a")))
        failures = html.escape(str(row.get("failure_count", "n/a")))
        parts.extend(
            [
                f'<rect x="24" y="{y - 36}" width="992" height="66" rx="10" fill="#ffffff" stroke="#dfe5ef"/>',
                f'<circle cx="58" cy="{y - 3}" r="13" fill="{color}"/>',
                f'<text x="51" y="{y + 3}" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#ffffff">{index + 1}</text>',
                f'<text x="88" y="{y - 11}" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#172033">{run}</text>',
                f'<text x="88" y="{y + 13}" font-family="Arial, sans-serif" font-size="12.5" fill="#536071">{lesson}</text>',
                f'<rect x="602" y="{y - 19}" width="260" height="17" rx="8.5" fill="#e8edf5"/>',
                f'<rect x="602" y="{y - 19}" width="{bar_width(row.get("success_rate"))}" height="17" rx="8.5" fill="{color}"/>',
                f'<text x="878" y="{y - 6}" font-family="Arial, sans-serif" font-size="13" fill="#172033">success {success}</text>',
                f'<text x="602" y="{y + 18}" font-family="Arial, sans-serif" font-size="12" fill="#536071">episodes {episodes} | final distance {distance} | failures {failures}</text>',
            ]
        )
        y += row_height
    parts.append(f'<text x="48" y="{height - 24}" font-family="Arial, sans-serif" font-size="12" fill="#536071">Teaching-scale PushT-style evidence. Not a real-robot benchmark.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_markdown(rows: list[dict[str, Any]], image_path: Path) -> str:
    lines: list[str] = [
        "# LunaVLA Homepage Summary",
        "",
        "This file collects the short result table used for the GitHub homepage.",
        "",
        f"SVG: `{relative(image_path)}`",
        "",
        "## Result Table",
        "",
    ]
    lines.extend(markdown_table(rows))
    lines.extend(
        [
            "",
            "## Evidence Links",
            "",
            "- Extended evaluation: `outputs/extended_evaluation_report.md`",
            "- Project card: `outputs/project_card.md`",
            "- Experiment ledger: `outputs/experiment_ledger.md`",
            "- Submission pack: `outputs/submission_pack/SUBMISSION_README.md`",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_homepage_summary.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows = compact_rows(read_extended_rows(resolve(args.extended_csv)))
    if not rows:
        raise FileNotFoundError("No evaluation rows found. Run `python scripts/run_extended_evaluation.py` first.")
    image_path = resolve(args.image)
    out_path = resolve(args.out)
    write_svg(image_path, rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(rows, image_path), encoding="utf-8")
    print(f"homepage summary: {out_path}")
    print(f"homepage result card: {image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
