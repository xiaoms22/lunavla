from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.trainer_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare clean vs noisy local JSONL demonstrations.")
    parser.add_argument("--clean-config", default="configs/act_pusht_jsonl_smoke.yaml", help="Clean JSONL config.")
    parser.add_argument("--noisy-config", default="configs/act_pusht_jsonl_noisy_smoke.yaml", help="Noisy JSONL config.")
    parser.add_argument("--episodes", type=int, default=None, help="Evaluation episodes. Defaults to each config.")
    parser.add_argument("--out", default="outputs/data_quality_comparison.md", help="Markdown comparison report.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return ", ".join(f"{key}:{value}" for key, value in sorted(value.items())) if value else "none"
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def run_jsonl_config(config_path: Path, label: str, episodes: int | None) -> dict[str, Any]:
    config = load_yaml(config_path)
    run_dir = resolve(config["artifacts"]["output_dir"])
    checkpoint_path = run_dir / config["artifacts"].get("checkpoint_name", "checkpoint.pt")
    config_ref = relative(config_path)
    run_ref = relative(run_dir)
    eval_episodes = episodes if episodes is not None else int(config["eval"].get("episodes", 5))

    run(
        [
            sys.executable,
            "scripts/export_pusht_jsonl_dataset.py",
            "--config",
            config_ref,
            "--report",
            f"outputs/jsonl_{label}_dataset_export.md",
        ]
    )
    run([sys.executable, "scripts/validate_configs.py", config_ref])
    run(
        [
            sys.executable,
            "scripts/inspect_dataset.py",
            "--config",
            config_ref,
            "--out",
            f"outputs/jsonl_{label}_dataset_inspection.md",
        ]
    )
    run([sys.executable, "trainer/train_act_pusht.py", "--config", config_ref])
    run([sys.executable, "eval_vla.py", "--checkpoint", relative(checkpoint_path), "--episodes", str(eval_episodes), "--save-rollouts"])
    run([sys.executable, "scripts/summarize_results.py", "--run-dir", run_ref])
    run([sys.executable, "scripts/web_demo_vla.py", "--run-dir", run_ref])
    run([sys.executable, "scripts/generate_project_report.py", "--run-dir", run_ref, "--title", f"LunaVLA JSONL {label.title()} Data Report"])
    run([sys.executable, "scripts/generate_resume_pack.py", "--run-dir", run_ref])
    run([sys.executable, "scripts/diagnose_run.py", "--run-dir", run_ref])
    run(
        [
            sys.executable,
            "scripts/generate_action_statistics.py",
            "--config",
            config_ref,
            "--run-dir",
            run_ref,
            "--out",
            f"outputs/jsonl_{label}_action_statistics.json",
            "--report",
            f"outputs/jsonl_{label}_action_statistics.md",
        ]
    )

    return {
        "label": label,
        "config": config,
        "config_path": config_path,
        "run_dir": run_dir,
        "data_path": resolve(config["dataset"]["path"]),
    }


def run_row(item: dict[str, Any]) -> dict[str, Any]:
    config = item["config"]
    dataset = config["dataset"]
    run_dir = item["run_dir"]
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    return {
        "label": item["label"],
        "config": relative(item["config_path"]),
        "dataset_path": dataset.get("path", "n/a"),
        "records": training.get("records", "n/a"),
        "episodes": dataset.get("num_episodes", "n/a"),
        "steps": dataset.get("steps_per_episode", "n/a"),
        "seed": dataset.get("seed", "n/a"),
        "start_range": f"{dataset.get('start_low', 0.05)}-{dataset.get('start_high', 0.95)}",
        "action_noise_std": dataset.get("action_noise_std", "n/a"),
        "final_loss": training.get("final_loss", "n/a"),
        "success_rate": evaluation.get("success_rate", "n/a"),
        "mean_final_distance": evaluation.get("mean_final_distance", "n/a"),
        "mean_action_smoothness": evaluation.get("mean_action_smoothness", "n/a"),
        "failure_cases": count_jsonl(run_dir / "failure_cases.jsonl"),
        "report": relative(run_dir / "project_report.md"),
    }


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def delta(clean: dict[str, Any], noisy: dict[str, Any], metric: str) -> str:
    clean_value = as_float(clean.get(metric))
    noisy_value = as_float(noisy.get(metric))
    if clean_value is None or noisy_value is None:
        return "n/a"
    value = noisy_value - clean_value
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.6g}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: list[dict[str, Any]], out_path: Path) -> str:
    clean = rows[0]
    noisy = rows[1] if len(rows) > 1 else {}
    csv_path = out_path.with_suffix(".csv")
    lines: list[str] = [
        "# LunaVLA Data Quality Comparison",
        "",
        "This report compares clean and noisier local JSONL demonstrations with the same policy, training, and evaluation shape.",
        "",
        "## Question",
        "",
        "What changes when the learner keeps the same ACT-style training loop but loads a noisier demonstration file?",
        "",
        "## Runs",
        "",
    ]
    lines.extend(markdown_table(rows))
    lines.extend(
        [
            "",
            f"CSV: `{relative(csv_path)}`",
            "",
            "## Metric Deltas: Noisy Minus Clean",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                {"metric": "success_rate", "delta": delta(clean, noisy, "success_rate"), "direction": "higher is better"},
                {"metric": "mean_final_distance", "delta": delta(clean, noisy, "mean_final_distance"), "direction": "lower is better"},
                {"metric": "mean_action_smoothness", "delta": delta(clean, noisy, "mean_action_smoothness"), "direction": "lower is usually smoother"},
                {"metric": "failure_cases", "delta": delta(clean, noisy, "failure_cases"), "direction": "lower is better"},
            ]
        )
    )
    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- This is a data-quality teaching experiment, not a benchmark claim.",
            "- The clean and noisy configs keep the same model, policy, chunk size, training steps, and eval episodes.",
            "- The changed variables are local JSONL source, random seed, start range, and action noise.",
            "- Inspect both rollout browsers before turning metric differences into a conclusion.",
            "",
            "## Resume-Safe Claim",
            "",
            (
                "I added a local JSONL data-quality comparison that exports clean and noisier PushT-style "
                "demonstrations, reloads both through `dataset.source: jsonl`, trains the same ACT-style policy, "
                "and compares rollout success, final distance, action smoothness, and failure cases."
            ),
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/run_data_quality_comparison.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    clean = run_jsonl_config(resolve(args.clean_config), "clean", args.episodes)
    noisy = run_jsonl_config(resolve(args.noisy_config), "noisy", args.episodes)
    rows = [run_row(clean), run_row(noisy)]

    out_path = resolve(args.out)
    csv_path = out_path.with_suffix(".csv")
    write_csv(csv_path, rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(rows, out_path), encoding="utf-8")

    expected = [
        clean["data_path"],
        noisy["data_path"],
        clean["run_dir"] / "project_report.md",
        noisy["run_dir"] / "project_report.md",
        out_path,
        csv_path,
    ]
    missing = [relative(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing data quality comparison artifacts: " + ", ".join(missing))
    print(f"data quality comparison: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
