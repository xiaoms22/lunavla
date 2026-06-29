from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


RUNS = [
    {
        "name": "CPU smoke",
        "run_dir": "outputs/cpu_smoke",
        "config": "configs/act_pusht_cpu_smoke.yaml",
        "command": "python scripts/run_cpu_smoke.py",
    },
    {
        "name": "Baseline",
        "run_dir": "outputs/act_pusht_baseline",
        "config": "configs/act_pusht_baseline.yaml",
        "command": "python scripts/run_baseline_evidence.py",
    },
    {
        "name": "Chunk-size ablation",
        "run_dir": "outputs/act_pusht_ablation_chunk_size",
        "config": "configs/act_pusht_ablation_chunk_size.yaml",
        "command": "python scripts/run_ablation_evidence.py",
    },
]


ARTIFACTS = [
    "checkpoint.pt",
    "training_summary.json",
    "eval_summary.json",
    "summary_report.md",
    "project_report.md",
    "resume_pack.md",
    "run_diagnostic.md",
    "web_demo.html",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a LunaVLA experiment ledger.")
    parser.add_argument("--out", default="outputs/experiment_ledger.md", help="Markdown ledger path.")
    parser.add_argument("--json-out", default="outputs/experiment_ledger.json", help="Machine-readable ledger path.")
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


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{key}:{format_value(item)}" for key, item in sorted(value.items()))
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def config_summary(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    dataset = config.get("dataset", {})
    training = config.get("training", {})
    evaluation = config.get("eval", {})
    artifacts = config.get("artifacts", {})
    return {
        "project_name": config.get("project_name", "n/a"),
        "dataset_source": dataset.get("source", "n/a"),
        "num_episodes": dataset.get("num_episodes", "n/a"),
        "steps_per_episode": dataset.get("steps_per_episode", "n/a"),
        "chunk_size": model.get("chunk_size", config.get("policy", {}).get("chunk_size", "n/a")),
        "num_steps": training.get("num_steps", "n/a"),
        "training_seed": training.get("seed", "n/a"),
        "eval_episodes": evaluation.get("episodes", "n/a"),
        "output_dir": artifacts.get("output_dir", "n/a"),
    }


def artifact_status(run_dir: Path) -> dict[str, str]:
    return {name: "yes" if (run_dir / name).exists() else "no" for name in ARTIFACTS}


def run_record(item: dict[str, str]) -> dict[str, Any]:
    run_dir = resolve(item["run_dir"])
    config_path = resolve(item["config"])
    config = read_yaml(config_path)
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    artifacts = artifact_status(run_dir)
    found = sum(1 for value in artifacts.values() if value == "yes")
    return {
        "name": item["name"],
        "command": item["command"],
        "run_dir": item["run_dir"],
        "config": item["config"],
        "config_sha256": sha256_file(config_path),
        "config_summary": config_summary(config),
        "metrics": {
            "records": training.get("records", "n/a"),
            "chunk_size": training.get("chunk_size", "n/a"),
            "final_loss": training.get("final_loss", "n/a"),
            "episodes": evaluation.get("episodes", "n/a"),
            "success_rate": evaluation.get("success_rate", "n/a"),
            "mean_final_distance": evaluation.get("mean_final_distance", "n/a"),
            "mean_action_smoothness": evaluation.get("mean_action_smoothness", "n/a"),
            "failure_count": evaluation.get("failure_count", "n/a"),
        },
        "artifacts": artifacts,
        "artifact_coverage": f"{found}/{len(ARTIFACTS)}",
        "diagnostic": relative(run_dir / "run_diagnostic.md"),
    }


def build_ledger() -> dict[str, Any]:
    records = [run_record(item) for item in RUNS]
    return {
        "project": "LunaVLA",
        "purpose": "Connect public commands, configs, metrics, and artifacts for reproducible project evidence.",
        "claim_boundary": "teaching-scale PushT-style imitation-learning loop, not real-robot deployment",
        "runs": records,
        "comparison": {
            "report": "outputs/run_comparison.md",
            "csv": "outputs/run_comparison.csv",
            "deltas_csv": "outputs/run_comparison_deltas.csv",
            "report_exists": (ROOT / "outputs/run_comparison.md").exists(),
        },
    }


def run_rows(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in ledger["runs"]:
        metrics = run["metrics"]
        rows.append(
            {
                "run": run["name"],
                "command": f"`{run['command']}`",
                "config": f"`{run['config']}`",
                "config_sha256": run["config_sha256"],
                "artifacts": run["artifact_coverage"],
                "records": metrics["records"],
                "chunk_size": metrics["chunk_size"],
                "final_loss": metrics["final_loss"],
                "success_rate": metrics["success_rate"],
                "mean_final_distance": metrics["mean_final_distance"],
            }
        )
    return rows


def config_rows(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in ledger["runs"]:
        summary = run["config_summary"]
        rows.append(
            {
                "run": run["name"],
                "dataset_source": summary["dataset_source"],
                "num_episodes": summary["num_episodes"],
                "steps_per_episode": summary["steps_per_episode"],
                "training_seed": summary["training_seed"],
                "eval_episodes": summary["eval_episodes"],
                "output_dir": summary["output_dir"],
            }
        )
    return rows


def artifact_rows(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in ledger["runs"]:
        for artifact, exists in run["artifacts"].items():
            rows.append(
                {
                    "run": run["name"],
                    "artifact": f"`{run['run_dir']}/{artifact}`",
                    "exists": exists,
                }
            )
    return rows


def build_markdown(ledger: dict[str, Any]) -> str:
    lines: list[str] = [
        "# LunaVLA Experiment Ledger",
        "",
        "This ledger connects public commands, configs, metrics, and generated artifacts so a learner can audit what was run.",
        "",
        "## Run Audit",
        "",
    ]
    lines.extend(markdown_table(run_rows(ledger)))
    lines.extend(["", "## Config Summary", ""])
    lines.extend(markdown_table(config_rows(ledger)))
    lines.extend(["", "## Artifact Coverage", ""])
    lines.extend(markdown_table(artifact_rows(ledger)))
    lines.extend(
        [
            "",
            "## Comparison Files",
            "",
            f"- Report: `{ledger['comparison']['report']}`",
            f"- CSV: `{ledger['comparison']['csv']}`",
            f"- Delta CSV: `{ledger['comparison']['deltas_csv']}`",
            "",
            "## How To Use This",
            "",
            "- Cite config hashes when you need to show which settings produced a result.",
            "- Cite run diagnostics before writing a resume bullet or interview claim.",
            "- Cite artifact coverage to show the run produced reports, rollout browser output, and metrics.",
            "- Keep the boundary honest: this is teaching-scale imitation-learning evidence, not a real-robot deployment claim.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_experiment_ledger.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    ledger = build_ledger()
    out_path = resolve(args.out)
    json_path = resolve(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(ledger), encoding="utf-8")
    json_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"experiment ledger: {out_path}")
    print(f"experiment ledger json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
