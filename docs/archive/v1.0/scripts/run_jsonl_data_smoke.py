from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.trainer_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the optional LunaVLA JSONL data smoke path.")
    parser.add_argument("--config", default="configs/act_pusht_jsonl_smoke.yaml", help="JSONL smoke config.")
    parser.add_argument("--episodes", type=int, default=None, help="Evaluation episodes. Defaults to config eval.episodes.")
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


def expected_outputs(run_dir: Path, data_path: Path) -> list[Path]:
    return [
        data_path,
        ROOT / "outputs/jsonl_dataset_export.md",
        ROOT / "outputs/jsonl_dataset_inspection.md",
        run_dir / "checkpoint.pt",
        run_dir / "training_summary.json",
        run_dir / "eval_summary.json",
        run_dir / "summary_report.md",
        run_dir / "project_report.md",
        run_dir / "resume_pack.md",
        run_dir / "run_diagnostic.md",
        run_dir / "web_demo.html",
        run_dir / "action_statistics.json",
        ROOT / "outputs/jsonl_action_statistics.json",
        ROOT / "outputs/jsonl_action_statistics.md",
    ]


def main() -> int:
    args = parse_args()
    python = sys.executable
    config_path = resolve(args.config)
    config = load_yaml(config_path)
    run_dir = resolve(config["artifacts"]["output_dir"])
    checkpoint_path = run_dir / config["artifacts"].get("checkpoint_name", "checkpoint.pt")
    data_path = resolve(config["dataset"]["path"])
    episodes = args.episodes if args.episodes is not None else int(config["eval"].get("episodes", 5))
    config_ref = relative(config_path)
    run_ref = relative(run_dir)

    run([python, "scripts/export_pusht_jsonl_dataset.py", "--config", config_ref])
    run([python, "scripts/validate_configs.py", config_ref])
    run([python, "scripts/inspect_dataset.py", "--config", config_ref, "--out", "outputs/jsonl_dataset_inspection.md"])
    run([python, "trainer/train_act_pusht.py", "--config", config_ref])
    run([python, "eval_vla.py", "--checkpoint", relative(checkpoint_path), "--episodes", str(episodes), "--save-rollouts"])
    run([python, "scripts/summarize_results.py", "--run-dir", run_ref])
    run([python, "scripts/web_demo_vla.py", "--run-dir", run_ref])
    run([python, "scripts/generate_project_report.py", "--run-dir", run_ref, "--title", "LunaVLA JSONL Data Smoke Report"])
    run([python, "scripts/generate_resume_pack.py", "--run-dir", run_ref])
    run([python, "scripts/diagnose_run.py", "--run-dir", run_ref])
    run(
        [
            python,
            "scripts/generate_action_statistics.py",
            "--config",
            config_ref,
            "--run-dir",
            run_ref,
            "--out",
            "outputs/jsonl_action_statistics.json",
            "--report",
            "outputs/jsonl_action_statistics.md",
        ]
    )

    missing = [relative(path) for path in expected_outputs(run_dir, data_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing JSONL smoke artifacts: " + ", ".join(missing))
    print(f"JSONL data smoke passed: {run_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
