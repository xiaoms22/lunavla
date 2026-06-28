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
    parser = argparse.ArgumentParser(description="Run the MiniMind-VLA baseline evidence path.")
    parser.add_argument("--config", default="configs/act_pusht_baseline.yaml", help="Baseline config path.")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes for the baseline report.")
    parser.add_argument("--asset-dir", default="images", help="Directory for README-visible assets.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def expected_outputs(run_dir: Path, asset_dir: Path) -> list[Path]:
    return [
        run_dir / "checkpoint.pt",
        run_dir / "eval_summary.json",
        run_dir / "summary_report.md",
        run_dir / "project_report.md",
        run_dir / "web_demo.html",
        asset_dir / "pusht_rollout.gif",
        asset_dir / "act_action_chunk.gif",
        asset_dir / "loss_curve.gif",
        asset_dir / "rollout_demo.png",
        asset_dir / "loss_curve_baseline.png",
        asset_dir / "result_table.svg",
    ]


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_yaml(config_path)
    run_dir = resolve(config["artifacts"]["output_dir"])
    checkpoint_name = config["artifacts"].get("checkpoint_name", "checkpoint.pt")
    checkpoint_path = run_dir / checkpoint_name
    asset_dir = resolve(args.asset_dir)
    python = sys.executable

    run([python, "trainer/train_act_pusht.py", "--config", str(config_path.relative_to(ROOT).as_posix())])
    run(
        [
            python,
            "eval_vla.py",
            "--checkpoint",
            str(checkpoint_path.relative_to(ROOT).as_posix()),
            "--episodes",
            str(args.episodes),
            "--save-rollouts",
        ]
    )
    run([python, "scripts/summarize_results.py", "--run-dir", str(run_dir.relative_to(ROOT).as_posix())])
    run([python, "scripts/web_demo_vla.py", "--run-dir", str(run_dir.relative_to(ROOT).as_posix())])
    run([python, "scripts/generate_project_report.py", "--run-dir", str(run_dir.relative_to(ROOT).as_posix())])
    run(
        [
            python,
            "scripts/export_readme_assets.py",
            "--run-dir",
            str(run_dir.relative_to(ROOT).as_posix()),
            "--out-dir",
            str(asset_dir.relative_to(ROOT).as_posix()),
        ]
    )

    missing = [path.relative_to(ROOT).as_posix() for path in expected_outputs(run_dir, asset_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing baseline evidence artifacts: " + ", ".join(missing))
    print(f"baseline evidence ready: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
