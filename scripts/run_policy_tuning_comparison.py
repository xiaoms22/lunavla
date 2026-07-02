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
    parser = argparse.ArgumentParser(description="Run a small BC policy tuning comparison.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/bc_pusht_cpu_smoke.yaml", "configs/bc_pusht_hidden64_smoke.yaml"],
        help="BC configs to train and compare.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes for each run.")
    parser.add_argument("--out", default="outputs/policy_tuning_comparison.md", help="Markdown comparison output.")
    parser.add_argument("--config-diff-out", default="outputs/policy_tuning_config_diff.md", help="Config diff report.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def run_dir_from_config(config_path: Path) -> Path:
    config = load_yaml(config_path)
    return resolve(config["artifacts"]["output_dir"])


def checkpoint_from_config(config_path: Path) -> Path:
    config = load_yaml(config_path)
    run_dir = resolve(config["artifacts"]["output_dir"])
    return run_dir / config["artifacts"].get("checkpoint_name", "checkpoint.pt")


def title_from_config(config_path: Path) -> str:
    config = load_yaml(config_path)
    hidden_dim = config["policy"].get("hidden_dim", "unknown")
    return f"LunaVLA BC Hidden-Dim {hidden_dim} Smoke Report"


def expected_outputs(run_dir: Path) -> list[Path]:
    return [
        run_dir / "checkpoint.pt",
        run_dir / "eval_summary.json",
        run_dir / "summary_report.md",
        run_dir / "project_report.md",
        run_dir / "run_diagnostic.md",
        run_dir / "web_demo.html",
        run_dir / "action_statistics.json",
    ]


def ensure_outputs(paths: list[Path], label: str) -> None:
    missing = [relative(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} artifacts: " + ", ".join(missing))


def run_bc_path(config_path: Path, episodes: int) -> Path:
    python = sys.executable
    run_dir = run_dir_from_config(config_path)
    checkpoint = checkpoint_from_config(config_path)
    run([python, "scripts/validate_configs.py", relative(config_path)])
    run([python, "trainer/train_bc_pusht.py", "--config", relative(config_path)])
    run([python, "eval_vla.py", "--checkpoint", relative(checkpoint), "--episodes", str(episodes), "--save-rollouts"])
    run([python, "scripts/summarize_results.py", "--run-dir", relative(run_dir)])
    run([python, "scripts/web_demo_vla.py", "--run-dir", relative(run_dir)])
    run([python, "scripts/generate_project_report.py", "--run-dir", relative(run_dir), "--title", title_from_config(config_path)])
    run([python, "scripts/generate_resume_pack.py", "--run-dir", relative(run_dir)])
    run([python, "scripts/diagnose_run.py", "--run-dir", relative(run_dir)])
    ensure_outputs(expected_outputs(run_dir), run_dir.name)
    return run_dir


def main() -> int:
    args = parse_args()
    python = sys.executable
    configs = [resolve(config) for config in args.configs]
    if len(configs) < 2:
        raise ValueError("At least two configs are required for a policy tuning comparison.")

    run([python, "scripts/validate_configs.py", *[relative(config) for config in configs]])
    run(
        [
            python,
            "scripts/generate_config_diff.py",
            "--baseline",
            relative(configs[0]),
            "--candidate",
            relative(configs[1]),
            "--out",
            args.config_diff_out,
            "--json-out",
            str(Path(args.config_diff_out).with_suffix(".json").as_posix()),
            "--expected-paths",
            "policy.hidden_dim",
        ]
    )

    run_dirs = [run_bc_path(config, args.episodes) for config in configs]
    out_path = resolve(args.out)
    run(
        [
            python,
            "scripts/compare_runs.py",
            "--runs",
            *[relative(run_dir) for run_dir in run_dirs],
            "--out",
            relative(out_path),
            "--config-diff",
            args.config_diff_out,
        ]
    )
    run([python, "scripts/generate_policy_ladder.py"])

    expected = [
        out_path,
        out_path.with_suffix(".csv"),
        out_path.with_name(out_path.stem + "_deltas.csv"),
        resolve(args.config_diff_out),
        resolve(Path(args.config_diff_out).with_suffix(".json")),
    ]
    ensure_outputs(expected, "policy tuning comparison")
    print(f"policy tuning comparison ready: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
