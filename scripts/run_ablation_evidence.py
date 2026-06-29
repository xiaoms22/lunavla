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
    parser = argparse.ArgumentParser(description="Run LunaVLA baseline plus chunk-size ablation evidence.")
    parser.add_argument("--baseline-config", default="configs/act_pusht_baseline.yaml", help="Baseline config path.")
    parser.add_argument(
        "--ablation-config",
        default="configs/act_pusht_ablation_chunk_size.yaml",
        help="Ablation config path.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes for each run.")
    parser.add_argument("--out", default="outputs/run_comparison.md", help="Ablation comparison Markdown path.")
    parser.add_argument("--config-diff-out", default="outputs/config_diff.md", help="Config diff Markdown path.")
    parser.add_argument("--asset-dir", default="images", help="Directory for README-visible baseline assets.")
    parser.add_argument("--skip-baseline", action="store_true", help="Reuse an existing baseline run directory.")
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


def expected_run_outputs(run_dir: Path, include_resume: bool = False) -> list[Path]:
    outputs = [
        run_dir / "checkpoint.pt",
        run_dir / "eval_summary.json",
        run_dir / "summary_report.md",
        run_dir / "project_report.md",
        run_dir / "web_demo.html",
    ]
    if include_resume:
        outputs.extend([run_dir / "resume_pack.md", run_dir / "run_diagnostic.md"])
    return outputs


def ensure_outputs(paths: list[Path], label: str) -> None:
    missing = [relative(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} artifacts: " + ", ".join(missing))


def run_eval_report(config_path: Path, episodes: int) -> Path:
    run_dir = run_dir_from_config(config_path)
    checkpoint = checkpoint_from_config(config_path)
    python = sys.executable
    run([python, "trainer/train_act_pusht.py", "--config", relative(config_path)])
    run(
        [
            python,
            "eval_vla.py",
            "--checkpoint",
            relative(checkpoint),
            "--episodes",
            str(episodes),
            "--save-rollouts",
        ]
    )
    run([python, "scripts/summarize_results.py", "--run-dir", relative(run_dir)])
    run([python, "scripts/web_demo_vla.py", "--run-dir", relative(run_dir)])
    run([python, "scripts/generate_project_report.py", "--run-dir", relative(run_dir)])
    run([python, "scripts/generate_resume_pack.py", "--run-dir", relative(run_dir)])
    run([python, "scripts/diagnose_run.py", "--run-dir", relative(run_dir)])
    ensure_outputs(expected_run_outputs(run_dir, include_resume=True), run_dir.name)
    return run_dir


def main() -> int:
    args = parse_args()
    baseline_config = resolve(args.baseline_config)
    ablation_config = resolve(args.ablation_config)
    out_path = resolve(args.out)
    config_diff_out = resolve(args.config_diff_out)
    asset_dir = resolve(args.asset_dir)
    python = sys.executable

    run([python, "scripts/validate_configs.py", relative(baseline_config), relative(ablation_config)])
    run(
        [
            python,
            "scripts/generate_config_diff.py",
            "--baseline",
            relative(baseline_config),
            "--candidate",
            relative(ablation_config),
            "--out",
            relative(config_diff_out),
        ]
    )
    if args.skip_baseline:
        baseline_dir = run_dir_from_config(baseline_config)
        ensure_outputs(expected_run_outputs(baseline_dir), "baseline")
    else:
        run(
            [
                python,
                "scripts/run_baseline_evidence.py",
                "--config",
                relative(baseline_config),
                "--episodes",
                str(args.episodes),
                "--asset-dir",
                relative(asset_dir),
            ]
        )
        baseline_dir = run_dir_from_config(baseline_config)

    ablation_dir = run_eval_report(ablation_config, args.episodes)
    run([python, "scripts/compare_runs.py", "--runs", relative(baseline_dir), relative(ablation_dir), "--out", relative(out_path)])
    run([python, "scripts/generate_resume_pack.py", "--run-dir", relative(baseline_dir), "--comparison", relative(out_path)])
    run([python, "scripts/generate_resume_pack.py", "--run-dir", relative(ablation_dir), "--comparison", relative(out_path)])
    run([python, "scripts/diagnose_run.py", "--run-dir", relative(baseline_dir)])
    run([python, "scripts/diagnose_run.py", "--run-dir", relative(ablation_dir)])

    comparison_outputs = [
        out_path,
        out_path.with_suffix(".csv"),
        out_path.with_name(out_path.stem + "_deltas.csv"),
        config_diff_out,
        config_diff_out.with_suffix(".json"),
        baseline_dir / "resume_pack.md",
        baseline_dir / "run_diagnostic.md",
        ablation_dir / "resume_pack.md",
        ablation_dir / "run_diagnostic.md",
    ]
    ensure_outputs(comparison_outputs, "comparison")
    print(f"ablation evidence ready: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
