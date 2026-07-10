from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the smallest CPU-only LunaVLA train/eval loop.")
    parser.add_argument("--config", default="configs/act_pusht_cpu_smoke.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace the configured local run.")
    return parser.parse_args()


def resolve(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def find_checkpoint(run_dir: Path, configured_name: str) -> Path:
    for name in (configured_name, "checkpoint.json", "checkpoint.pt"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def main() -> int:
    args = parse_args()
    python = sys.executable
    config_path = resolve(args.config)
    config = load_config(config_path)
    artifact_config = config.get("artifacts", {})
    run_dir = resolve(str(artifact_config.get("output_dir", "outputs/cpu_smoke")))
    checkpoint_name = str(artifact_config.get("checkpoint_name", "checkpoint.json"))

    run([python, "scripts/validate_configs.py", relative(config_path)])
    train_command = [python, "trainer/train_act_pusht.py", "--config", relative(config_path)]
    if args.overwrite:
        train_command.append("--overwrite")
    run(train_command)
    checkpoint = find_checkpoint(run_dir, checkpoint_name)
    run(
        [
            python,
            "eval_vla.py",
            "--checkpoint",
            relative(checkpoint),
            "--output-dir",
            relative(run_dir),
            "--save-rollouts",
        ]
    )
    manifest_command = [
        python,
        "scripts/create_run_manifest.py",
        "--run-dir",
        relative(run_dir),
        "--config",
        relative(config_path),
        "--checkpoint",
        relative(checkpoint),
        "--metrics",
        relative(run_dir / "eval_summary.json"),
        "--experiment-family",
        "cpu-smoke",
        "--overwrite",
    ]
    run(manifest_command)

    expected = [
        checkpoint,
        run_dir / "training_summary.json",
        run_dir / "eval_summary.json",
        run_dir / "manifest.json",
        run_dir / "config.resolved.json",
        run_dir / "evidence.json",
    ]
    missing = [relative(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing smoke artifacts: " + ", ".join(missing))
    if not any((run_dir / "rollouts").glob("*.json")):
        raise FileNotFoundError(f"No rollout JSON files found in {relative(run_dir / 'rollouts')}")
    run([python, "scripts/create_run_manifest.py", "--run-dir", relative(run_dir), "--check"])
    print(f"CPU smoke test passed: {relative(run_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
