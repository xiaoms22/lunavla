from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and summarize the smallest LunaVLA v1.1 teaching loop.")
    parser.add_argument("--config", default="configs/act_pusht_cpu_smoke.yaml")
    parser.add_argument("--out", default="outputs/quickstart_summary.md")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def resolve(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in {config_path}")
    run_dir = resolve(str(config.get("artifacts", {}).get("output_dir", "outputs/cpu_smoke")))

    if not args.skip_run:
        run([sys.executable, "scripts/check_environment.py"])
        command = [sys.executable, "scripts/run_cpu_smoke.py", "--config", relative(config_path)]
        if args.overwrite:
            command.append("--overwrite")
        run(command)

    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    manifest = read_json(run_dir / "manifest.json")
    required = [
        run_dir / "training_summary.json",
        run_dir / "eval_summary.json",
        run_dir / "manifest.json",
        run_dir / "config.resolved.json",
        run_dir / "evidence.json",
    ]
    status = "ready" if all(path.is_file() for path in required) else "needs attention"
    lines = [
        "# LunaVLA quickstart summary",
        "",
        f"Status: `{status}`",
        "",
        "This is a local CPU smoke result, not controlled release evidence.",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| policy | `{manifest.get('policy_id', training.get('policy_name', 'n/a'))}` |",
        f"| task | `{manifest.get('task_id', 'n/a')}` |",
        f"| records | `{training.get('records', 'n/a')}` |",
        f"| chunk size | `{training.get('chunk_size', 'n/a')}` |",
        f"| final loss | `{training.get('final_loss', 'n/a')}` |",
        f"| eval episodes | `{evaluation.get('episodes', 'n/a')}` |",
        f"| success rate | `{evaluation.get('success_rate', 'n/a')}` |",
        f"| mean final distance | `{evaluation.get('mean_final_distance', 'n/a')}` |",
        "",
        "## Inspect next",
        "",
        f"- `{relative(run_dir / 'manifest.json')}` for hashes, seeds, runtime, and command.",
        f"- `{relative(run_dir / 'eval_summary.json')}` for aggregate rollout metrics.",
        f"- `{relative(run_dir / 'rollouts')}` for episode-level behavior.",
        "",
        "To publish results, use the controlled runner and build a validated `results/v1.1` snapshot.",
        "",
    ]
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"quickstart summary: {relative(output)}")
    return 1 if args.strict and status != "ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())
