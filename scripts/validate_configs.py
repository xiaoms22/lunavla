from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.config import ExperimentConfig


DEFAULT_CONFIGS = [
    "configs/act_pusht_cpu_smoke.yaml",
    "configs/bc_pusht_cpu_smoke.yaml",
    "configs/bc_pusht_hidden64_smoke.yaml",
    "configs/act_pusht_jsonl_smoke.yaml",
    "configs/act_pusht_jsonl_noisy_smoke.yaml",
    "configs/act_pusht_baseline.yaml",
    "configs/act_pusht_ablation_chunk_size.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate strict, versioned LunaVLA experiment configs.")
    parser.add_argument("configs", nargs="*", default=DEFAULT_CONFIGS)
    return parser.parse_args()


def resolve(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def validate(path: Path) -> list[str]:
    if not path.is_file():
        return ["file does not exist"]
    try:
        config = ExperimentConfig.load(path)
    except (OSError, TypeError, ValueError) as exc:
        return [str(exc)]
    errors: list[str] = []
    output_dir = Path(str(config.artifacts["output_dir"]))
    if output_dir.is_absolute() or not output_dir.as_posix().startswith("outputs/"):
        errors.append("artifacts.output_dir must be repository-relative under outputs/")
    if config.artifacts["checkpoint_name"] != "checkpoint.json":
        errors.append("artifacts.checkpoint_name must be checkpoint.json")
    report_path = config.artifacts.get("report_path")
    if report_path and not Path(str(report_path)).as_posix().startswith(output_dir.as_posix() + "/"):
        errors.append("artifacts.report_path must be inside artifacts.output_dir")
    if config.dataset["source"] == "jsonl":
        data_path = resolve(str(config.dataset["path"]))
        if not data_path.is_file():
            errors.append(f"dataset.path does not exist: {display(data_path)}")
    return errors


def main() -> int:
    failed = False
    for raw in parse_args().configs:
        path = resolve(raw)
        errors = validate(path)
        print(f"[{'fail' if errors else 'ok'}] {display(path)}")
        for error in errors:
            print(f"  - {error}")
        failed = failed or bool(errors)
    if failed:
        return 1
    print("config validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
