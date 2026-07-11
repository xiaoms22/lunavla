from __future__ import annotations

import argparse
from numbers import Integral
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lunavla.config import ExperimentConfig as V2ExperimentConfig
from trainer.config import ExperimentConfig as V1ExperimentConfig


DEFAULT_CONFIGS = [
    "configs/act_pusht_cpu_smoke.yaml",
    "configs/bc_pusht_cpu_smoke.yaml",
    "configs/bc_pusht_hidden64_smoke.yaml",
    "configs/act_pusht_jsonl_smoke.yaml",
    "configs/act_pusht_jsonl_noisy_smoke.yaml",
    "configs/act_pusht_baseline.yaml",
    "configs/act_pusht_ablation_chunk_size.yaml",
    "configs/v2/numpy_baseline.yaml",
    "configs/v2/transformer_chunk_cpu.yaml",
    "configs/v2/transformer_visual_cpu.yaml",
    "configs/v2/transformer_visual_state_only_cpu.yaml",
    "configs/v2/transformer_chunk_cuda.yaml",
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
        with path.open("r", encoding="utf-8-sig") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise TypeError("configuration file must contain a mapping")
        raw_schema_version = raw.get("schema_version", 0)
        if isinstance(raw_schema_version, bool) or not isinstance(
            raw_schema_version, Integral
        ):
            raise TypeError("schema_version must be an integer")
        schema_version = int(raw_schema_version)
        config = (
            V2ExperimentConfig.from_mapping(raw)
            if schema_version == 2
            else V1ExperimentConfig.from_mapping(raw)
        )
    except yaml.YAMLError as exc:
        problem = getattr(exc, "problem", None) or "malformed YAML"
        return [f"invalid YAML: {problem}"]
    except (OSError, TypeError, ValueError) as exc:
        return [str(exc)]
    errors: list[str] = []
    output_dir = Path(str(config.artifacts["output_dir"]))
    if output_dir.is_absolute() or not output_dir.as_posix().startswith("outputs/"):
        errors.append("artifacts.output_dir must be repository-relative under outputs/")
    allowed_checkpoints = {"checkpoint.json"} if schema_version == 1 else {
        "checkpoint.json",
        "checkpoint.pt",
    }
    if config.artifacts["checkpoint_name"] not in allowed_checkpoints:
        errors.append(
            "artifacts.checkpoint_name must be checkpoint.json or the v2 PyTorch checkpoint.pt"
        )
    report_path = config.artifacts.get("report_path")
    if report_path and not Path(str(report_path)).as_posix().startswith(output_dir.as_posix() + "/"):
        errors.append("artifacts.report_path must be inside artifacts.output_dir")
    dataset_source = config.dataset.get("source", config.dataset.get("type"))
    if dataset_source == "jsonl":
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
