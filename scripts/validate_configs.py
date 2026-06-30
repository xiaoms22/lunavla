from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SECTIONS = ["model", "project_name", "framework", "policy", "task", "dataset", "training", "eval", "artifacts"]
SUPPORTED_DATASETS = {"mock_pusht", "jsonl"}
SUPPORTED_POLICIES = {"act", "bc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LunaVLA runnable config files.")
    parser.add_argument(
        "configs",
        nargs="*",
        default=[
            "configs/act_pusht_cpu_smoke.yaml",
            "configs/bc_pusht_cpu_smoke.yaml",
            "configs/act_pusht_jsonl_smoke.yaml",
            "configs/act_pusht_jsonl_noisy_smoke.yaml",
            "configs/act_pusht_baseline.yaml",
            "configs/act_pusht_ablation_chunk_size.yaml",
        ],
        help="Config files to validate.",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def non_negative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value >= 0


def validate_config(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"{relative(path)} does not exist"]

    try:
        config = load_yaml(path)
    except Exception as exc:  # noqa: BLE001 - report friendly config errors.
        return [f"{relative(path)} could not be parsed: {exc}"]

    for section in REQUIRED_SECTIONS:
        require(section in config, f"missing top-level section `{section}`", errors)
    if errors:
        return errors

    model = config["model"]
    policy = config["policy"]
    dataset = config["dataset"]
    training = config["training"]
    eval_config = config["eval"]
    artifacts = config["artifacts"]

    require(isinstance(config["project_name"], str) and bool(config["project_name"]), "`project_name` must be a non-empty string", errors)
    require(config["framework"] == "lunavla", "`framework` should be `lunavla`", errors)
    require(config["task"] == "pusht", "`task` should be `pusht` for current public configs", errors)

    require(model.get("observation_dim") == 4, "`model.observation_dim` must be 4 for the PushT-style record", errors)
    require(model.get("instruction_dim") == 8, "`model.instruction_dim` must be 8 for current instruction features", errors)
    require(model.get("action_dim") == 2, "`model.action_dim` must be 2 for the current action space", errors)
    require(positive_int(model.get("chunk_size")), "`model.chunk_size` must be a positive integer", errors)

    require(policy.get("name") in SUPPORTED_POLICIES, f"`policy.name` must be one of {sorted(SUPPORTED_POLICIES)}", errors)
    require(positive_int(policy.get("chunk_size")), "`policy.chunk_size` must be a positive integer", errors)
    require(
        model.get("chunk_size") == policy.get("chunk_size"),
        "`model.chunk_size` and `policy.chunk_size` must match",
        errors,
    )
    if policy.get("name") == "bc":
        require(policy.get("chunk_size") == 1, "`policy.chunk_size` must be 1 for next-action BC", errors)
        require(positive_int(policy.get("hidden_dim")), "`policy.hidden_dim` must be a positive integer for BC", errors)

    source = dataset.get("source", "mock_pusht")
    require(source in SUPPORTED_DATASETS, f"`dataset.source` must be one of {sorted(SUPPORTED_DATASETS)}", errors)
    if source == "mock_pusht":
        require(positive_int(dataset.get("num_episodes")), "`dataset.num_episodes` must be a positive integer", errors)
        require(positive_int(dataset.get("steps_per_episode")), "`dataset.steps_per_episode` must be a positive integer", errors)
        require(positive_int(dataset.get("seed")), "`dataset.seed` must be a positive integer", errors)
        if "action_noise_std" in dataset:
            require(non_negative_number(dataset.get("action_noise_std")), "`dataset.action_noise_std` must be non-negative", errors)
        if "start_low" in dataset and "start_high" in dataset:
            require(
                non_negative_number(dataset.get("start_low")) and non_negative_number(dataset.get("start_high")) and dataset.get("start_low") < dataset.get("start_high"),
                "`dataset.start_low` must be smaller than `dataset.start_high`",
                errors,
            )
    if source == "jsonl":
        data_path = dataset.get("path")
        require(isinstance(data_path, str) and bool(data_path), "`dataset.path` is required when source is jsonl", errors)
        if isinstance(data_path, str) and data_path:
            require(resolve(data_path).exists(), f"`dataset.path` does not exist: {data_path}", errors)
        if "action_noise_std" in dataset:
            require(non_negative_number(dataset.get("action_noise_std")), "`dataset.action_noise_std` must be non-negative", errors)
        if "start_low" in dataset and "start_high" in dataset:
            require(
                non_negative_number(dataset.get("start_low")) and non_negative_number(dataset.get("start_high")) and dataset.get("start_low") < dataset.get("start_high"),
                "`dataset.start_low` must be smaller than `dataset.start_high`",
                errors,
            )

    require(training.get("device") in {"cpu", "cuda"}, "`training.device` must be `cpu` or `cuda`", errors)
    require(positive_int(training.get("batch_size")), "`training.batch_size` must be a positive integer", errors)
    require(positive_int(training.get("num_steps")), "`training.num_steps` must be a positive integer", errors)
    require(positive_number(training.get("learning_rate")), "`training.learning_rate` must be positive", errors)
    require(isinstance(training.get("seed"), int), "`training.seed` must be an integer", errors)
    require(positive_int(training.get("log_interval")), "`training.log_interval` must be a positive integer", errors)

    require(positive_int(eval_config.get("episodes")), "`eval.episodes` must be a positive integer", errors)
    require(positive_int(eval_config.get("rollout_steps")), "`eval.rollout_steps` must be a positive integer", errors)
    require(non_negative_number(eval_config.get("success_distance")), "`eval.success_distance` must be non-negative", errors)

    output_dir = artifacts.get("output_dir")
    checkpoint_name = artifacts.get("checkpoint_name")
    report_path = artifacts.get("report_path")
    require(isinstance(output_dir, str) and output_dir.startswith("outputs/"), "`artifacts.output_dir` must live under outputs/", errors)
    require(checkpoint_name == "checkpoint.pt", "`artifacts.checkpoint_name` should be checkpoint.pt", errors)
    require(
        isinstance(report_path, str) and isinstance(output_dir, str) and report_path.startswith(output_dir + "/"),
        "`artifacts.report_path` must be inside artifacts.output_dir",
        errors,
    )
    return errors


def main() -> int:
    args = parse_args()
    failed = False
    for config_arg in args.configs:
        path = resolve(config_arg)
        errors = validate_config(path)
        if errors:
            failed = True
            print(f"[fail] {relative(path)}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[ok] {relative(path)}")
    if failed:
        raise SystemExit(1)
    print("config validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
