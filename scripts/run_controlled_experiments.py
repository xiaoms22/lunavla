from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
MIN_TRAIN_SEEDS = 5
MIN_EVAL_EPISODES = 20
DEFAULT_SEEDS = [11, 22, 33, 44, 55]
ANALYSIS_SEED = 202611


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the predeclared LunaVLA v1.1 controlled experiment matrix.")
    parser.add_argument("--suite", choices=("chunk", "bc-capacity", "data-quality", "all"), default="all")
    parser.add_argument("--chunk-template", default="configs/act_pusht_baseline.yaml")
    parser.add_argument("--bc-template", default="configs/bc_pusht_cpu_smoke.yaml")
    parser.add_argument("--clean-template", default="configs/act_pusht_jsonl_smoke.yaml")
    parser.add_argument("--noisy-template", default="configs/act_pusht_jsonl_noisy_smoke.yaml")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--eval-episodes", type=int, default=MIN_EVAL_EPISODES)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--out-dir", default="outputs/controlled_v11")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-reduced-design",
        action="store_true",
        help="Allow fewer than 5x20 for development; manifests are marked observational, not controlled.",
    )
    return parser.parse_args()


def resolve(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.setdefault(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {key!r} must be a mapping")
    return value


def set_identity(config: dict[str, Any], policy_type: str, chunk_size: int) -> None:
    policy = ensure_mapping(config, "policy")
    policy["type"] = policy_type
    if "name" in policy:
        policy["name"] = policy_type
    policy["chunk_size"] = int(chunk_size)
    model = config.get("model")
    if isinstance(model, dict):
        model["policy_type"] = policy_type
        model["chunk_size"] = int(chunk_size)
    task = config.get("task")
    if isinstance(task, dict):
        task["id"] = "pusht_style_point_reach"
    else:
        config["task"] = "pusht_style_point_reach"


def prepare_config(
    template: dict[str, Any],
    *,
    family: str,
    treatment: str,
    policy_type: str,
    chunk_size: int,
    train_seed: int,
    eval_seed: int,
    eval_episodes: int,
    run_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    set_identity(config, policy_type, chunk_size)
    config["project_name"] = f"v11-{family}-{treatment}-seed-{train_seed}"
    training = ensure_mapping(config, "training")
    training["device"] = "cpu"
    training["seed"] = int(train_seed)
    evaluation = ensure_mapping(config, "eval")
    evaluation["seed"] = int(eval_seed)
    evaluation["episodes"] = int(eval_episodes)
    evaluation["execution_mode"] = "open_loop_chunk"
    artifacts = ensure_mapping(config, "artifacts")
    artifacts["output_dir"] = relative(run_dir)
    artifacts["checkpoint_name"] = "checkpoint.json"
    artifacts["report_path"] = relative(run_dir / "summary_report.md")
    return config


def design_is_controlled(seeds: list[int], eval_episodes: int, allow_reduced: bool) -> bool:
    enough = len(set(seeds)) >= MIN_TRAIN_SEEDS and eval_episodes >= MIN_EVAL_EPISODES
    if not enough and not allow_reduced:
        raise ValueError(
            f"Controlled evidence requires at least {MIN_TRAIN_SEEDS} training seeds x "
            f"{MIN_EVAL_EPISODES} eval episodes; use --allow-reduced-design only for development"
        )
    return enough


def treatment_specs(args: argparse.Namespace, family: str) -> list[dict[str, Any]]:
    if family == "chunk":
        if sorted(set(args.chunk_sizes)) != [1, 2, 4, 8]:
            raise ValueError("The controlled chunk suite must include exactly chunk sizes 1, 2, 4, and 8")
        template = load_yaml(resolve(args.chunk_template))
        return [
            {
                "name": f"chunk-{size}",
                "template": template,
                "policy_type": "numpy_linear_chunk",
                "chunk_size": size,
                "value": size,
            }
            for size in sorted(args.chunk_sizes)
        ]
    if family == "bc-capacity":
        if len(set(args.hidden_dims)) < 2:
            raise ValueError("BC capacity requires at least two hidden dimensions")
        template = load_yaml(resolve(args.bc_template))
        return [
            {
                "name": f"hidden-{size}",
                "template": template,
                "policy_type": "numpy_bc_mlp",
                "chunk_size": 1,
                "value": size,
            }
            for size in sorted(set(args.hidden_dims))
        ]
    if family == "data-quality":
        clean = load_yaml(resolve(args.clean_template))
        noisy = load_yaml(resolve(args.noisy_template))
        clean_noise = float(ensure_mapping(clean, "dataset").get("action_noise_std", 0.0))
        noisy_noise = float(ensure_mapping(noisy, "dataset").get("action_noise_std", 0.03))
        if noisy_noise <= clean_noise:
            raise ValueError("Noisy template action_noise_std must exceed the clean template")
        # Both treatments derive from the clean base so start distribution, episode seed,
        # split, policy, budget, and evaluation remain identical.
        return [
            {
                "name": "clean",
                "template": clean,
                "policy_type": "numpy_linear_chunk",
                "chunk_size": int(ensure_mapping(clean, "policy").get("chunk_size", 4)),
                "value": clean_noise,
            },
            {
                "name": "noisy",
                "template": clean,
                "policy_type": "numpy_linear_chunk",
                "chunk_size": int(ensure_mapping(clean, "policy").get("chunk_size", 4)),
                "value": noisy_noise,
            },
        ]
    raise ValueError(f"Unknown family: {family}")


def config_for_spec(
    args: argparse.Namespace,
    family: str,
    spec: dict[str, Any],
    train_seed: int,
    run_dir: Path,
    dataset_path: Path | None,
) -> dict[str, Any]:
    config = prepare_config(
        spec["template"],
        family=family,
        treatment=spec["name"],
        policy_type=spec["policy_type"],
        chunk_size=spec["chunk_size"],
        train_seed=train_seed,
        eval_seed=args.eval_seed,
        eval_episodes=args.eval_episodes,
        run_dir=run_dir,
    )
    policy = ensure_mapping(config, "policy")
    if family == "bc-capacity":
        policy["hidden_dim"] = int(spec["value"])
    if family == "data-quality":
        dataset = ensure_mapping(config, "dataset")
        dataset["source"] = "jsonl"
        dataset["path"] = relative(dataset_path) if dataset_path else ""
        dataset["action_noise_std"] = float(spec["value"])
    return config


def normalized_invariants(config: dict[str, Any], family: str) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.pop("project_name", None)
    artifacts = ensure_mapping(normalized, "artifacts")
    artifacts.pop("output_dir", None)
    artifacts.pop("report_path", None)
    training = ensure_mapping(normalized, "training")
    training.pop("seed", None)
    policy = ensure_mapping(normalized, "policy")
    model = normalized.get("model") if isinstance(normalized.get("model"), dict) else {}
    if family == "chunk":
        policy.pop("chunk_size", None)
        model.pop("chunk_size", None)
    elif family == "bc-capacity":
        policy.pop("hidden_dim", None)
    elif family == "data-quality":
        dataset = ensure_mapping(normalized, "dataset")
        dataset.pop("path", None)
        dataset.pop("action_noise_std", None)
    return normalized


def assert_invariants(configs: list[dict[str, Any]], family: str) -> str:
    canonical = [json.dumps(normalized_invariants(config, family), sort_keys=True) for config in configs]
    if len(set(canonical)) != 1:
        raise ValueError(f"Undeclared configuration difference in {family} suite")
    return __import__("hashlib").sha256(canonical[0].encode("utf-8")).hexdigest()


def find_checkpoint(run_dir: Path) -> Path:
    for name in ("checkpoint.json", "checkpoint.pt"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def train_entry(policy_type: str) -> str:
    return "trainer/train_bc_pusht.py" if policy_type == "numpy_bc_mlp" else "trainer/train_act_pusht.py"


def run_one(
    args: argparse.Namespace,
    family: str,
    spec: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    controlled: bool,
) -> None:
    write_yaml(config_path, config)
    python = sys.executable
    run([python, "scripts/validate_configs.py", relative(config_path)])
    run([python, train_entry(spec["policy_type"]), "--config", relative(config_path)])
    checkpoint = find_checkpoint(run_dir)
    run(
        [
            python,
            "eval_vla.py",
            "--checkpoint",
            relative(checkpoint),
            "--episodes",
            str(args.eval_episodes),
            "--output-dir",
            relative(run_dir),
            "--save-rollouts",
        ]
    )
    eval_seeds = [args.eval_seed + index for index in range(args.eval_episodes)]
    command = f"{python} {train_entry(spec['policy_type'])} --config {relative(config_path)}"
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
        "--train-seeds",
        str(config["training"]["seed"]),
        "--eval-seeds",
        *[str(seed) for seed in eval_seeds],
        "--experiment-family",
        family,
        "--command",
        command,
        "--overwrite",
    ]
    if controlled:
        manifest_command.append("--controlled")
    run(manifest_command)


def rollout_rows(run_dir: Path, family: str, treatment: str, train_seed: int, eval_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(sorted((run_dir / "rollouts").glob("*.json"))):
        rollout = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "family": family,
                "treatment": treatment,
                "train_seed": train_seed,
                "eval_seed": eval_seed + index,
                "success": int(bool(rollout.get("success"))),
                "final_distance": float(rollout["final_distance"]),
                "action_smoothness": float(rollout.get("action_smoothness", 0.0)),
                "steps": int(rollout.get("steps", 0)),
            }
        )
    return rows


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [math.nan, math.nan]
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
    return [(centre - spread) / denominator, (centre + spread) / denominator]


def paired_bootstrap(differences: list[float], samples: int, seed: int = ANALYSIS_SEED) -> list[float]:
    if not differences:
        return [math.nan, math.nan]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(rng.choice(differences) for _ in differences) / len(differences))
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return [low, high]


def aggregate(rows: list[dict[str, Any]], treatment: str) -> dict[str, Any]:
    selected = [row for row in rows if row["treatment"] == treatment]
    successes = sum(int(row["success"]) for row in selected)
    trials = len(selected)
    return {
        "treatment": treatment,
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials if trials else None,
        "success_wilson_95": wilson(successes, trials),
        "mean_final_distance": float(np.mean([row["final_distance"] for row in selected])) if selected else None,
        "mean_action_smoothness": float(np.mean([row["action_smoothness"] for row in selected])) if selected else None,
    }


def contrasts(rows: list[dict[str, Any]], treatments: list[str], bootstrap_samples: int) -> list[dict[str, Any]]:
    reference = treatments[0]
    indexed = {
        (row["treatment"], row["train_seed"], row["eval_seed"]): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    keys = sorted((seed, eval_seed) for treatment, seed, eval_seed in indexed if treatment == reference)
    for treatment in treatments[1:]:
        for metric_name in ("final_distance", "action_smoothness"):
            differences = [
                float(indexed[(treatment, seed, eval_seed)][metric_name])
                - float(indexed[(reference, seed, eval_seed)][metric_name])
                for seed, eval_seed in keys
                if (treatment, seed, eval_seed) in indexed
            ]
            output.append(
                {
                    "reference": reference,
                    "treatment": treatment,
                    "metric": metric_name,
                    "paired_n": len(differences),
                    "mean_difference": float(np.mean(differences)) if differences else None,
                    "paired_bootstrap_95": paired_bootstrap(differences, bootstrap_samples),
                    "analysis_seed": ANALYSIS_SEED,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def selected_families(suite: str) -> list[str]:
    return ["chunk", "bc-capacity", "data-quality"] if suite == "all" else [suite]


def main() -> int:
    args = parse_args()
    args.seeds = sorted(set(args.seeds))
    controlled = design_is_controlled(args.seeds, args.eval_episodes, args.allow_reduced_design)
    output_root = resolve(args.out_dir)
    families = selected_families(args.suite)

    matrix: list[tuple[str, dict[str, Any], int, Path, Path, dict[str, Any]]] = []
    design: dict[str, Any] = {
        "controlled": controlled,
        "train_seeds": args.seeds,
        "eval_seeds": [args.eval_seed + index for index in range(args.eval_episodes)],
        "analysis_seed": ANALYSIS_SEED,
        "families": {},
    }
    for family in families:
        specs = treatment_specs(args, family)
        example_configs: list[dict[str, Any]] = []
        for spec in specs:
            dataset_path = output_root / "datasets" / f"{spec['name']}.jsonl" if family == "data-quality" else None
            for train_seed in args.seeds:
                run_dir = output_root / family / spec["name"] / f"seed-{train_seed}"
                config_path = output_root / "configs" / family / f"{spec['name']}-seed-{train_seed}.yaml"
                config = config_for_spec(args, family, spec, train_seed, run_dir, dataset_path)
                matrix.append((family, spec, train_seed, run_dir, config_path, config))
                if train_seed == args.seeds[0]:
                    example_configs.append(config)
        invariant_hash = assert_invariants(example_configs, family)
        design["families"][family] = {
            "treatments": [spec["name"] for spec in specs],
            "invariant_sha256": invariant_hash,
        }

    print(f"design: {len(matrix)} training runs x {args.eval_episodes} fixed evaluation episodes")
    for family, spec, train_seed, run_dir, config_path, _ in matrix:
        print(f"- {family}/{spec['name']}/seed-{train_seed}: {relative(config_path)} -> {relative(run_dir)}")
    if args.dry_run:
        return 0

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to replace {output_root}; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    write_json(output_root / "design.json", design)

    # Generate each clean/noisy dataset once from the same base distribution.
    generated_datasets: set[Path] = set()
    for family, spec, _, _, config_path, config in matrix:
        if family != "data-quality":
            continue
        dataset_path = resolve(config["dataset"]["path"])
        if dataset_path in generated_datasets:
            continue
        write_yaml(config_path, config)
        run(
            [
                sys.executable,
                "scripts/export_pusht_jsonl_dataset.py",
                "--config",
                relative(config_path),
                "--report",
                relative(output_root / "datasets" / f"{spec['name']}_export.md"),
            ]
        )
        generated_datasets.add(dataset_path)

    all_rows: list[dict[str, Any]] = []
    for family, spec, train_seed, run_dir, config_path, config in matrix:
        run_one(args, family, spec, config, config_path, run_dir, controlled)
        rows = rollout_rows(run_dir, family, spec["name"], train_seed, args.eval_seed)
        if len(rows) != args.eval_episodes:
            raise RuntimeError(f"Expected {args.eval_episodes} rollouts in {run_dir}, found {len(rows)}")
        all_rows.extend(rows)

    write_csv(output_root / "per_episode.csv", all_rows)
    for family in families:
        family_rows = [row for row in all_rows if row["family"] == family]
        treatments = design["families"][family]["treatments"]
        summary = {
            "family": family,
            "controlled": controlled,
            "aggregates": [aggregate(family_rows, name) for name in treatments],
            "contrasts": contrasts(family_rows, treatments, args.bootstrap_samples),
        }
        write_json(output_root / family / "summary.json", summary)

    print(f"controlled experiment evidence: {relative(output_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
