"""Artifact-producing orchestration for a resolved v2 experiment."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, fields
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import numpy as np

from lunavla.ablation import AblationPair, evaluate_action_error_pairs
from lunavla.config import ExperimentConfig
from lunavla.contracts import DatasetSource, Observation, TaskEnv, Transition, VLAPolicy
from lunavla.engine import Engine, EngineConfig, EvaluationResult
from lunavla.evidence import wilson_interval
from lunavla.manifest import RunManifest, git_source_state, sha256_transitions
from lunavla.memory_data import PointReachTaskEnv, make_point_reach_demonstrations
from lunavla.numpy_policy import register_numpy_policies
from lunavla.registry import PolicyRegistry


def _safe_output_dir(root: Path, raw: str) -> Path:
    root = root.resolve()
    relative = Path(raw)
    if relative.is_absolute() or not relative.as_posix().startswith("outputs/"):
        raise ValueError("artifacts.output_dir must be repository-relative under outputs/")
    raw_outputs_root = root / "outputs"
    if raw_outputs_root.is_symlink():
        raise ValueError("repository outputs/ must not be a symbolic link")
    outputs_root = raw_outputs_root.resolve()
    if outputs_root != raw_outputs_root:
        raise ValueError("repository outputs/ must resolve inside the repository")
    target = (root / relative).resolve()
    if target == outputs_root or outputs_root not in target.parents:
        raise ValueError("artifacts.output_dir must be a child of outputs/")
    return target


def _prepare_output(path: Path, *, overwrite: bool) -> Path:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing run directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    return staging


def _commit_output(staging: Path, target: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing run directory: {target}")
        backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
        target.rename(backup)
    try:
        staging.rename(target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


class _TupleDataset:
    def __init__(self, transitions: Iterable[Transition]) -> None:
        self._transitions = tuple(transitions)
        if not self._transitions:
            raise ValueError("experiment dataset cannot be empty")

    def load(self) -> tuple[Transition, ...]:
        return self._transitions


def _language_source(config: ExperimentConfig) -> DatasetSource:
    from lunavla.language_tasks import LanguageTemplateDatasetSource

    return LanguageTemplateDatasetSource(
        "train",
        max_steps=int(config.task["max_steps"]),
        initial_state=None,
        seed=int(config.dataset["seed"]),
        episode_count=int(config.dataset.get("episode_count", 6)),
    )


def _visual_source(config: ExperimentConfig) -> DatasetSource:
    from lunavla.visual_tasks import ObservationMode, RenderedVisualDatasetSource

    image_shape = config.policy.get("image_shape")
    image_size = int(image_shape[0]) if image_shape else int(config.task.get("render_size", 64))
    first_seed = int(config.dataset["seed"])
    seeds = tuple(
        range(first_seed, first_seed + int(config.dataset.get("episode_count", 6)))
    )
    state_only = bool(config.dataset["parameters"].get("state_only", False))
    observation_mode = cast(
        ObservationMode,
        config.dataset["parameters"]["observation_mode"],
    )
    return RenderedVisualDatasetSource(
        seeds=seeds,
        state_only=state_only,
        observation_mode=observation_mode,
        image_size=image_size,
        max_steps=int(config.task["max_steps"]),
    )


def _jsonl_source(config: ExperimentConfig, root: Path) -> DatasetSource:
    from dataset import load_jsonl

    raw_path = Path(str(config.dataset["path"]))
    path = (root / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise ValueError("dataset.path must resolve inside the selected LunaVLA checkout")
    records = load_jsonl(path)
    transitions = [
        Transition(
            observation=Observation(
                np.asarray(record.observation, dtype=np.float32),
                instruction=record.language_instruction,
            ),
            action=np.asarray(record.action, dtype=np.float32),
            reward=-float(record.metadata.get("next_distance_to_goal", 0.0)),
            next_observation=Observation(
                np.asarray(record.next_observation, dtype=np.float32),
                instruction=record.language_instruction,
            ),
            terminated=record.terminated,
            info={
                **record.metadata,
                "episode_id": record.episode_id,
                "timestep": record.timestep,
                "task_id": record.task_id,
                "success": record.success,
            },
        )
        for record in records
    ]
    return _TupleDataset(transitions)


def _dataset_source(config: ExperimentConfig, root: Path) -> DatasetSource:
    source_type = str(config.dataset["type"])
    task_id = str(config.task["id"])
    if source_type in {"memory", "mock_pusht", "generated"}:
        if task_id == "language_conditioned_point_reach":
            return _language_source(config)
        if task_id == "rendered_visual_point_reach":
            return _visual_source(config)
        parameters = config.dataset["parameters"]
        return make_point_reach_demonstrations(
            episodes=int(config.dataset.get("episode_count", 16)),
            steps_per_episode=int(parameters.get("steps_per_episode", 24)),
            seed=int(config.dataset["seed"]),
            action_gain=float(parameters.get("action_gain", 0.35)),
        )
    if source_type == "jsonl":
        return _jsonl_source(config, root)
    if source_type == "lerobot":
        from lunavla.lerobot_adapter import LeRobotDatasetSource

        return LeRobotDatasetSource.from_repo_id(
            str(config.dataset["repo_id"]),
            revision=str(config.dataset["revision"]),
            episodes=config.dataset["episodes"],
            video_backend=str(config.dataset["video_backend"]),
            return_uint8=bool(config.dataset["return_uint8"]),
        )
    raise ValueError(f"unsupported v2 dataset.type: {source_type!r}")


def _task_env(config: ExperimentConfig) -> TaskEnv:
    task_id = str(config.task["id"])
    parameters = config.task["parameters"]
    if task_id == "lerobot_pusht":
        from lunavla.pusht_env_adapter import PushTEnvAdapter

        return PushTEnvAdapter()
    if task_id == "language_conditioned_point_reach":
        from lunavla.language_tasks import LanguageTaskSuiteEnv

        return LanguageTaskSuiteEnv(
            split=str(parameters.get("language_split", "heldout")),  # type: ignore[arg-type]
        )
    if task_id == "rendered_visual_point_reach":
        from lunavla.visual_tasks import ObservationMode, RenderedVisualTaskSuiteEnv

        family = str(config.task["family"])
        families = (
            ("direct_reach", "waypoint_reach") if family == "all" else (family,)
        )
        observation_mode = cast(
            ObservationMode,
            config.dataset["parameters"]["observation_mode"],
        )
        return RenderedVisualTaskSuiteEnv(
            families=families,  # type: ignore[arg-type]
            state_only=bool(config.dataset["parameters"].get("state_only", False)),
            observation_mode=observation_mode,
            image_size=int(config.task.get("render_size", 64)),
        )
    if task_id == "pusht_style_point_reach":
        goal = config.task.get("goal", [0.8, 0.2])
        return PointReachTaskEnv(
            goal=goal,
            start_low=float(parameters.get("start_low", 0.05)),
            start_high=float(parameters.get("start_high", 0.95)),
            action_clip=float(parameters.get("action_clip", 0.12)),
            success_distance=float(parameters.get("success_distance", 0.10)),
        )
    raise ValueError(f"unsupported task.id: {task_id!r}")


def _evaluate_with_cleanup(
    engine: Engine,
    policy: VLAPolicy,
    environment: TaskEnv,
) -> EvaluationResult:
    """Evaluate once and preserve exactly-once cleanup across engine boundaries."""

    class CloseOnceTaskEnv:
        def __init__(self, wrapped: TaskEnv) -> None:
            self._wrapped = wrapped
            self._closed = False

        def reset(self, *, seed: int | None = None) -> Observation:
            return self._wrapped.reset(seed=seed)

        def step(self, action: np.ndarray[Any, Any]) -> Transition:
            return self._wrapped.step(action)

        def close(self) -> None:
            if not self._closed:
                self._closed = True
                self._wrapped.close()

    guarded_environment = CloseOnceTaskEnv(environment)
    try:
        return engine.evaluate(policy, guarded_environment)
    finally:
        guarded_environment.close()


def _registry(config: ExperimentConfig) -> PolicyRegistry:
    registry = PolicyRegistry()
    register_numpy_policies(registry)
    if config.policy["type"] in {"transformer_chunk_cvae", "transformer_chunk", "act"}:
        try:
            from lunavla.transformer_policy import register_transformer_policy
        except ImportError as exc:
            raise RuntimeError(
                "the transformer policy requires the v2-core profile; "
                "run `uv sync --extra v2-core`"
            ) from exc
        register_transformer_policy(registry)
    return registry


def _policy_config(config: ExperimentConfig) -> dict[str, Any]:
    policy_type = str(config.policy["type"])
    if policy_type in {"numpy_linear_chunk", "numpy_bc_mlp"}:
        allowed = {
            "state_dim",
            "instruction_dim",
            "action_dim",
            "chunk_size",
            "hidden_dim",
            "device",
        }
        result = {key: value for key, value in config.policy.items() if key in allowed}
        result["seed"] = int(config.training["seed"])
        return result

    from lunavla.transformer_policy import TransformerPolicyConfig

    allowed = {item.name for item in fields(TransformerPolicyConfig)}
    result = {key: value for key, value in config.policy.items() if key in allowed}
    if "num_layers" in config.policy:
        result["num_encoder_layers"] = int(config.policy["num_layers"])
        result["num_decoder_layers"] = int(config.policy["num_layers"])
    result["seed"] = int(config.training["seed"])
    result["device"] = str(config.training["device"])
    result["kl_weight"] = float(config.training["kl_weight"])
    return result


def _episode_id(transition: Transition) -> int | str:
    value = transition.info.get("episode_id", transition.info.get("episode_index"))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("every transition must declare an integer or string episode id")
    return value


def _episode_ids(transitions: Sequence[Transition]) -> list[int | str]:
    return sorted({_episode_id(item) for item in transitions}, key=lambda value: str(value))


def _split_transitions(
    transitions: Sequence[Transition],
    config: ExperimentConfig,
) -> dict[str, tuple[Transition, ...]]:
    if not transitions:
        raise ValueError("dataset source returned no transitions")
    parameters = config.dataset["parameters"]
    fractions = np.asarray(
        [
            parameters.get("train_fraction", 0.8),
            parameters.get("validation_fraction", 0.1),
            parameters.get("test_fraction", 0.1),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0):
        raise ValueError("dataset split fractions must be finite and non-negative")
    if not np.isclose(float(np.sum(fractions)), 1.0, atol=1e-8):
        raise ValueError("dataset split fractions must sum to one")
    raw_split_seed = parameters.get("split_seed", config.dataset["seed"])
    if isinstance(raw_split_seed, bool) or not isinstance(raw_split_seed, Integral):
        raise TypeError("dataset.parameters.split_seed must be an integer")
    episode_ids = _episode_ids(transitions)
    shuffled = list(np.random.default_rng(int(raw_split_seed)).permutation(episode_ids))
    raw_counts = fractions * len(shuffled)
    counts = np.floor(raw_counts).astype(int)
    remainder_count = len(shuffled) - int(np.sum(counts))
    for index in np.argsort(-(raw_counts - counts))[:remainder_count]:
        counts[index] += 1
    if len(shuffled) >= 3:
        for index in np.flatnonzero((fractions > 0) & (counts == 0)):
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] += 1
    train_end = int(counts[0])
    validation_end = train_end + int(counts[1])
    membership = {
        "train": set(shuffled[:train_end]),
        "validation": set(shuffled[train_end:validation_end]),
        "test": set(shuffled[validation_end:]),
    }
    if any(
        membership[left] & membership[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise AssertionError("episode split construction leaked ids")
    return {
        name: tuple(item for item in transitions if _episode_id(item) in ids)
        for name, ids in membership.items()
    }


def _configured_ablation(
    policy: VLAPolicy,
    config: ExperimentConfig,
) -> dict[str, object] | None:
    language_mode = str(config.evaluation["language_ablation"])
    image_mode = str(config.evaluation["image_ablation"])
    if language_mode != "none" and image_mode != "none":
        raise ValueError("run one language or image ablation at a time")
    bootstrap_samples = int(config.evaluation["parameters"].get("bootstrap_samples", 10_000))
    if bootstrap_samples <= 0:
        raise ValueError("evaluation.parameters.bootstrap_samples must be positive")
    seed = int(config.evaluation["seed"])
    if language_mode != "none":
        if config.task["id"] != "language_conditioned_point_reach":
            raise ValueError("language ablations require language_conditioned_point_reach")
        from lunavla.language_tasks import (
            build_language_examples,
            make_instruction_ablation_pairs,
        )

        language_pairs = make_instruction_ablation_pairs(
            build_language_examples("heldout"),
            language_mode,  # type: ignore[arg-type]
            seed=seed,
        )
        return evaluate_action_error_pairs(
            policy,
            cast(Sequence[AblationPair], language_pairs),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ).to_dict()
    if image_mode in {"occlusion", "shuffle"}:
        if config.task["id"] != "rendered_visual_point_reach":
            raise ValueError("image ablations require rendered_visual_point_reach")
        from lunavla.visual_tasks import build_visual_examples, make_image_ablation_pairs

        image_shape = config.policy.get("image_shape")
        image_size = int(image_shape[0]) if image_shape else int(config.task["render_size"])
        evaluation_seeds = config.evaluation.get("seeds")
        if evaluation_seeds is None:
            evaluation_seeds = range(seed, seed + int(config.evaluation["episodes"]))
        examples = build_visual_examples(
            seeds=tuple(int(value) for value in evaluation_seeds), image_size=image_size
        )
        image_pairs = make_image_ablation_pairs(
            examples,
            image_mode,  # type: ignore[arg-type]
            seed=seed,
        )
        return evaluate_action_error_pairs(
            policy,
            cast(Sequence[AblationPair], image_pairs),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ).to_dict()
    if image_mode == "state_only":
        if config.task["id"] != "rendered_visual_point_reach":
            raise ValueError("state_only requires rendered_visual_point_reach")
        if config.policy.get("image_shape") is not None or not bool(
            config.dataset["parameters"].get("state_only", False)
        ):
            raise ValueError(
                "state_only requires policy.image_shape=null and dataset.parameters.state_only=true"
            )
        return {
            "ablation_mode": "state_only",
            "paired": False,
            "claim_allowed": False,
            "claim_gate": "compare predeclared multi-seed visual and state-only runs",
        }
    return None


def run_experiment(
    config: ExperimentConfig,
    *,
    root: Path,
    overwrite: bool = False,
    require_device: str | None = None,
    require_clean: bool = False,
    command: Sequence[str] = (),
) -> RunManifest:
    """Train/evaluate one config and atomically establish its evidence directory."""

    config = ExperimentConfig.from_mapping(config.to_dict())
    if require_clean:
        dirty, _ = git_source_state(root)
        if dirty:
            raise RuntimeError("--require-clean refuses a run from a dirty or non-Git source tree")
    device = str(config.training["device"])
    if require_device is not None and not device.startswith(require_device):
        raise ValueError(
            f"configuration device {device!r} does not satisfy --require-device "
            f"{require_device!r}"
        )
    final_output_dir = _safe_output_dir(root, str(config.artifacts["output_dir"]))
    output_dir = _prepare_output(final_output_dir, overwrite=overwrite)
    try:
        source = _dataset_source(config, root)
        all_transitions = tuple(source.load())
        splits = _split_transitions(all_transitions, config)
        split_name = str(config.dataset["split"])
        transitions = splits[split_name]
        if not transitions:
            raise ValueError(f"dataset split {split_name!r} contains no episodes")
        engine = Engine(
            EngineConfig(
                device=device,
                seed=int(config.training["seed"]),
                eval_seed=int(config.evaluation["seed"]),
                eval_seeds=(
                    tuple(int(value) for value in config.evaluation["seeds"])
                    if "seeds" in config.evaluation
                    else None
                ),
                batch_size=int(config.training["batch_size"]),
                train_steps=int(config.training["steps"]),
                learning_rate=float(config.training["learning_rate"]),
                execution_mode=str(config.evaluation["execution_mode"]),
                temporal_ensemble_decay=config.policy.get("temporal_ensemble_decay"),
                eval_episodes=int(config.evaluation["episodes"]),
                max_steps=int(config.task["max_steps"]),
            ),
            registry=_registry(config),
        )
        training = engine.train(
            str(config.policy["type"]),
            _TupleDataset(transitions),
            policy_config=_policy_config(config),
        )
        evaluation = _evaluate_with_cleanup(engine, training.policy, _task_env(config))
        ablation = _configured_ablation(training.policy, config)

        checkpoint_path = output_dir / str(config.artifacts["checkpoint_name"])
        checkpoint_path = engine.save_checkpoint(
            training.policy,
            checkpoint_path,
            metadata={"config_sha256": config.sha256()},
        )
        successes = sum(int(item.success) for item in evaluation.episodes)
        success_interval = wilson_interval(successes, len(evaluation.episodes))
        metrics: dict[str, Any] = {
            "final_loss": training.final_loss,
            "success_rate": evaluation.success_rate,
            "success_wilson_95": list(success_interval),
            "mean_reward": evaluation.mean_reward,
            "mean_steps": evaluation.mean_steps,
            "execution_mode": evaluation.execution_mode,
        }
        if ablation is not None:
            metrics["ablation"] = ablation
            ablation_path = output_dir / "ablation.json"
            ablation_path.write_text(
                json.dumps(ablation, indent=2, sort_keys=True), encoding="utf-8"
            )
        resolved_config_path = output_dir / "resolved_config.json"
        resolved_config_path.write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        rollouts_path = output_dir / "rollouts.json"
        rollouts_path.write_text(
            json.dumps([asdict(item) for item in evaluation.episodes], indent=2),
            encoding="utf-8",
        )
        split_ids: dict[str, list[int | str]] = {
            name: _episode_ids(items) for name, items in splits.items()
        }
        manifest = RunManifest.create(
            root=root,
            config=config,
            data_sha256=sha256_transitions(transitions),
            checkpoint_path=checkpoint_path,
            dataset_split=split_ids,
            command=command or ("lunavla-v2", "train"),
            metrics=metrics,
            ablation=ablation,
            artifact_paths={
                "resolved_config.json": resolved_config_path,
                "metrics.json": metrics_path,
                "rollouts.json": rollouts_path,
                **(
                    {"ablation.json": ablation_path}
                    if ablation is not None
                    else {}
                ),
            },
        )
        manifest.write(output_dir / "manifest.json")
        _commit_output(output_dir, final_output_dir, overwrite=overwrite)
        return manifest
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
