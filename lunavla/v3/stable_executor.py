from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .artifacts import sha256_file
from .config import ExperimentConfig
from .diagnostic_engine import DiagnosticRouterV1, typed_episode_key
from .diagnostics import InterventionSpecV1, StateRouteSpecV1
from .engine import EngineV3, dataset_for_config
from .fake_tasks import FakePointEnvV3
from .policy import PolicySampleV3, VLAPolicyV3
from .stable_contracts import StableEvidenceDesignV1, StableEvidenceRowV1
from .stable_workflow import StableExecutionBatchV1


_ROOT = Path(__file__).resolve().parents[2]
_ACT_CONFIG = _ROOT / "configs/v3/act_fake_libero_cpu.yaml"
_DIFFUSION_CONFIG = _ROOT / "configs/v3/diffusion_fake_libero_cpu.yaml"
_DIAGNOSTIC_CONFIG = _ROOT / "configs/v3/diagnostic_act_image.yaml"


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tree_hash(root: Path) -> str:
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise ValueError("checkpoint tree is empty")
    return _stable_hash(files)


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _git_identity() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("stable evidence requires a clean Git worktree")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(sha) != 40:
        raise ValueError("stable evidence requires a full Git SHA")
    return sha


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise TypeError(f"stable executor config must contain a mapping: {path}")
    return dict(value)


def _group_key(
    design: StableEvidenceDesignV1,
    key: tuple[Any, ...],
) -> tuple[str, int, int | None, str | None]:
    policy, train_seed, task_id, _evaluation_id, route, _intervention = key
    if design.study_id == "fixture_policy_ladder":
        return (str(policy), int(train_seed), None, None)
    return (str(policy), int(train_seed), int(task_id), str(route))


def _group_name(group: tuple[str, int, int | None, str | None]) -> str:
    policy, train_seed, task_id, route = group
    return "__".join(
        (
            policy,
            f"seed-{train_seed}",
            "task-none" if task_id is None else f"task-{task_id}",
            "route-none" if route is None else f"route-{route}",
        )
    )


def _physical_seed(task_id: int | None, evaluation_id: int) -> int:
    return 1000 + (0 if task_id is None else 100 * task_id) + evaluation_id


class TeachingFixtureStableExecutor:
    def __init__(self, *, git_sha: str | None = None) -> None:
        self.git_sha = _git_identity() if git_sha is None else git_sha
        if len(self.git_sha) != 40 or any(item not in "0123456789abcdef" for item in self.git_sha):
            raise ValueError("stable executor git_sha must be a full lowercase Git SHA")

    @staticmethod
    def _config(
        design: StableEvidenceDesignV1,
        group: tuple[str, int, int | None, str | None],
    ) -> ExperimentConfig:
        policy, train_seed, task_id, route = group
        if design.study_id == "fixture_policy_ladder":
            template = _ACT_CONFIG if policy == "act_v3" else _DIFFUSION_CONFIG
        else:
            template = _DIAGNOSTIC_CONFIG
        payload = _mapping(template)
        payload["training"]["seed"] = train_seed
        payload["artifacts"]["output_dir"] = "outputs/v3/stable-evidence-cell"
        payload["evaluation"]["episodes"] = 1
        payload["evaluation"]["seeds"] = [1000]
        payload["evaluation"]["seed"] = 1000
        if task_id is not None:
            payload["prompt"]["public_slots"]["stable_task_stratum"] = task_id
        if route is not None:
            payload["routing"]["mode"] = route
        return ExperimentConfig.from_mapping(payload)

    @staticmethod
    def _donor_instructions(
        config: ExperimentConfig,
        task_id: int,
        evaluation_ids: tuple[int, ...],
    ) -> dict[str, tuple[str, str]]:
        records: list[tuple[str, str]] = []
        for evaluation_id in evaluation_ids:
            seed = _physical_seed(task_id, evaluation_id)
            env = FakePointEnvV3(
                str(config.task["id"]),
                int(config.evaluation["max_steps"]),
                str(config.dataset["parameters"].get("instruction_variant", "constant_v1")),
            )
            try:
                observation = env.reset(seed=seed)
            finally:
                env.close()
            if observation.instruction is None:
                raise ValueError("stable prompt shuffle requires instructions")
            records.append((typed_episode_key(observation.episode_id), observation.instruction))
        result: dict[str, tuple[str, str]] = {}
        for index, (recipient, instruction) in enumerate(records):
            for offset in range(1, len(records)):
                donor, donor_instruction = records[(index + offset) % len(records)]
                if donor != recipient and donor_instruction != instruction:
                    result[recipient] = (donor, donor_instruction)
                    break
            if recipient not in result:
                raise ValueError("stable prompt donor bank cannot find different content")
        return result

    @staticmethod
    def _intervention(name: str | None) -> InterventionSpecV1 | None:
        if name is None:
            return None
        return InterventionSpecV1(name, "prompt", name, "rollout", {})

    @staticmethod
    def _rollout(
        engine: EngineV3,
        policy: VLAPolicyV3,
        config: ExperimentConfig,
        *,
        seed: int,
    ) -> tuple[bool, float, float, float | None]:
        env = FakePointEnvV3(
            str(config.task["id"]),
            int(config.evaluation["max_steps"]),
            str(config.dataset["parameters"].get("instruction_variant", "constant_v1")),
        )
        actions_taken: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        first_action_mse: float | None = None
        final_distance = float("inf")
        success = False
        try:
            canonical = env.reset(seed=seed)
            routed = (
                None
                if engine.diagnostic_router is None
                else engine.diagnostic_router.route_observation(canonical, phase="eval")
            )
            observation = canonical if routed is None else routed.observation
            policy.reset(seed)
            history = [observation]
            steps = 0
            while steps < int(config.evaluation["max_steps"]):
                window = history[-policy.spec.history :]
                padding = max(0, policy.spec.history - len(window))
                sample = PolicySampleV3(
                    (window[0],) * padding + tuple(window),
                    np.asarray([False] * padding + [True] * len(window), dtype=bool),
                    None,
                    None,
                    observation.episode_id,
                    observation.step_index,
                )
                chunk = policy.predict_chunk(sample)
                valid = chunk.values[chunk.valid_mask]
                actions = (
                    valid[: policy.spec.execution_steps]
                    if config.evaluation["execution_mode"] == "open_loop_chunk"
                    else valid[:1]
                )
                stop = False
                for action in actions:
                    action_value = np.asarray(action, dtype=np.float32)
                    if first_action_mse is None:
                        state = np.asarray(
                            canonical.state["state.proprioception"], dtype=np.float32
                        )
                        oracle = np.clip(state[2:4] - state[:2], -0.1, 0.1)
                        first_action_mse = float(np.mean((action_value - oracle) ** 2))
                    transition = env.step(action_value)
                    actions_taken.append(np.array(action_value, copy=True))
                    final_distance = float(transition.info["distance"])
                    success = bool(transition.info["success"])
                    canonical = transition.next_observation
                    routed = (
                        None
                        if engine.diagnostic_router is None
                        else engine.diagnostic_router.route_observation(
                            canonical, phase="eval"
                        )
                    )
                    observation = canonical if routed is None else routed.observation
                    history.append(observation)
                    steps += 1
                    if transition.terminated or transition.truncated or steps >= int(
                        config.evaluation["max_steps"]
                    ):
                        stop = True
                        break
                if stop:
                    break
        finally:
            env.close()
        if not np.isfinite(final_distance):
            raise FloatingPointError("stable rollout did not produce a finite final distance")
        smoothness = 0.0
        if len(actions_taken) > 1:
            smoothness = float(
                np.mean(
                    [
                        np.linalg.norm(current - previous)
                        for previous, current in zip(actions_taken, actions_taken[1:])
                    ]
                )
            )
        return success, final_distance, smoothness, first_action_mse

    def execute(
        self,
        design: StableEvidenceDesignV1,
        *,
        only_train_seed: int | None,
        output_dir: Path,
    ) -> StableExecutionBatchV1:
        keys = tuple(
            key
            for key in (
                (
                    policy,
                    train_seed,
                    task_id,
                    evaluation_id,
                    route,
                    intervention,
                )
                for policy in design.policies
                for train_seed in design.train_seeds
                for task_id in (design.task_ids or (None,))
                for evaluation_id in design.evaluation_ids
                for route in (design.routes or (None,))
                for intervention in (design.interventions or (None,))
            )
            if only_train_seed is None or key[1] == only_train_seed
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        rows: list[StableEvidenceRowV1] = []
        checkpoint_hashes: dict[str, str] = {}
        metrics_hashes: dict[str, str] = {}
        groups = tuple(sorted({_group_key(design, key) for key in keys}))
        for group in groups:
            policy_id, train_seed, task_id, route_name = group
            group_dir = output_dir / _group_name(group)
            group_dir.mkdir()
            config = self._config(design, group)
            config_path = _write_json(group_dir / "resolved-config.json", config.to_dict())
            engine = EngineV3(config)
            policy, losses = engine.train(dataset_for_config(config).source("train"))
            if engine.policy_spec is None or engine.normalization is None:
                raise RuntimeError("stable Engine did not establish policy contracts")
            checkpoint_root = group_dir / "checkpoint"
            checkpoint_root.mkdir()
            checkpoint_target = checkpoint_root / str(config.artifacts["checkpoint_name"])
            policy.save_checkpoint(
                checkpoint_target,
                metadata={
                    "stable_design_sha256": design.sha256(),
                    "stable_group": list(group),
                },
            )
            checkpoint_sha256 = _tree_hash(checkpoint_root)
            lock = (
                _ROOT / "requirements-v3-diffusion-cpu.lock"
                if policy_id == "diffusion_v3"
                else _ROOT / "requirements-v3-core-cpu.lock"
            )
            group_metrics_path = _write_json(
                group_dir / "training-metrics.json",
                {
                    "schema_version": 1,
                    "group": list(group),
                    "losses": list(losses),
                    "finite": all(np.isfinite(loss) for loss in losses),
                },
            )
            metrics_sha256 = sha256_file(group_metrics_path)
            manifest_path = _write_json(
                group_dir / "run-manifest.json",
                {
                    "schema_version": 1,
                    "design_sha256": design.sha256(),
                    "git_sha": self.git_sha,
                    "group": list(group),
                    "config_sha256": sha256_file(config_path),
                    "feature_schema_sha256": config.feature_schema.sha256(),
                    "policy_spec_sha256": engine.policy_spec.sha256(),
                    "normalization_sha256": engine.normalization.sha256(),
                    "dependency_lock_sha256": sha256_file(lock),
                    "checkpoint_sha256": checkpoint_sha256,
                    "metrics_sha256": metrics_sha256,
                },
            )
            manifest_sha256 = sha256_file(manifest_path)
            if train_seed == design.repeat_train_seed:
                checkpoint_hashes[_group_name(group)] = checkpoint_sha256
                metrics_hashes[_group_name(group)] = metrics_sha256
            group_keys = tuple(key for key in keys if _group_key(design, key) == group)
            donor_instructions: dict[str, tuple[str, str]] = {}
            if design.study_id == "fixture_prompt_interventions":
                if task_id is None:
                    raise ValueError("prompt intervention groups require a task stratum")
                donor_instructions = self._donor_instructions(
                    config, task_id, design.evaluation_ids
                )
            for key in group_keys:
                _, _, row_task_id, evaluation_id, row_route, intervention = key
                if config.diagnostics["enabled"]:
                    engine.diagnostic_router = DiagnosticRouterV1(
                        config,
                        engine.normalization,
                        route=StateRouteSpecV1(
                            str(row_route), tuple(config.routing["state_features"])
                        ),
                        intervention=self._intervention(intervention),
                        donor_instructions=(
                            donor_instructions if intervention == "shuffle" else None
                        ),
                        counterfactual_transform_id=(
                            "fake_target_swap_v1"
                            if intervention == "counterfactual"
                            else None
                        ),
                    )
                success, final_metric, smoothness, first_action_mse = self._rollout(
                    engine,
                    policy,
                    config,
                    seed=_physical_seed(row_task_id, int(evaluation_id)),
                )
                rows.append(
                    StableEvidenceRowV1(
                        study_id=design.study_id,
                        policy=str(policy_id),
                        train_seed=int(train_seed),
                        task_id=row_task_id,
                        evaluation_id=int(evaluation_id),
                        route=row_route,
                        intervention=intervention,
                        git_sha=self.git_sha,
                        dependency_lock_sha256=sha256_file(lock),
                        upstream_identity_sha256=engine.policy_spec.model_source.sha256(),
                        run_manifest_sha256=manifest_sha256,
                        metrics_sha256=metrics_sha256,
                        success=success,
                        final_metric=final_metric,
                        smoothness=smoothness,
                        first_action_mse=first_action_mse,
                        failure_count=0,
                    )
                )
        _write_json(
            output_dir / "execution-manifest.json",
            {
                "schema_version": 1,
                "design_sha256": design.sha256(),
                "git_sha": self.git_sha,
                "only_train_seed": only_train_seed,
                "rows": len(rows),
                "groups": [_group_name(group) for group in groups],
            },
        )
        return StableExecutionBatchV1(
            tuple(rows),
            _stable_hash(checkpoint_hashes),
            _stable_hash(metrics_hashes),
        )
