"""Fail-closed orchestration, verification, and snapshots for v2 evidence studies.

The runner deliberately keeps the study matrix separate from the ordinary
single-run CLI.  Every training job is derived from :class:`EvidenceDesign`,
and every aggregate result remains traceable to a schema-3 ``RunManifest``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from lunavla.config import ExperimentConfig
from lunavla.contracts import Observation
from lunavla.evidence import (
    ClaimDecision,
    EvidenceManifest,
    EvidenceSource,
    EvidenceStatistic,
    hierarchical_paired_bootstrap_interval,
    wilson_interval,
)
from lunavla.evidence_design import EvidenceDesign
from lunavla.manifest import MANIFEST_SCHEMA_VERSION, RunManifest, git_source_state, sha256_file


_FULL_TRAIN_SEEDS = (11, 22, 33, 44, 55)
_FULL_EVAL_SEEDS = tuple(range(1000, 1024))
_FULL_COMMON = {
    "data_seed": 42,
    "split_seed": 42,
    "analysis_seed": 202_611,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "evaluation_episodes": 24,
    "bootstrap_samples": 10_000,
}
_FULL_SUITE = {
    "language": {
        "design_id": "language-alpha2",
        "base_config": "configs/v2/transformer_chunk_cpu.yaml",
        "arms": (
            ("control", "control", "none"),
            ("mask", "intervention", "mask"),
            ("shuffle", "intervention", "shuffle"),
            ("counterfactual", "intervention", "counterfactual"),
        ),
        "dataset_episodes": 96,
        "training_steps": 1_000,
    },
    "visual": {
        "design_id": "visual-beta1",
        "base_config": "configs/v2/transformer_visual_cpu.yaml",
        "arms": (
            ("control", "control", "none"),
            ("occlusion", "intervention", "occlusion"),
            ("shuffle", "intervention", "shuffle"),
            ("state_only", "baseline", "state_only"),
        ),
        "dataset_episodes": 64,
        "training_steps": 1_500,
    },
}
_FULL_METRICS = (
    ("success_rate", "binary", "negative"),
    ("final_distance", "continuous", "positive"),
    ("first_action_mse", "continuous", "positive"),
)
_PAIR_FIELDS = (
    "pair_id",
    "cell_id",
    "run_id",
    "train_seed",
    "eval_seed",
    "arm_id",
    "arm_mode",
    "family",
    "success",
    "final_distance",
    "first_action_mse",
    "total_reward",
    "steps",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _safe_child(root: Path, relative: str | Path) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"unsafe evidence-relative path: {raw}")
    resolved_root = root.resolve()
    resolved = (root / raw).resolve()
    if resolved_root not in resolved.parents:
        raise ValueError(f"evidence path escapes its root: {raw}")
    return resolved


def _repository_root(path: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(path.resolve().parent), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("evidence design is not inside a Git checkout")
    root = Path(result.stdout.strip()).resolve()
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence design must be inside the selected repository") from exc
    return root


def is_full_design(design: EvidenceDesign) -> bool:
    """Return whether a design exactly matches the predeclared release matrix."""

    suite = _FULL_SUITE[design.suite]
    budget = design.budget
    return (
        design.design_id == suite["design_id"]
        and design.base_config == suite["base_config"]
        and design.seeds.train == _FULL_TRAIN_SEEDS
        and design.seeds.data == _FULL_COMMON["data_seed"]
        and design.seeds.split == _FULL_COMMON["split_seed"]
        and design.seeds.evaluation == _FULL_EVAL_SEEDS
        and design.seeds.bootstrap == _FULL_COMMON["analysis_seed"]
        and tuple((arm.id, arm.role, arm.mode) for arm in design.arms) == suite["arms"]
        and tuple(
            (metric.name, metric.kind, metric.direction) for metric in design.metrics
        )
        == _FULL_METRICS
        and budget.dataset_episodes == suite["dataset_episodes"]
        and budget.batch_size == _FULL_COMMON["batch_size"]
        and budget.training_steps == suite["training_steps"]
        and math.isclose(
            budget.learning_rate,
            float(_FULL_COMMON["learning_rate"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and budget.evaluation_episodes == _FULL_COMMON["evaluation_episodes"]
        and budget.bootstrap_samples == _FULL_COMMON["bootstrap_samples"]
    )


def _is_valid_reduced_design(design: EvidenceDesign) -> bool:
    suite = _FULL_SUITE[design.suite]
    budget = design.budget
    return (
        design.design_id == suite["design_id"]
        and design.base_config == suite["base_config"]
        and set(design.seeds.train).issubset(_FULL_TRAIN_SEEDS)
        and set(design.seeds.evaluation).issubset(_FULL_EVAL_SEEDS)
        and design.seeds.data == _FULL_COMMON["data_seed"]
        and design.seeds.split == _FULL_COMMON["split_seed"]
        and design.seeds.bootstrap == _FULL_COMMON["analysis_seed"]
        and tuple((arm.id, arm.role, arm.mode) for arm in design.arms) == suite["arms"]
        and tuple(
            (metric.name, metric.kind, metric.direction) for metric in design.metrics
        )
        == _FULL_METRICS
        and budget.dataset_episodes <= cast(int, suite["dataset_episodes"])
        and budget.batch_size <= _FULL_COMMON["batch_size"]
        and budget.training_steps <= cast(int, suite["training_steps"])
        and budget.learning_rate == _FULL_COMMON["learning_rate"]
        and budget.evaluation_episodes == len(design.seeds.evaluation)
        and budget.bootstrap_samples <= _FULL_COMMON["bootstrap_samples"]
    )


@dataclass(frozen=True)
class EvidenceJob:
    run_id: str
    train_seed: int
    variant: str
    arm_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRunPlan:
    design_id: str
    design_sha256: str
    reduced_design: bool
    expected_training_runs: int
    expected_arm_episodes: int
    reproducibility_run_id: str | None
    jobs: tuple[EvidenceJob, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "design_sha256": self.design_sha256,
            "reduced_design": self.reduced_design,
            "observational": self.reduced_design,
            "claim_allowed": False,
            "expected_training_runs": self.expected_training_runs,
            "expected_arm_episodes": self.expected_arm_episodes,
            "reproducibility_run_id": self.reproducibility_run_id,
            "jobs": [job.to_dict() for job in self.jobs],
        }


def derive_plan(
    design: EvidenceDesign,
    *,
    allow_reduced_design: bool = False,
) -> EvidenceRunPlan:
    """Derive the only permitted training/evaluation matrix from a design."""

    reduced = not is_full_design(design)
    if reduced and not allow_reduced_design:
        raise ValueError(
            "design is smaller than or differs from the fixed release matrix; "
            "pass --allow-reduced-design for observational CI output"
        )
    if reduced and not _is_valid_reduced_design(design):
        raise ValueError(
            "--allow-reduced-design accepts only a smaller canonical matrix; "
            "design identity, arms, metrics, seeds, and learning rate remain fixed"
        )
    control = next(arm for arm in design.arms if arm.role == "control")
    jobs: list[EvidenceJob] = []
    if design.suite == "language":
        arm_ids = tuple(arm.id for arm in design.arms)
        for train_seed in design.seeds.train:
            jobs.append(
                EvidenceJob(
                    run_id=f"{design.design_id}-seed-{train_seed}",
                    train_seed=train_seed,
                    variant="shared",
                    arm_ids=arm_ids,
                )
            )
    else:
        image_arms = tuple(arm.id for arm in design.arms if arm.mode != "state_only")
        baseline_arms = tuple(arm.id for arm in design.arms if arm.mode == "state_only")
        if control.id not in image_arms:
            raise ValueError("visual image jobs must contain the control arm")
        for train_seed in design.seeds.train:
            jobs.append(
                EvidenceJob(
                    run_id=f"{design.design_id}-image-seed-{train_seed}",
                    train_seed=train_seed,
                    variant="image",
                    arm_ids=image_arms,
                )
            )
            if baseline_arms:
                jobs.append(
                    EvidenceJob(
                        run_id=f"{design.design_id}-state-only-seed-{train_seed}",
                        train_seed=train_seed,
                        variant="state_only",
                        arm_ids=baseline_arms,
                    )
                )
    expected_episodes = sum(len(job.arm_ids) for job in jobs) * len(
        design.seeds.evaluation
    )
    return EvidenceRunPlan(
        design_id=design.design_id,
        design_sha256=design.sha256(),
        reduced_design=reduced,
        expected_training_runs=len(jobs),
        expected_arm_episodes=expected_episodes,
        reproducibility_run_id=(
            None
            if reduced
            else next(
                job.run_id
                for job in jobs
                if job.train_seed == 11
                and (design.suite == "language" or job.variant == "image")
            )
        ),
        jobs=tuple(jobs),
    )


def _resolved_job_config(
    design: EvidenceDesign,
    base: ExperimentConfig,
    job: EvidenceJob,
) -> ExperimentConfig:
    payload = base.to_dict()
    payload["project_name"] = job.run_id
    dataset = payload["dataset"]
    training = payload["training"]
    evaluation = payload["evaluation"]
    artifacts = payload["artifacts"]
    assert isinstance(dataset, dict)
    assert isinstance(training, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(artifacts, dict)
    dataset["seed"] = design.seeds.data
    dataset["episode_count"] = design.budget.dataset_episodes
    parameters = dataset.setdefault("parameters", {})
    assert isinstance(parameters, dict)
    parameters["split_seed"] = design.seeds.split
    training.update(
        {
            "seed": job.train_seed,
            "batch_size": design.budget.batch_size,
            "steps": design.budget.training_steps,
            "learning_rate": design.budget.learning_rate,
        }
    )
    evaluation.update(
        {
            "episodes": len(design.seeds.evaluation),
            "seed": design.seeds.evaluation[0],
            "seeds": list(design.seeds.evaluation),
            "language_ablation": "none",
            "image_ablation": "state_only" if job.variant == "state_only" else "none",
            "parameters": {
                **dict(evaluation.get("parameters", {})),
                "bootstrap_samples": design.budget.bootstrap_samples,
            },
        }
    )
    artifacts["output_dir"] = f"{design.output.run_root}/runs/{job.run_id}"
    if job.variant == "state_only":
        policy = payload["policy"]
        assert isinstance(policy, dict)
        policy["image_shape"] = None
        parameters["state_only"] = True
    return ExperimentConfig.from_mapping(payload)


def _validate_canonical_full_config(
    config: ExperimentConfig,
    *,
    suite: str,
    variant: str,
) -> None:
    """Lock every model/device hyperparameter used by a claim-bearing design."""

    policy = config.policy
    training = config.training
    expected_image = suite == "visual" and variant == "image"
    expected_state_dim = 2 if suite == "language" else 3
    expected_instruction_dim = 16 if suite == "language" else 32
    expected_task = (
        "language_conditioned_point_reach"
        if suite == "language"
        else "rendered_visual_point_reach"
    )
    expected_family = "point_reach" if suite == "language" else "all"
    expected_max_steps = 40 if suite == "language" else 24
    dataset_parameters = config.dataset["parameters"]
    expected_dataset_parameters: dict[str, Any] = {"split_seed": 42}
    if suite == "visual":
        expected_dataset_parameters["observation_mode"] = "vision_required"
    if variant == "state_only":
        expected_dataset_parameters["state_only"] = True
    expected_policy_fields = {
        "type",
        "state_dim",
        "instruction_dim",
        "image_shape",
        "action_dim",
        "chunk_size",
        "d_model",
        "nhead",
        "num_layers",
        "latent_dim",
        "dropout",
        "temporal_ensemble_decay",
        "device",
    }
    checks = {
        "policy.type": policy["type"] == "transformer_chunk_cvae",
        "policy.fields": set(policy) == expected_policy_fields,
        "policy.state_dim": policy["state_dim"] == expected_state_dim,
        "policy.instruction_dim": policy["instruction_dim"] == expected_instruction_dim,
        "policy.action_dim": policy["action_dim"] == 2,
        "policy.d_model": policy.get("d_model") == 64,
        "policy.nhead": policy.get("nhead") == 4,
        "policy.num_layers": policy.get("num_layers") == 2,
        "policy.latent_dim": policy.get("latent_dim") == 16,
        "policy.chunk_size": policy["chunk_size"] == 4,
        "policy.dropout": float(policy.get("dropout", -1.0)) == 0.0,
        "policy.temporal_ensemble_decay": float(
            policy.get("temporal_ensemble_decay", -1.0)
        )
        == 0.25,
        "policy.device": policy["device"] == "cpu",
        "training.device": training["device"] == "cpu",
        "training.kl_weight": float(training["kl_weight"]) == 0.01,
        "task.id": config.task["id"] == expected_task,
        "task.fields": set(config.task)
        == (
            {"id", "family", "max_steps", "goal", "parameters"}
            if suite == "language"
            else {"id", "family", "max_steps", "render_size", "parameters"}
        ),
        "task.family": config.task["family"] == expected_family,
        "task.max_steps": config.task["max_steps"] == expected_max_steps,
        "task.render_size": (
            "render_size" not in config.task
            if suite == "language"
            else config.task.get("render_size") == 32
        ),
        "task.language_split": (
            config.task["parameters"].get("language_split") == "heldout"
            if suite == "language"
            else True
        ),
        "task.parameters": config.task["parameters"]
        == ({"language_split": "heldout"} if suite == "language" else {}),
        "task.goal": config.task.get("goal") == [0.8, 0.2]
        if suite == "language"
        else "goal" not in config.task,
        "dataset.type": config.dataset["type"] == "memory",
        "dataset.fields": set(config.dataset)
        == {"type", "split", "seed", "episode_count", "parameters"},
        "dataset.split": config.dataset["split"] == "train",
        "dataset.observation_mode": (
            "observation_mode" not in dataset_parameters
            if suite == "language"
            else dataset_parameters.get("observation_mode") == "vision_required"
        ),
        "dataset.state_only": bool(dataset_parameters.get("state_only", False))
        == (variant == "state_only"),
        "dataset.parameters": dataset_parameters == expected_dataset_parameters,
        "dataset.seed": config.dataset["seed"] == 42,
        "dataset.episode_count": config.dataset["episode_count"]
        == (96 if suite == "language" else 64),
        "training.batch_size": training["batch_size"] == 32,
        "training.fields": set(training)
        == {"device", "seed", "batch_size", "steps", "learning_rate", "kl_weight"},
        "training.steps": training["steps"] == (1_000 if suite == "language" else 1_500),
        "training.learning_rate": float(training["learning_rate"]) == 3e-4,
        "evaluation.execution_mode": config.evaluation["execution_mode"]
        == "receding_horizon",
        "evaluation.fields": set(config.evaluation)
        == {
            "execution_mode",
            "episodes",
            "seed",
            "seeds",
            "language_ablation",
            "image_ablation",
            "parameters",
        },
        "evaluation.ablation": config.evaluation["language_ablation"] == "none"
        and config.evaluation["image_ablation"]
        == ("state_only" if variant == "state_only" else "none"),
        "evaluation.parameters": config.evaluation["parameters"]
        == {"bootstrap_samples": 10_000},
        "evaluation.seeds": config.evaluation["seed"] == 1000
        and config.evaluation["episodes"] == 24
        and tuple(config.evaluation["seeds"]) == _FULL_EVAL_SEEDS,
        "policy.image_shape": (policy.get("image_shape") is not None) == expected_image,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "claim-bearing evidence config differs from the fixed Transformer contract: "
            + ", ".join(failed)
        )


@dataclass(frozen=True)
class _InstructionIntervention:
    replacements: Mapping[str, str | None]

    def apply(
        self,
        observation: Observation,
        *,
        episode_seed: int,
        step_index: int,
    ) -> Observation:
        del episode_seed, step_index
        instruction = observation.instruction or ""
        if instruction not in self.replacements:
            raise ValueError("instruction intervention donor bank has no matching instruction")
        replacement = self.replacements[instruction]
        return Observation(
            state=observation.state,
            instruction=replacement,
            image=observation.image,
        )


@dataclass(frozen=True)
class _ImageIntervention:
    mode: str
    donor_specs: Mapping[int, Mapping[str, Any]]
    image_size: int

    def apply(
        self,
        observation: Observation,
        *,
        episode_seed: int,
        step_index: int,
    ) -> Observation:
        del step_index
        if observation.image is None:
            raise ValueError("image intervention received an image-free observation")
        if self.mode == "occlusion":
            image = np.zeros_like(observation.image)
        elif self.mode == "shuffle":
            from lunavla.visual_tasks import render_point_reach

            try:
                donor = self.donor_specs[episode_seed]
            except KeyError as exc:
                raise ValueError(f"donor bank has no image for eval seed {episode_seed}") from exc
            image = render_point_reach(
                observation.state[:2],
                goal=donor["goal"],
                waypoint=donor["waypoint"],
                family=str(donor["family"]),  # type: ignore[arg-type]
                image_size=self.image_size,
            )
        else:
            raise ValueError(f"unsupported image intervention mode: {self.mode}")
        return Observation(
            state=observation.state,
            instruction=observation.instruction,
            image=np.asarray(image).copy(),
        )


def _language_interventions(
    design: EvidenceDesign,
) -> tuple[dict[str, _InstructionIntervention], dict[str, Any]]:
    from lunavla.language_tasks import build_language_examples, make_instruction_ablation_pairs

    examples = build_language_examples("heldout")
    result: dict[str, _InstructionIntervention] = {}
    bank: dict[str, Any] = {}
    for arm in design.arms:
        if arm.mode == "none":
            continue
        pairs = make_instruction_ablation_pairs(
            examples,
            arm.mode,  # type: ignore[arg-type]
            seed=design.seeds.bootstrap,
        )
        replacements = {
            str(pair.control.observation.instruction): pair.ablated.observation.instruction
            for pair in pairs
        }
        result[arm.id] = _InstructionIntervention(replacements)
        bank[arm.id] = {
            "mode": arm.mode,
            "replacements": replacements,
            "pair_ids": [pair.pair_id for pair in pairs],
        }
    return result, bank


def _visual_interventions(
    design: EvidenceDesign,
    config: ExperimentConfig,
) -> tuple[dict[str, _ImageIntervention], dict[str, Any]]:
    image_shape = config.policy.get("image_shape")
    image_size = int(image_shape[0]) if image_shape else int(config.task["render_size"])
    fixtures = _evaluation_fixture(config, design)["episodes"]
    result: dict[str, _ImageIntervention] = {}
    bank: dict[str, Any] = {}
    for arm in design.arms:
        if arm.mode not in {"occlusion", "shuffle"}:
            continue
        donors: dict[int, Mapping[str, Any]] = {}
        entries: list[dict[str, Any]] = []
        if arm.mode == "shuffle":
            rng = np.random.default_rng(design.seeds.bootstrap)
            for family in ("direct_reach", "waypoint_reach"):
                family_fixtures = [item for item in fixtures if item["family"] == family]
                if len(family_fixtures) < 2:
                    raise ValueError("visual shuffle requires two eval fixtures per family")
                indices = np.arange(len(family_fixtures))
                permutation: np.ndarray[Any, np.dtype[np.int64]] | None = None
                for _ in range(256):
                    candidate = rng.permutation(indices)
                    if np.all(candidate != indices):
                        permutation = candidate
                        break
                if permutation is None:
                    permutation = np.roll(indices, 1)
                for index, donor_index in enumerate(permutation):
                    source = family_fixtures[index]
                    donor = family_fixtures[int(donor_index)]
                    seed = int(source["eval_seed"])
                    donors[seed] = donor
                    entries.append(
                        {
                            "eval_seed": seed,
                            "donor_eval_seed": int(donor["eval_seed"]),
                            "family": family,
                            "goal": donor["goal"],
                            "waypoint": donor["waypoint"],
                        }
                    )
        else:
            entries = [
                {
                    "eval_seed": int(item["eval_seed"]),
                    "family": item["family"],
                    "occlusion": "full_image_zero",
                }
                for item in fixtures
            ]
        if arm.mode == "shuffle" and set(donors) != set(design.seeds.evaluation):
            raise ValueError("visual donor bank does not cover every evaluation seed")
        result[arm.id] = _ImageIntervention(arm.mode, donors, image_size)
        bank[arm.id] = {"mode": arm.mode, "entries": entries}
    return result, bank


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_pair_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _evaluation_fixture(
    config: ExperimentConfig,
    design: EvidenceDesign,
) -> dict[str, Any]:
    from lunavla.run import _task_env

    episodes: list[dict[str, Any]] = []
    for eval_seed in design.seeds.evaluation:
        env = _task_env(config)
        observation = env.reset(seed=eval_seed)
        inner = getattr(env, "_env", None)
        if inner is None:
            raise ValueError("task suite did not expose its selected deterministic fixture")
        if design.suite == "visual":
            episodes.append(
                {
                    "eval_seed": eval_seed,
                    "family": str(inner.spec.family),
                    "start": observation.state[:2].tolist(),
                    "goal": np.asarray(inner.spec.goal).tolist(),
                    "waypoint": np.asarray(inner.spec.waypoint).tolist(),
                }
            )
        else:
            episodes.append(
                {
                    "eval_seed": eval_seed,
                    "task_id": str(inner.spec.task_id),
                    "start": observation.state.tolist(),
                    "target": np.asarray(inner.spec.target).tolist(),
                    "instruction": observation.instruction,
                }
            )
    payload = {
        "suite": design.suite,
        "evaluation_seeds": list(design.seeds.evaluation),
        "episodes": episodes,
    }
    return {**payload, "sha256": _sha256_json(payload)}


def _non_image_pairing_sha256(transitions: Sequence[Any]) -> str:
    rows: list[dict[str, Any]] = []
    for transition in transitions:
        info = {
            key: value
            for key, value in transition.info.items()
            if key not in {"modality", "observation_mode"}
        }
        rows.append(
            {
                "state": transition.observation.state.tolist(),
                "instruction": transition.observation.instruction,
                "action": transition.action.tolist(),
                "reward": transition.reward,
                "next_state": transition.next_observation.state.tolist(),
                "terminated": transition.terminated,
                "info": info,
            }
        )
    return _sha256_json(rows)


def _first_action_mse(
    policy: Any,
    config: ExperimentConfig,
    *,
    episode_seed: int,
    intervention: Any | None,
) -> float:
    """Measure first-action error on the exact rollout initial condition."""

    from lunavla.run import _task_env

    env = _task_env(config)
    observation = env.reset(seed=episode_seed)
    task_id = str(config.task["id"])
    inner = getattr(env, "_env", None)
    if task_id == "language_conditioned_point_reach":
        if inner is None:
            raise ValueError("language task suite did not expose its selected task")
        target = np.clip(
            np.asarray(inner.spec.target, dtype=np.float32)
            - np.asarray(observation.state, dtype=np.float32),
            -float(inner.action_clip),
            float(inner.action_clip),
        ).astype(np.float32)
    elif task_id == "rendered_visual_point_reach":
        if inner is None:
            raise ValueError("visual task suite did not expose its selected task")
        target = np.asarray(inner.expert_action(), dtype=np.float32)
    else:
        raise ValueError("first-action evidence is only defined for language/visual suites")
    policy_observation = observation
    if intervention is not None:
        policy_observation = intervention.apply(
            observation,
            episode_seed=episode_seed,
            step_index=0,
        )
    chunk = policy.predict_chunk(policy_observation)
    valid = np.flatnonzero(chunk.valid_mask)
    if valid.size == 0:
        raise ValueError("policy returned no valid first action")
    prediction = np.asarray(chunk.values[int(valid[0])], dtype=np.float32)
    return float(np.mean((prediction - target) ** 2))


def _execute_job(
    *,
    root: Path,
    run_dir: Path,
    design: EvidenceDesign,
    job: EvidenceJob,
    config: ExperimentConfig,
    command: Sequence[str],
) -> RunManifest:
    # These private helpers are intentionally isolated here while the public
    # train/evaluate API is frozen for RC.  No evidence semantics live in run.py.
    from lunavla.engine import Engine, EngineConfig, ObservationIntervention
    from lunavla.manifest import sha256_transitions
    from lunavla.run import (
        _TupleDataset,
        _dataset_source,
        _episode_ids,
        _policy_config,
        _registry,
        _split_transitions,
        _task_env,
    )

    run_dir.mkdir(parents=True)
    source = _dataset_source(config, root)
    all_transitions = tuple(source.load())
    splits = _split_transitions(all_transitions, config)
    transitions = splits[str(config.dataset["split"])]
    if not transitions:
        raise ValueError("evidence training split is empty")
    engine = Engine(
        EngineConfig(
            device=str(config.training["device"]),
            seed=job.train_seed,
            eval_seed=design.seeds.evaluation[0],
            eval_seeds=design.seeds.evaluation,
            batch_size=design.budget.batch_size,
            train_steps=design.budget.training_steps,
            learning_rate=design.budget.learning_rate,
            execution_mode=str(config.evaluation["execution_mode"]),
            temporal_ensemble_decay=config.policy.get("temporal_ensemble_decay"),
            eval_episodes=len(design.seeds.evaluation),
            max_steps=int(config.task["max_steps"]),
        ),
        registry=_registry(config),
    )
    training = engine.train(
        str(config.policy["type"]),
        _TupleDataset(transitions),
        policy_config=_policy_config(config),
    )
    interventions: dict[str, ObservationIntervention]
    if design.suite == "language":
        language_interventions, donor_bank = _language_interventions(design)
        interventions = dict(language_interventions)
    elif job.variant == "image":
        image_interventions, donor_bank = _visual_interventions(design, config)
        interventions = dict(image_interventions)
    else:
        interventions, donor_bank = {}, {}

    arms_by_id = {arm.id: arm for arm in design.arms}
    rows: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    arm_metrics: dict[str, Any] = {}
    pair_records: dict[str, dict[str, Any]] = {}
    for arm_id in job.arm_ids:
        arm = arms_by_id[arm_id]
        intervention = interventions.get(arm_id)
        evaluation = engine.evaluate(
            training.policy,
            _task_env(config),
            intervention=intervention,
        )
        distances = [item.final_distance for item in evaluation.episodes]
        if any(value is None for value in distances):
            raise ValueError("evidence episodes must report final_distance")
        finite_distances = cast(list[float], distances)
        arm_metrics[arm_id] = {
            "success_rate": evaluation.success_rate,
            "mean_final_distance": float(np.mean(finite_distances)),
            "mean_reward": evaluation.mean_reward,
            "mean_steps": evaluation.mean_steps,
        }
        first_action_errors: list[float] = []
        for episode in evaluation.episodes:
            pair_id = f"{job.train_seed}:{episode.seed}"
            cell_id = f"{pair_id}:{arm_id}"
            first_action_mse = _first_action_mse(
                training.policy,
                config,
                episode_seed=episode.seed,
                intervention=intervention,
            )
            first_action_errors.append(first_action_mse)
            row = {
                "pair_id": pair_id,
                "cell_id": cell_id,
                "run_id": job.run_id,
                "train_seed": job.train_seed,
                "eval_seed": episode.seed,
                "arm_id": arm_id,
                "arm_mode": arm.mode,
                "family": episode.family or episode.task or "all",
                "success": int(episode.success),
                "final_distance": episode.final_distance,
                "first_action_mse": first_action_mse,
                "total_reward": episode.total_reward,
                "steps": episode.steps,
            }
            rows.append(row)
            record = pair_records.setdefault(
                pair_id,
                {
                    "pair_id": pair_id,
                    "train_seed": job.train_seed,
                    "eval_seed": episode.seed,
                    "arms": [],
                },
            )
            cast(list[str], record["arms"]).append(arm_id)
            rollouts.append(
                {
                    **asdict(episode),
                    "arm_id": arm_id,
                    "pair_id": pair_id,
                    "cell_id": cell_id,
                    "first_action_mse": first_action_mse,
                }
            )
        arm_metrics[arm_id]["mean_first_action_mse"] = float(
            np.mean(first_action_errors)
        )

    checkpoint_path = engine.save_checkpoint(
        training.policy,
        run_dir / str(config.artifacts["checkpoint_name"]),
        metadata={"config_sha256": config.sha256(), "design_sha256": design.sha256()},
    )
    resolved_path = _write_json(run_dir / "resolved_config.json", config.to_dict())
    metrics = {
        "final_loss": training.final_loss,
        "arms": arm_metrics,
        "observational": not is_full_design(design),
        "claim_allowed": False,
    }
    metrics_path = _write_json(run_dir / "metrics.json", metrics)
    rollouts_path = _write_json(run_dir / "rollouts.json", rollouts)
    pairs_path = _write_pair_csv(run_dir / "per_pair.csv", rows)
    donor_path = _write_json(run_dir / "donor_bank.json", donor_bank)
    donor_hash = sha256_file(donor_path)
    fixture = _evaluation_fixture(config, design)
    non_image_hash = _non_image_pairing_sha256(transitions)
    split_ids = {name: _episode_ids(items) for name, items in splits.items()}
    manifest = RunManifest.create(
        root=root,
        config=config,
        data_sha256=sha256_transitions(transitions),
        checkpoint_path=checkpoint_path,
        dataset_split=split_ids,
        command=command,
        metrics=metrics,
        artifact_paths={
            "resolved_config.json": resolved_path,
            "metrics.json": metrics_path,
            "rollouts.json": rollouts_path,
            "per_pair.csv": pairs_path,
            "donor_bank.json": donor_path,
        },
        design_id=design.design_id,
        design_sha256=design.sha256(),
        condition={
            "variant": job.variant,
            "train_seed": job.train_seed,
            "arm_ids": list(job.arm_ids),
        },
        eval_fixture={
            "evidence_fixture_sha256": fixture["sha256"],
            "evidence_episodes": fixture["episodes"],
        },
        paired_data={
            "donor_bank_sha256": donor_hash,
            "non_image_pairing_sha256": non_image_hash,
        },
        arms=[arms_by_id[arm_id].to_dict() for arm_id in job.arm_ids],
        pairs=list(pair_records.values()),
        runtime_determinism={"seeded": True},
    )
    manifest.write(run_dir / "manifest.json")
    return manifest


def _read_pair_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _PAIR_FIELDS:
            raise ValueError(f"unexpected per-pair CSV fields in {path}")
        rows: list[dict[str, Any]] = []
        for line, raw in enumerate(reader, start=2):
            try:
                row = {
                    **raw,
                    "train_seed": int(raw["train_seed"]),
                    "eval_seed": int(raw["eval_seed"]),
                    "success": int(raw["success"]),
                    "final_distance": float(raw["final_distance"]),
                    "first_action_mse": float(raw["first_action_mse"]),
                    "total_reward": float(raw["total_reward"]),
                    "steps": int(raw["steps"]),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid per-pair CSV value at {path}:{line}") from exc
            if row["success"] not in {0, 1}:
                raise ValueError(f"success must be 0 or 1 at {path}:{line}")
            for name in ("final_distance", "first_action_mse", "total_reward"):
                if not math.isfinite(float(row[name])):
                    raise ValueError(f"{name} must be finite at {path}:{line}")
            rows.append(row)
    return rows


def _paired_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    control_arm: str,
    treatment_arm: str,
    metric: str,
    family: str | None = None,
) -> tuple[list[float], list[float], list[int], list[int]]:
    index: dict[tuple[int, int, str], float] = {}
    for row in rows:
        if family is not None and row["family"] != family:
            continue
        key = (int(row["train_seed"]), int(row["eval_seed"]), str(row["arm_id"]))
        if key in index:
            raise ValueError(f"duplicate evidence cell: {key}")
        index[key] = float(row[metric])
    pair_keys = sorted(
        (seed, episode)
        for seed, episode, arm_id in index
        if arm_id == control_arm
    )
    control: list[float] = []
    treatment: list[float] = []
    train_seeds: list[int] = []
    episode_ids: list[int] = []
    for train_seed, episode_id in pair_keys:
        treatment_key = (train_seed, episode_id, treatment_arm)
        if treatment_key not in index:
            raise ValueError(f"missing paired treatment cell: {treatment_key}")
        control.append(index[(train_seed, episode_id, control_arm)])
        treatment.append(index[treatment_key])
        train_seeds.append(train_seed)
        episode_ids.append(episode_id)
    return control, treatment, train_seeds, episode_ids


def _aggregate(
    design: EvidenceDesign,
    plan: EvidenceRunPlan,
    manifests: Sequence[RunManifest],
    rows: Sequence[Mapping[str, Any]],
    sources: Sequence[EvidenceSource],
    *,
    reproducibility_verified: bool,
) -> EvidenceManifest:
    statistics: list[EvidenceStatistic] = []
    control = next(arm for arm in design.arms if arm.role == "control")
    interval_support: dict[tuple[str, str, str], bool] = {}
    for arm in design.arms:
        arm_rows = [row for row in rows if row["arm_id"] == arm.id]
        successes = sum(int(row["success"]) for row in arm_rows)
        lower, upper = wilson_interval(successes, len(arm_rows))
        statistics.append(
            EvidenceStatistic(
                statistic_id=f"{arm.id}-success",
                metric="success_rate",
                scope=arm.id,
                method="wilson",
                estimate=successes / len(arm_rows),
                lower=lower,
                upper=upper,
                sample_n=len(arm_rows),
            )
        )
        if arm.id == control.id:
            continue
        families: tuple[str | None, ...] = (None,)
        if design.suite == "visual":
            families += ("direct_reach", "waypoint_reach")
        for family in families:
            scope = "all" if family is None else family
            for metric in ("final_distance", "first_action_mse", "success"):
                values = _paired_values(
                    rows,
                    control_arm=control.id,
                    treatment_arm=arm.id,
                    metric=metric,
                    family=family,
                )
                interval = hierarchical_paired_bootstrap_interval(
                    values[0],
                    values[1],
                    train_seeds=values[2],
                    episode_ids=values[3],
                    metric="success_rate" if metric == "success" else metric,
                    samples=design.budget.bootstrap_samples,
                    seed=design.seeds.bootstrap,
                )
                metric_name = "success_rate" if metric == "success" else metric
                statistic_id = f"{arm.id}-{scope}-{metric_name.replace('_rate', '')}"
                statistics.append(
                    EvidenceStatistic(
                        statistic_id=statistic_id,
                        metric=metric_name,
                        scope=f"{arm.id}:{scope}",
                        method="hierarchical_paired_bootstrap",
                        estimate=interval.mean_difference,
                        lower=interval.lower,
                        upper=interval.upper,
                        sample_n=interval.paired_n,
                        train_seed_n=interval.train_seed_n,
                    )
                )
                supported = (
                    interval.supports("negative")
                    if metric == "success"
                    else interval.supports("positive")
                )
                interval_support[(arm.id, scope, metric_name)] = supported

    expected_cells = {
        (job.train_seed, eval_seed, arm_id)
        for job in plan.jobs
        for eval_seed in design.seeds.evaluation
        for arm_id in job.arm_ids
    }
    actual_cells = {
        (int(row["train_seed"]), int(row["eval_seed"]), str(row["arm_id"]))
        for row in rows
    }
    source_ids = {source.run_id for source in sources}
    manifest_ids = {manifest.run_id for manifest in manifests}
    git_shas = {manifest.git_sha for manifest in manifests}
    dependencies = {_canonical_json(manifest.dependencies) for manifest in manifests}
    cell_ids = [str(row["cell_id"]) for row in rows]
    design_hash = all(
        manifest.design_id == design.design_id
        and manifest.design_sha256 == design.sha256()
        for manifest in manifests
    )
    donor_hashes = all(
        isinstance(manifest.paired_data.get("donor_bank_sha256"), str)
        and len(str(manifest.paired_data["donor_bank_sha256"])) == 64
        for manifest in manifests
    )
    matrix_complete = (
        actual_cells == expected_cells
        and len(rows) == len(expected_cells)
        and len(manifests) == plan.expected_training_runs
    )
    integrity = {
        "design_hash": design_hash,
        "donor_bank_hashes": donor_hashes,
        "file_hashes": source_ids == manifest_ids == {job.run_id for job in plan.jobs},
        "matrix_complete": matrix_complete,
        "reproducibility": reproducibility_verified,
        "single_dependencies": len(dependencies) == 1,
        "single_git_sha": len(git_shas) == 1 and "unknown" not in git_shas,
        "source_clean": all(not manifest.git_dirty for manifest in manifests),
        "unique_cells": len(set(cell_ids)) == len(cell_ids),
    }
    full_gate = not plan.reduced_design and all(integrity.values())
    if design.suite == "language":
        counterfactual = next(
            (arm for arm in design.arms if arm.mode == "counterfactual"), None
        )
        checks = {
            "controlled_full_design": full_gate,
            "counterfactual_distance_worse": bool(
                counterfactual
                and interval_support.get((counterfactual.id, "all", "final_distance"), False)
            ),
            "control_success_advantage": bool(
                counterfactual
                and interval_support.get((counterfactual.id, "all", "success_rate"), False)
            ),
        }
        claim_id = "instruction_following"
        allowed = "Controlled evidence supports instruction-following in the declared task suite."
        denied = "Instruction-following has not yet been established."
    else:
        occlusion = next((arm for arm in design.arms if arm.mode == "occlusion"), None)
        state_only = next((arm for arm in design.arms if arm.mode == "state_only"), None)
        checks = {"controlled_full_design": full_gate}
        for label, candidate_arm in (("occlusion", occlusion), ("state_only", state_only)):
            for scope in ("all", "direct_reach", "waypoint_reach"):
                checks[f"{label}_{scope}_distance_worse"] = bool(
                    candidate_arm
                    and interval_support.get(
                        (candidate_arm.id, scope, "final_distance"), False
                    )
                )
        claim_id = "visual_control_contribution"
        allowed = "Controlled evidence supports a visual-control contribution in both task families."
        denied = "Visual-control contribution has not yet been established."
    if plan.reduced_design:
        checks = {name: False for name in checks}
    claim = ClaimDecision.from_checks(
        claim_id=claim_id,
        checks=checks,
        allowed_statement=allowed,
        denied_statement=denied,
    )
    return EvidenceManifest(
        schema_version=1,
        design_id=design.design_id,
        design_sha256=design.sha256(),
        reduced_design=plan.reduced_design,
        matrix_complete=matrix_complete,
        integrity_checks=tuple(integrity.items()),
        sources=tuple(sources),
        statistics=tuple(statistics),
        claims=(claim,),
    )


def _manifest_sources(output_root: Path, manifests: Sequence[RunManifest]) -> tuple[EvidenceSource, ...]:
    return tuple(
        EvidenceSource(
            run_id=manifest.run_id,
            manifest_sha256=sha256_file(
                _safe_child(output_root, Path("runs") / manifest.run_id / "manifest.json")
            ),
        )
        for manifest in manifests
    )


def _checkpoint_path(run_dir: Path, manifest: RunManifest) -> Path:
    return _safe_child(run_dir, str(manifest.config["artifacts"]["checkpoint_name"]))


def _reproducibility_record(
    output_root: Path,
    plan: EvidenceRunPlan,
) -> dict[str, Any]:
    if plan.reproducibility_run_id is None:
        return {"required": False, "verified": True}
    original_id = plan.reproducibility_run_id
    repeat_id = f"{original_id}-repeat"
    original_dir = _safe_child(output_root, Path("runs") / original_id)
    repeat_dir = _safe_child(output_root, Path("reproducibility") / repeat_id)
    original = RunManifest.verify_run_dir(original_dir)
    repeated = RunManifest.verify_run_dir(repeat_dir)
    checkpoint_hashes = (
        sha256_file(_checkpoint_path(original_dir, original)),
        sha256_file(_checkpoint_path(repeat_dir, repeated)),
    )
    metrics_hashes = (
        sha256_file(original_dir / "metrics.json"),
        sha256_file(repeat_dir / "metrics.json"),
    )
    verified = (
        checkpoint_hashes[0] == checkpoint_hashes[1]
        and metrics_hashes[0] == metrics_hashes[1]
        and original.config_sha256 == repeated.config_sha256
        and original.data_sha256 == repeated.data_sha256
        and original.git_sha == repeated.git_sha
        and original.dependencies == repeated.dependencies
        and not original.git_dirty
        and not repeated.git_dirty
    )
    return {
        "required": True,
        "verified": verified,
        "original_run_id": original_id,
        "repeat_run_id": repeat_id,
        "checkpoint_sha256": list(checkpoint_hashes),
        "metrics_sha256": list(metrics_hashes),
        "config_sha256": [original.config_sha256, repeated.config_sha256],
        "data_sha256": [original.data_sha256, repeated.data_sha256],
    }


def run_evidence_design(
    design_path: str | Path,
    *,
    allow_reduced_design: bool = False,
    command: Sequence[str] = (),
) -> EvidenceManifest:
    """Execute one immutable evidence design and atomically publish its output."""

    source = Path(design_path)
    design = EvidenceDesign.load(source)
    plan = derive_plan(design, allow_reduced_design=allow_reduced_design)
    root = _repository_root(source)
    base_path = _safe_child(root, design.base_config)
    base = ExperimentConfig.load(base_path)
    dirty, _ = git_source_state(root)
    if dirty:
        raise RuntimeError("controlled evidence requires a clean Git source tree")
    target = _safe_child(root, design.output.run_root)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _write_json(staging / "design.yaml", design.to_dict())
        _write_json(staging / "base_config.json", base.to_dict())
        plan_payload = {**plan.to_dict(), "base_config_sha256": base.sha256()}
        _write_json(staging / "plan.json", plan_payload)
        manifests: list[RunManifest] = []
        all_rows: list[dict[str, Any]] = []
        for job in plan.jobs:
            config = _resolved_job_config(design, base, job)
            if not plan.reduced_design:
                _validate_canonical_full_config(
                    config,
                    suite=design.suite,
                    variant=job.variant,
                )
            run_dir = staging / "runs" / job.run_id
            manifests.append(
                _execute_job(
                    root=root,
                    run_dir=run_dir,
                    design=design,
                    job=job,
                    config=config,
                    command=command,
                )
            )
            all_rows.extend(_read_pair_rows(run_dir / "per_pair.csv"))
        if plan.reproducibility_run_id is not None:
            repeat_job = next(
                job for job in plan.jobs if job.run_id == plan.reproducibility_run_id
            )
            repeat_config = _resolved_job_config(design, base, repeat_job)
            repeat_dir = (
                staging
                / "reproducibility"
                / f"{plan.reproducibility_run_id}-repeat"
            )
            _execute_job(
                root=root,
                run_dir=repeat_dir,
                design=design,
                job=repeat_job,
                config=repeat_config,
                command=command,
            )
        reproducibility = _reproducibility_record(staging, plan)
        _write_json(staging / "reproducibility.json", reproducibility)
        if reproducibility["verified"] is not True:
            raise RuntimeError("seed-11 checkpoint/metrics reproducibility sentinel failed")
        aggregate = _aggregate(
            design,
            plan,
            manifests,
            all_rows,
            _manifest_sources(staging, manifests),
            reproducibility_verified=True,
        )
        aggregate.write(staging / "evidence_manifest.json")
        _write_pair_csv(staging / "per_pair.csv", all_rows)
        _write_json(
            staging / "aggregate.json",
            {
                "statistics": [item.to_dict() for item in aggregate.statistics],
                "claims": [item.to_dict() for item in aggregate.claims],
            },
        )
        verify_evidence(staging)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verify_evidence(target)
    return EvidenceManifest.load(target / "evidence_manifest.json")


@dataclass(frozen=True)
class EvidenceVerification:
    design_id: str
    design_sha256: str
    reduced_design: bool
    source_count: int
    arm_episode_count: int
    git_sha: str

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "verified": True}


def verify_evidence(output_root: str | Path) -> EvidenceVerification:
    """Verify hashes, provenance, and exact matrix completeness for an evidence root."""

    root = Path(output_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("evidence output root must be a real directory")
    design = EvidenceDesign.load(root / "design.yaml")
    base_payload = json.loads((root / "base_config.json").read_text(encoding="utf-8"))
    if not isinstance(base_payload, Mapping):
        raise TypeError("base_config.json must contain an object")
    base = ExperimentConfig.from_mapping(base_payload)
    manifest = EvidenceManifest.load(root / "evidence_manifest.json")
    plan = derive_plan(design, allow_reduced_design=True)
    expected_plan = {**plan.to_dict(), "base_config_sha256": base.sha256()}
    recorded_plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if _canonical_json(recorded_plan) != _canonical_json(expected_plan):
        raise ValueError("plan.json does not match the matrix derived from EvidenceDesign")
    if manifest.design_id != design.design_id or manifest.design_sha256 != design.sha256():
        raise ValueError("evidence manifest design identity/hash mismatch")
    if manifest.reduced_design != plan.reduced_design:
        raise ValueError("evidence manifest reduced_design does not match the design")
    if not manifest.matrix_complete:
        raise ValueError("evidence manifest declares an incomplete matrix")
    if not all(passed for _, passed in manifest.integrity_checks):
        raise ValueError("evidence manifest contains a failed integrity check")
    if plan.reduced_design and any(claim.allowed for claim in manifest.claims):
        raise ValueError("reduced evidence must remain observational with claims closed")

    expected_jobs = {job.run_id: job for job in plan.jobs}
    sources = {source.run_id: source for source in manifest.sources}
    if len(sources) != len(manifest.sources):
        raise ValueError("duplicate source run_id in evidence manifest")
    if set(sources) != set(expected_jobs):
        raise ValueError("source manifest set does not match the derived training matrix")
    dependencies: set[bytes] = set()
    git_shas: set[str] = set()
    actual_cells: set[tuple[int, int, str]] = set()
    cell_ids: set[str] = set()
    pair_arms: dict[str, set[str]] = {}
    visual_pairing_hashes: dict[int, set[str]] = {}
    data_hashes_by_variant: dict[str, set[str]] = {}
    splits_by_variant: dict[str, set[bytes]] = {}
    all_non_image_hashes: set[str] = set()
    aggregate_rows: list[dict[str, Any]] = []
    run_manifests: list[RunManifest] = []
    for run_id, job in expected_jobs.items():
        run_dir = _safe_child(root, Path("runs") / run_id)
        run_manifest = RunManifest.verify_run_dir(run_dir)
        run_manifests.append(run_manifest)
        if run_manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("schema 2 run manifests cannot be controlled evidence")
        path = run_dir / "manifest.json"
        if sha256_file(path) != sources[run_id].manifest_sha256:
            raise ValueError(f"source manifest hash mismatch: {run_id}")
        if run_manifest.design_id != design.design_id:
            raise ValueError(f"source design_id mismatch: {run_id}")
        if run_manifest.design_sha256 != design.sha256():
            raise ValueError(f"source design_sha256 mismatch: {run_id}")
        if run_manifest.git_dirty or run_manifest.source_diff_sha256 is not None:
            raise ValueError(f"dirty source manifest is not controlled evidence: {run_id}")
        git_shas.add(run_manifest.git_sha)
        dependencies.add(_canonical_json(run_manifest.dependencies))
        if run_manifest.train_seeds != [job.train_seed]:
            raise ValueError(f"source train seed mismatch: {run_id}")
        if tuple(run_manifest.eval_seeds) != design.seeds.evaluation:
            raise ValueError(f"source evaluation seeds mismatch: {run_id}")
        if run_manifest.data_seeds != [design.seeds.data]:
            raise ValueError(f"source data seed mismatch: {run_id}")
        condition_arms = tuple(run_manifest.condition.get("arm_ids", ()))
        if (
            condition_arms != job.arm_ids
            or run_manifest.condition.get("variant") != job.variant
            or run_manifest.condition.get("train_seed") != job.train_seed
        ):
            raise ValueError(f"source arm matrix mismatch: {run_id}")
        expected_config = _resolved_job_config(design, base, job)
        if run_manifest.config != expected_config.to_dict():
            raise ValueError(f"source resolved config differs from EvidenceDesign: {run_id}")
        if run_manifest.config_sha256 != expected_config.sha256():
            raise ValueError(f"source config_sha256 differs from EvidenceDesign: {run_id}")
        if not plan.reduced_design:
            _validate_canonical_full_config(
                expected_config,
                suite=design.suite,
                variant=job.variant,
            )
        expected_fixture = _evaluation_fixture(expected_config, design)
        if (
            run_manifest.eval_fixture.get("evidence_fixture_sha256")
            != expected_fixture["sha256"]
            or run_manifest.eval_fixture.get("evidence_episodes")
            != expected_fixture["episodes"]
        ):
            raise ValueError(f"source evaluation fixture mismatch: {run_id}")
        donor_path = run_dir / "donor_bank.json"
        donor_hash = sha256_file(donor_path)
        if (
            run_manifest.paired_data.get("donor_bank_sha256") != donor_hash
            or run_manifest.artifact_sha256.get("donor_bank.json") != donor_hash
        ):
            raise ValueError(f"source donor bank hash mismatch: {run_id}")
        non_image_hash = run_manifest.paired_data.get("non_image_pairing_sha256")
        if not isinstance(non_image_hash, str) or len(non_image_hash) != 64:
            raise ValueError(f"source non-image pairing hash is invalid: {run_id}")
        if design.suite == "visual":
            visual_pairing_hashes.setdefault(job.train_seed, set()).add(non_image_hash)
        all_non_image_hashes.add(non_image_hash)
        data_hashes_by_variant.setdefault(job.variant, set()).add(run_manifest.data_sha256)
        splits_by_variant.setdefault(job.variant, set()).add(
            _canonical_json(run_manifest.dataset_split)
        )
        expected_arms = [
            next(arm for arm in design.arms if arm.id == arm_id).to_dict()
            for arm_id in job.arm_ids
        ]
        if run_manifest.arms != expected_arms:
            raise ValueError(f"source manifest arms differ from the design: {run_id}")
        rows = _read_pair_rows(run_dir / "per_pair.csv")
        aggregate_rows.extend(rows)
        expected_pairs = [
            {
                "pair_id": f"{job.train_seed}:{eval_seed}",
                "train_seed": job.train_seed,
                "eval_seed": eval_seed,
                "arms": list(job.arm_ids),
            }
            for eval_seed in design.seeds.evaluation
        ]
        if _canonical_json(run_manifest.pairs) != _canonical_json(expected_pairs):
            raise ValueError(f"source structured pair records mismatch: {run_id}")
        fixture_by_seed = {
            int(item["eval_seed"]): item for item in expected_fixture["episodes"]
        }
        rollout_payload = json.loads((run_dir / "rollouts.json").read_text(encoding="utf-8"))
        if not isinstance(rollout_payload, list):
            raise TypeError(f"source rollouts must contain a list: {run_id}")
        rollouts_by_cell: dict[str, Mapping[str, Any]] = {}
        for rollout in rollout_payload:
            if not isinstance(rollout, Mapping):
                raise TypeError(f"source rollout entries must be objects: {run_id}")
            required_rollout_fields = {
                "cell_id",
                "pair_id",
                "arm_id",
                "success",
                "final_distance",
                "first_action_mse",
                "total_reward",
                "steps",
            }
            if not required_rollout_fields.issubset(rollout):
                raise ValueError(f"source rollout is missing evidence fields: {run_id}")
            cell_id = str(rollout.get("cell_id"))
            if cell_id in rollouts_by_cell:
                raise ValueError(f"duplicate rollout cell_id: {cell_id}")
            rollouts_by_cell[cell_id] = rollout
        for row in rows:
            pair_id = str(row["pair_id"])
            expected_pair = f"{row['train_seed']}:{row['eval_seed']}"
            if pair_id != expected_pair:
                raise ValueError(f"pair_id does not identify a shared episode pair: {pair_id}")
            cell_id = str(row["cell_id"])
            expected_cell = f"{pair_id}:{row['arm_id']}"
            if cell_id != expected_cell:
                raise ValueError(f"cell_id does not identify its pair/arm: {cell_id}")
            if cell_id in cell_ids:
                raise ValueError(f"duplicate evidence cell_id: {cell_id}")
            cell_ids.add(cell_id)
            pair_arms.setdefault(pair_id, set()).add(str(row["arm_id"]))
            arm = next(item for item in design.arms if item.id == row["arm_id"])
            if row["arm_mode"] != arm.mode:
                raise ValueError(f"per-pair arm mode mismatch: {cell_id}")
            fixture = fixture_by_seed[int(row["eval_seed"])]
            expected_family = (
                fixture["family"] if design.suite == "visual" else fixture["task_id"]
            )
            if row["family"] != expected_family:
                raise ValueError(f"per-pair task family mismatch: {cell_id}")
            if row["run_id"] != run_id:
                raise ValueError(f"per-pair run_id mismatch: {cell_id}")
            rollout = rollouts_by_cell.get(cell_id)
            if rollout is None:
                raise ValueError(f"per-pair cell has no hashed rollout: {cell_id}")
            rollout_values = {
                "pair_id": rollout.get("pair_id"),
                "arm_id": rollout.get("arm_id"),
                "success": int(bool(rollout.get("success"))),
                "final_distance": float(rollout["final_distance"]),
                "first_action_mse": float(rollout["first_action_mse"]),
                "total_reward": float(rollout["total_reward"]),
                "steps": int(rollout["steps"]),
            }
            row_values = {
                name: row[name]
                for name in (
                    "pair_id",
                    "arm_id",
                    "success",
                    "final_distance",
                    "first_action_mse",
                    "total_reward",
                    "steps",
                )
            }
            if rollout_values != row_values:
                raise ValueError(f"per-pair values differ from hashed rollout: {cell_id}")
            cell = (int(row["train_seed"]), int(row["eval_seed"]), str(row["arm_id"]))
            if cell in actual_cells:
                raise ValueError(f"duplicate evidence matrix cell: {cell}")
            actual_cells.add(cell)
        if set(rollouts_by_cell) != {str(row["cell_id"]) for row in rows}:
            raise ValueError(f"source rollout cells differ from per-pair cells: {run_id}")
    if len(git_shas) != 1 or "unknown" in git_shas:
        raise ValueError("evidence sources must use one known Git SHA")
    if len(dependencies) != 1:
        raise ValueError("evidence sources must use one dependency set")
    if design.suite == "visual" and any(
        len(values) != 1 for values in visual_pairing_hashes.values()
    ):
        raise ValueError("visual image/state-only non-image training data are not paired")
    if len(all_non_image_hashes) != 1:
        raise ValueError("evidence sources do not share one non-image paired dataset")
    if any(len(values) != 1 for values in data_hashes_by_variant.values()):
        raise ValueError("data_sha256 differs within an evidence training variant")
    if any(len(values) != 1 for values in splits_by_variant.values()):
        raise ValueError("dataset_split differs within an evidence training variant")
    expected_cells = {
        (job.train_seed, eval_seed, arm_id)
        for job in plan.jobs
        for eval_seed in design.seeds.evaluation
        for arm_id in job.arm_ids
    }
    if actual_cells != expected_cells:
        raise ValueError("per-pair rows do not form the complete derived matrix")
    if len(actual_cells) != plan.expected_arm_episodes:
        raise ValueError("arm episode count does not match the derived matrix")
    expected_pair_arms = {arm.id for arm in design.arms}
    expected_pair_ids = {
        f"{train_seed}:{eval_seed}"
        for train_seed in design.seeds.train
        for eval_seed in design.seeds.evaluation
    }
    if set(pair_arms) != expected_pair_ids or any(
        arms != expected_pair_arms for arms in pair_arms.values()
    ):
        raise ValueError("each episode pair must contain exactly the predeclared arms")
    top_rows = _read_pair_rows(root / "per_pair.csv")
    if _canonical_json(top_rows) != _canonical_json(aggregate_rows):
        raise ValueError("aggregate per_pair.csv does not match hashed source rows")
    recorded_reproducibility = json.loads(
        (root / "reproducibility.json").read_text(encoding="utf-8")
    )
    recomputed_reproducibility = _reproducibility_record(root, plan)
    if _canonical_json(recorded_reproducibility) != _canonical_json(
        recomputed_reproducibility
    ) or recomputed_reproducibility["verified"] is not True:
        raise ValueError("reproducibility sentinel is missing, tampered, or failed")
    recomputed = _aggregate(
        design,
        plan,
        run_manifests,
        aggregate_rows,
        tuple(sources[run_id] for run_id in expected_jobs),
        reproducibility_verified=True,
    )
    if _canonical_json(recomputed.to_dict()) != _canonical_json(manifest.to_dict()):
        raise ValueError("evidence statistics or claim gates do not match source rows")
    recorded_aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    expected_aggregate = {
        "statistics": [item.to_dict() for item in recomputed.statistics],
        "claims": [item.to_dict() for item in recomputed.claims],
    }
    if _canonical_json(recorded_aggregate) != _canonical_json(expected_aggregate):
        raise ValueError("aggregate.json does not match recomputed statistics and claims")
    return EvidenceVerification(
        design_id=design.design_id,
        design_sha256=design.sha256(),
        reduced_design=plan.reduced_design,
        source_count=len(sources),
        arm_episode_count=len(actual_cells),
        git_sha=next(iter(git_shas)),
    )


def snapshot_evidence(output_root: str | Path, out: str | Path) -> Path:
    """Copy only a verified, review-sized evidence snapshot (never checkpoints)."""

    source = Path(output_root).resolve()
    verification = verify_evidence(source)
    destination = Path(out).resolve()
    parts = destination.parts
    if not any(
        parts[index : index + 2] == ("results", "v2")
        for index in range(max(0, len(parts) - 1))
    ) or destination.name == "v2":
        raise ValueError("evidence snapshots must be written under results/v2/")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    copied: list[str] = []

    def copy(relative: str | Path, *, target: str | Path | None = None) -> None:
        source_path = _safe_child(source, relative)
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(f"snapshot source must be a real file: {relative}")
        destination_path = _safe_child(staging, target or relative)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)
        copied.append(Path(target or relative).as_posix())

    try:
        for name in (
            "design.yaml",
            "base_config.json",
            "plan.json",
            "reproducibility.json",
            "evidence_manifest.json",
            "aggregate.json",
            "per_pair.csv",
        ):
            copy(name)
        evidence = EvidenceManifest.load(source / "evidence_manifest.json")
        for item in evidence.sources:
            prefix = Path("runs") / item.run_id
            for name in ("manifest.json", "resolved_config.json", "metrics.json", "donor_bank.json"):
                copy(prefix / name)
            rollouts = json.loads((source / prefix / "rollouts.json").read_text(encoding="utf-8"))
            if not isinstance(rollouts, list):
                raise ValueError(f"rollouts.json must contain a list: {item.run_id}")
            sample_path = _safe_child(staging, prefix / "rollouts.sample.json")
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(sample_path, rollouts[:2])
            copied.append((prefix / "rollouts.sample.json").as_posix())
        reproducibility = json.loads(
            (source / "reproducibility.json").read_text(encoding="utf-8")
        )
        if reproducibility.get("required") is True:
            repeat_id = str(reproducibility["repeat_run_id"])
            repeat_prefix = Path("reproducibility") / repeat_id
            for name in ("manifest.json", "metrics.json"):
                copy(repeat_prefix / name)
        forbidden = {".pt", ".pth", ".ckpt", ".safetensors"}
        if any(path.suffix in forbidden or "checkpoint" in path.name for path in staging.rglob("*")):
            raise AssertionError("snapshot unexpectedly contains a checkpoint")
        file_hashes = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        _write_json(
            staging / "snapshot_manifest.json",
            {
                "schema_version": 1,
                "verification": verification.to_dict(),
                "files": file_hashes,
                "source_evidence_manifest_sha256": sha256_file(
                    source / "evidence_manifest.json"
                ),
            },
        )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
