from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .act_policy import ActPolicyV3
from .artifacts import sha256_file
from .config import ExperimentConfig
from .engine import EngineV3, dataset_for_config
from .normalization import fit_normalization_stats
from .policy import PolicySampleV3
from .v31_contracts import VLMBackendSpecV1
from .v31_evidence import V31EvidenceDesignV1, V31EvidenceRowV1, v31_row_inventory_sha256
from .v31_evidence_workflow import V31ExecutionBatchV1
from .v31_tasks import V31RolloutEnvV1, make_v31_episode
from .v31_vlm import (
    ContentAddressedFrozenFeatureProviderV1,
    FrozenFeatureCacheReaderV1,
    TransformersFrozenExtractor,
)


_ROOT = Path(__file__).resolve().parents[2]


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _git_identity() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_ROOT, text=True
        ).strip()
    )
    return sha, dirty


class RealFrozenV31EvidenceExecutor:
    """Exact 2,400-row executor backed by real features and dynamic rollouts."""

    def __init__(
        self,
        *,
        source_training_root: str | Path,
        repeat_training_root: str | Path,
        backend_spec: str | Path,
        model_root: str | Path,
        base_cache: str | Path,
        processor_sha256: str,
        device_environment_sha256: str,
        max_steps: int = 64,
        online_batch_size: int = 20,
    ) -> None:
        self.source_training_root = Path(source_training_root).resolve()
        self.repeat_training_root = Path(repeat_training_root).resolve()
        self.spec = VLMBackendSpecV1.from_mapping(
            json.loads(Path(backend_spec).read_text(encoding="utf-8"))
        )
        self.model_root = Path(model_root).resolve()
        self.base = FrozenFeatureCacheReaderV1(base_cache)
        if self.base.index.backend_spec_sha256 != self.spec.sha256():
            raise ValueError("real evidence cache and backend spec differ")
        self.extractor = TransformersFrozenExtractor(self.spec, self.model_root)
        if self.extractor.output_dim != self.base.output_dim:
            raise ValueError("real evidence model and feature cache dimensions differ")
        self.processor_sha256 = processor_sha256
        self.device_environment_sha256 = device_environment_sha256
        if isinstance(max_steps, bool) or max_steps != 64:
            raise ValueError("scientific v3.1 evidence fixes max_steps=64")
        self.max_steps = max_steps
        if isinstance(online_batch_size, bool) or online_batch_size != 20:
            raise ValueError("scientific v3.1 evidence fixes online_batch_size=20")
        self.online_batch_size = online_batch_size
        self.git_sha, self.git_dirty = _git_identity()
        self.dependency_hash = sha256_file(_ROOT / "requirements-v3-vlm-cpu.lock")
        self.feature_source_hash = _stable_hash(
            {
                "backend_spec_sha256": self.spec.sha256(),
                "base_cache_binding_sha256": self.base.binding_sha256,
                "processor_sha256": processor_sha256,
                "image_token_layout": self.spec.image_token_layout,
            }
        )

    def _training_root(self, only_seed: int | None) -> Path:
        return self.source_training_root if only_seed is None else self.repeat_training_root

    def _validate_training(self, root: Path, seeds: tuple[int, ...]) -> bool:
        manifest = json.loads((root / "development-manifest.json").read_text())
        if set(manifest) != {
            "schema_version",
            "execution_role",
            "git_sha",
            "git_dirty",
            "feature_source",
            "cache_index_sha256",
            "train_seeds",
            "training_runs",
            "claim_allowed",
            "release_eligible",
            "gate_reasons",
            "runs",
        }:
            raise ValueError("real evidence training manifest fields are invalid")
        expected_runs = len(seeds) * 2
        if (
            manifest["schema_version"] != 1
            or manifest["git_sha"] != self.git_sha
            or manifest["git_dirty"] is not False
            or tuple(manifest["train_seeds"]) != seeds
            or manifest["training_runs"] != expected_runs
            or manifest["cache_index_sha256"] != self.base.index.sha256()
        ):
            raise ValueError("real evidence training generation is not clean or exact")
        runs = manifest["runs"]
        if not isinstance(runs, list) or len(runs) != expected_runs:
            raise ValueError("real evidence training run matrix is incomplete")
        if any(item["steps"] != 100 for item in runs):
            raise ValueError("real evidence training budget must be exactly 100 steps")
        return True

    def _load_policy(self, root: Path, seed: int, arm: str) -> tuple[ActPolicyV3, str]:
        cell = root / f"seed-{seed}" / arm
        config = ExperimentConfig.from_mapping(
            json.loads((cell / "resolved-config.json").read_text())
        )
        if config.training["seed"] != seed or config.training["steps"] != 100:
            raise ValueError("real evidence checkpoint config drifted")
        bundle = dataset_for_config(config)
        engine = EngineV3(config)
        engine.normalization = fit_normalization_stats(
            bundle.select("train"), config.feature_schema
        )
        checkpoint = cell / "act_v3.pt"
        policy = engine.restore_policy(checkpoint)
        if not isinstance(policy, ActPolicyV3):
            raise TypeError("real v3.1 evidence requires ActPolicyV3")
        return policy, sha256_file(checkpoint)

    def _cell(
        self,
        *,
        policy: ActPolicyV3,
        provider: ContentAddressedFrozenFeatureProviderV1 | None,
        intervention: str,
        seed: int,
        arm: str,
        task_id: str,
        stratum: str,
        episodes: int,
        checkpoint_sha256: str,
    ) -> list[V31EvidenceRowV1]:
        if provider is not None:
            policy.configure_rollout_feature_provider(provider, intervention=intervention)
        environments = [
            V31RolloutEnvV1(
                task_id=task_id,
                stratum=stratum,
                data_seed=42,
                episode_index=index,
                max_steps=self.max_steps,
            )
            for index in range(episodes)
        ]
        observations = [
            env.reset(seed=1000 + index) for index, env in enumerate(environments)
        ]
        active = [True] * episodes
        first_mse: list[float | None] = [None] * episodes
        final_distance = [float("inf")] * episodes
        success = [False] * episodes
        try:
            for _ in range(self.max_steps):
                if provider is not None and intervention == "control":
                    provider.ensure_observations(
                        tuple(observations[index] for index in range(episodes) if active[index]),
                        batch_size=self.online_batch_size,
                    )
                for index in range(episodes):
                    if not active[index]:
                        continue
                    observation = observations[index]
                    policy.reset(1000 + index)
                    sample = PolicySampleV3(
                        (observation,),
                        np.asarray([True], dtype=bool),
                        None,
                        None,
                        observation.episode_id,
                        observation.step_index,
                    )
                    action = policy.predict_chunk(sample).values[0]
                    if first_mse[index] is None:
                        expert = make_v31_episode(
                            task_id=task_id,
                            split="test",
                            stratum=stratum,
                            data_seed=42,
                            index=index,
                        ).transitions[0].action
                        first_mse[index] = float(np.mean(np.square(action - expert)))
                    transition = environments[index].step(action)
                    observations[index] = transition.next_observation
                    final_distance[index] = float(transition.info["final_distance"])
                    success[index] = bool(transition.info["success"])
                    active[index] = not (transition.terminated or transition.truncated)
                if not any(active):
                    break
        finally:
            for env in environments:
                env.close()
        if any(item is None for item in first_mse) or not bool(
            np.all(np.isfinite(final_distance))
        ):
            raise ValueError("real rollout matrix contains incomplete metrics")
        complete_first_mse = tuple(
            float(item) for item in first_mse if item is not None
        )
        if len(complete_first_mse) != episodes:
            raise ValueError("real rollout matrix contains incomplete first-action metrics")
        return [
            V31EvidenceRowV1(
                train_seed=seed,
                arm=arm,
                task_id=task_id,
                held_out_stratum=stratum,
                episode_index=index,
                pair_id=f"{task_id}:{stratum}:{index}",
                git_sha=self.git_sha,
                dependency_lock_sha256=self.dependency_hash,
                feature_source_sha256=self.feature_source_hash,
                checkpoint_sha256=checkpoint_sha256,
                success=success[index],
                final_distance=final_distance[index],
                first_action_mse=complete_first_mse[index],
            )
            for index in range(episodes)
        ]

    def execute(
        self,
        design: V31EvidenceDesignV1,
        *,
        only_train_seed: int | None,
        output_dir: Path,
    ) -> V31ExecutionBatchV1:
        output_dir.mkdir(parents=True, exist_ok=False)
        seeds = design.train_seeds if only_train_seed is None else (only_train_seed,)
        root = self._training_root(only_train_seed)
        training_valid = self._validate_training(root, seeds)
        provider = ContentAddressedFrozenFeatureProviderV1(
            base_cache=self.base,
            extractor=self.extractor,
            root=output_dir / "runtime-cache",
            processor_sha256=self.processor_sha256,
            device_environment_sha256=self.device_environment_sha256,
        )
        rows: list[V31EvidenceRowV1] = []
        checkpoint_records: dict[str, str] = {}
        for seed in seeds:
            baseline, baseline_hash = self._load_policy(root, seed, "baseline")
            control_hash: str | None = None
            for arm in design.arms:
                if arm == "baseline":
                    policy = baseline
                    checkpoint_hash = baseline_hash
                    runtime_provider = None
                    intervention = "control"
                else:
                    policy, checkpoint_hash = self._load_policy(root, seed, "smol_control")
                    control_hash = checkpoint_hash if control_hash is None else control_hash
                    if checkpoint_hash != control_hash:
                        raise ValueError("feature interventions must reuse the control checkpoint")
                    runtime_provider = provider
                    intervention = {
                        "smol_control": "control",
                        "feature_mask": "feature_mask",
                        "feature_shuffle": "feature_shuffle",
                    }[arm]
                checkpoint_records[f"seed-{seed}/{arm}"] = checkpoint_hash
                for task_id in design.task_ids:
                    for stratum in design.held_out_strata:
                        rows.extend(
                            self._cell(
                                policy=policy,
                                provider=runtime_provider,
                                intervention=intervention,
                                seed=seed,
                                arm=arm,
                                task_id=task_id,
                                stratum=stratum,
                                episodes=design.episodes_per_cell,
                                checkpoint_sha256=checkpoint_hash,
                            )
                        )
        rows_tuple = tuple(rows)
        checkpoints_hash = _stable_hash(checkpoint_records)
        metrics_hash = v31_row_inventory_sha256(rows_tuple)
        seed_checkpoint_hashes = {
            seed: _stable_hash(
                {
                    key: value
                    for key, value in checkpoint_records.items()
                    if key.startswith(f"seed-{seed}/")
                }
            )
            for seed in seeds
        }
        seed_metrics_hashes = {
            seed: v31_row_inventory_sha256(
                tuple(row for row in rows_tuple if row.train_seed == seed)
            )
            for seed in seeds
        }
        (output_dir / "rows.json").write_text(
            json.dumps([row.to_dict() for row in rows_tuple], indent=2, sort_keys=True) + "\n"
        )
        (output_dir / "checkpoints.json").write_text(
            json.dumps(checkpoint_records, indent=2, sort_keys=True) + "\n"
        )
        scientific = bool(training_valid and not self.git_dirty)
        (output_dir / "execution.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "feature_source": "real_frozen_vlm",
                    "online_batch_size": self.online_batch_size,
                    "checkpoints_sha256": checkpoints_hash,
                    "metrics_sha256": metrics_hash,
                    "seed_checkpoint_sha256": seed_checkpoint_hashes,
                    "seed_metrics_sha256": seed_metrics_hashes,
                    "scientific_eligible": scientific,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return V31ExecutionBatchV1(
            rows_tuple,
            checkpoints_hash,
            metrics_hash,
            "real_frozen_vlm",
            seed_checkpoint_hashes,
            seed_metrics_hashes,
            scientific,
        )
