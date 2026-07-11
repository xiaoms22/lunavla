from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .artifacts import (
    ArtifactHashRecordV1,
    RunManifestV4R2,
    RunManifestV4R3,
    run_manifest_from_mapping,
    sha256_file,
    verify_run_directory,
)
from .config import ExperimentConfig
from .diagnostic_engine import (
    DiagnosticExecutionError,
    DiagnosticRouterV1,
    RoutedObservationV1,
    typed_episode_key,
)
from .diagnostics import (
    DiagnosticTraceRowV1,
    DiagnosticDesignV1,
    DonorBankV1,
    DonorRecordV1,
    FailureRecordV1,
    PromptParityManifestV1,
    PromptParityRecordV1,
    StateRouteSpecV1,
)
from .engine import EngineV3, _execute_alpha
from .fake_tasks import FakePointEnvV3
from .normalization import NormalizationStatsV1


_ALLOWED_WORDING = "诊断框架已在确定性 fixture 上运行并通过完整性验证"


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _sha_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _pair_id(train_seed: int, evaluation_seed: int, route: str, arm: str) -> str:
    return _sha_mapping(
        {
            "train_seed": train_seed,
            "evaluation_seed": evaluation_seed,
            "route": route,
            "arm": arm,
        }
    )


def _load_design(path: str | Path) -> DiagnosticDesignV1:
    source = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(source, Mapping):
        raise TypeError("diagnostic design YAML must contain a mapping")
    return DiagnosticDesignV1.from_mapping(source)


def _resolve_base_config(design_path: Path, design: DiagnosticDesignV1) -> Path:
    candidate = Path(design.base_config)
    if candidate.is_file():
        return candidate
    relative = design_path.parent / candidate
    if relative.is_file():
        return relative
    raise FileNotFoundError(f"diagnostic base config does not exist: {design.base_config}")


def _instruction_entries(
    config: ExperimentConfig, seeds: Sequence[int]
) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    variant = str(config.dataset["parameters"].get("instruction_variant", "constant_v1"))
    for seed in seeds:
        env = FakePointEnvV3(
            str(config.task["id"]), int(config.evaluation["max_steps"]), variant
        )
        try:
            observation = env.reset(seed=seed)
        finally:
            env.close()
        if observation.instruction is None:
            raise ValueError("shuffle donor bank requires instruction-bearing observations")
        entries.append((typed_episode_key(observation.episode_id), observation.instruction))
    return entries


def _derangement(
    entries: Sequence[tuple[str, str]], *, donor_seed: int
) -> dict[str, tuple[str, str]]:
    if len(entries) < 2:
        raise ValueError("shuffle donor bank requires at least two evaluation episodes")
    rng = np.random.default_rng(donor_seed)
    indices = np.arange(len(entries))
    for _ in range(100):
        candidate = rng.permutation(indices)
        if all(
            int(candidate[index]) != index
            and entries[int(candidate[index])][1] != entries[index][1]
            for index in range(len(entries))
        ):
            return {
                entries[index][0]: entries[int(candidate[index])]
                for index in range(len(entries))
            }
    raise ValueError("donor bank cannot form a content-distinct derangement")


def _instruction_donor_bank(
    config: ExperimentConfig, seeds: Sequence[int], *, donor_seed: int
) -> tuple[DonorBankV1, dict[str, tuple[str, str]]]:
    entries = _instruction_entries(config, seeds)
    mapping = _derangement(entries, donor_seed=donor_seed)
    records = tuple(
        DonorRecordV1(
            recipient,
            donor,
            "evaluation",
            None,
            hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            hashlib.sha256(donor_instruction.encode("utf-8")).hexdigest(),
        )
        for recipient, instruction in entries
        for donor, donor_instruction in (mapping[recipient],)
    )
    return DonorBankV1("instruction", "evaluation", donor_seed, records), mapping


def _image_donor_bank(
    config: ExperimentConfig,
    seeds: Sequence[int],
    instruction_mapping: Mapping[str, tuple[str, str]],
    *,
    donor_seed: int,
) -> tuple[
    DonorBankV1,
    dict[tuple[str, int], tuple[str, Mapping[str, np.ndarray[Any, Any]]]],
]:
    cameras = tuple(config.prompt["camera_order"])
    if not cameras:
        cameras = tuple(item.name for item in config.feature_schema.by_role("image"))
    if not cameras:
        raise ValueError("image donor bank requires image features")
    variant = str(config.dataset["parameters"].get("instruction_variant", "constant_v1"))
    trajectories: dict[str, list[Mapping[str, np.ndarray[Any, Any]]]] = {}
    for seed in seeds:
        env = FakePointEnvV3(
            str(config.task["id"]), int(config.evaluation["max_steps"]), variant
        )
        try:
            observation = env.reset(seed=seed)
            key = typed_episode_key(observation.episode_id)
            values: list[Mapping[str, np.ndarray[Any, Any]]] = []
            for _ in range(int(config.evaluation["max_steps"])):
                values.append({name: np.array(observation.images[name], copy=True) for name in cameras})
                transition = env.step(np.zeros(2, dtype=np.float32))
                observation = transition.next_observation
            trajectories[key] = values
        finally:
            env.close()
    runtime: dict[tuple[str, int], tuple[str, Mapping[str, np.ndarray[Any, Any]]]] = {}
    records: list[DonorRecordV1] = []
    for recipient, (donor, _) in instruction_mapping.items():
        for step, recipient_images in enumerate(trajectories[recipient]):
            donor_images = trajectories[donor][step]
            recipient_hash = hashlib.sha256(
                b"".join(np.asarray(recipient_images[name]).tobytes(order="C") for name in cameras)
            ).hexdigest()
            donor_hash = hashlib.sha256(
                b"".join(np.asarray(donor_images[name]).tobytes(order="C") for name in cameras)
            ).hexdigest()
            records.append(
                DonorRecordV1(
                    recipient, donor, "evaluation", step, recipient_hash, donor_hash
                )
            )
            runtime[(recipient, step)] = (donor, donor_images)
    return DonorBankV1("image", "evaluation", donor_seed, tuple(records)), runtime


@dataclass(frozen=True)
class EvidenceManifestV2:
    design_sha256: str
    base_config_sha256: str
    source_manifests: tuple[ArtifactHashRecordV1, ...]
    matrix_sha256: str
    aggregate_sha256: str
    per_pair_sha256: str
    expected_pairs: int
    observed_pairs: int
    homogeneous: bool
    matrix_complete: bool
    reduced_design: bool
    claim_allowed: bool
    allowed_wording: str

    def __post_init__(self) -> None:
        for name in (
            "design_sha256", "base_config_sha256", "matrix_sha256",
            "aggregate_sha256", "per_pair_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(item not in "0123456789abcdef" for item in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        sources = tuple(self.source_manifests)
        if not sources:
            raise ValueError("evidence requires source manifests")
        if len({item.path for item in sources}) != len(sources):
            raise ValueError("source manifest paths must be unique")
        for name in ("expected_pairs", "observed_pairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.observed_pairs > self.expected_pairs:
            raise ValueError("observed_pairs cannot exceed expected_pairs")
        for name in (
            "homogeneous", "matrix_complete", "reduced_design", "claim_allowed"
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.claim_allowed:
            raise ValueError("Beta 1 evidence cannot open scientific claims")
        if self.matrix_complete != (self.observed_pairs == self.expected_pairs):
            raise ValueError("matrix_complete conflicts with pair counts")
        if self.allowed_wording != _ALLOWED_WORDING:
            raise ValueError("unexpected Beta 1 allowed wording")
        object.__setattr__(self, "source_manifests", sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "contract_revision": 1,
            "format": "lunavla_v3_diagnostic_evidence",
            "design_sha256": self.design_sha256,
            "base_config_sha256": self.base_config_sha256,
            "source_manifests": [item.to_dict() for item in self.source_manifests],
            "matrix_sha256": self.matrix_sha256,
            "aggregate_sha256": self.aggregate_sha256,
            "per_pair_sha256": self.per_pair_sha256,
            "expected_pairs": self.expected_pairs,
            "observed_pairs": self.observed_pairs,
            "homogeneous": self.homogeneous,
            "matrix_complete": self.matrix_complete,
            "reduced_design": self.reduced_design,
            "claim_allowed": self.claim_allowed,
            "allowed_wording": self.allowed_wording,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceManifestV2":
        fields = {
            "schema_version", "contract_revision", "format", "design_sha256",
            "base_config_sha256", "source_manifests", "matrix_sha256",
            "aggregate_sha256", "per_pair_sha256", "expected_pairs",
            "observed_pairs", "homogeneous", "matrix_complete", "reduced_design",
            "claim_allowed", "allowed_wording",
        }
        if set(value) != fields:
            raise ValueError("invalid EvidenceManifestV2 fields")
        if value["schema_version"] != 2 or isinstance(value["schema_version"], bool):
            raise ValueError("evidence schema_version must be integer 2")
        if value["contract_revision"] != 1 or isinstance(value["contract_revision"], bool):
            raise ValueError("evidence contract_revision must be integer 1")
        if value["format"] != "lunavla_v3_diagnostic_evidence":
            raise ValueError("unsupported evidence format")
        sources = value["source_manifests"]
        if isinstance(sources, (str, bytes, Mapping)) or not isinstance(sources, Sequence):
            raise TypeError("source_manifests must be a sequence")
        return cls(
            design_sha256=value["design_sha256"],
            base_config_sha256=value["base_config_sha256"],
            source_manifests=tuple(
                ArtifactHashRecordV1.from_mapping(item) for item in sources
            ),
            matrix_sha256=value["matrix_sha256"],
            aggregate_sha256=value["aggregate_sha256"],
            per_pair_sha256=value["per_pair_sha256"],
            expected_pairs=value["expected_pairs"],
            observed_pairs=value["observed_pairs"],
            homogeneous=value["homogeneous"],
            matrix_complete=value["matrix_complete"],
            reduced_design=value["reduced_design"],
            claim_allowed=value["claim_allowed"],
            allowed_wording=value["allowed_wording"],
        )

    def save(self, path: str | Path) -> Path:
        return _write_json(Path(path), self.to_dict())


def _write_trace(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _parity_manifest(
    config: ExperimentConfig,
    normalization: NormalizationStatsV1,
    route: StateRouteSpecV1,
    seeds: Sequence[int],
) -> PromptParityManifestV1:
    router = DiagnosticRouterV1(config, normalization, route=route)
    variant = str(config.dataset["parameters"].get("instruction_variant", "constant_v1"))
    records: list[PromptParityRecordV1] = []
    for seed in seeds:
        env = FakePointEnvV3(
            str(config.task["id"]), int(config.evaluation["max_steps"]), variant
        )
        try:
            observation = env.reset(seed=seed)
        finally:
            env.close()
        sample_id = typed_episode_key(observation.episode_id)
        for phase in ("train", "eval", "deploy"):
            routed = router.route_observation(observation, phase=phase)
            records.append(
                PromptParityRecordV1(
                    sample_id=sample_id,
                    phase=phase,
                    prompt_sha256=routed.prompt_spec.rendered_sha256,
                    camera_order=tuple(config.prompt["camera_order"]),
                    assistant_target=str(config.prompt["assistant_target"]),
                    expert_state_keys=routed.expert_state_keys,
                    prompt_state_keys=routed.prompt_state_keys,
                    feature_schema_sha256=config.feature_schema.sha256(),
                )
            )
    return PromptParityManifestV1(tuple(records))


def _diagnostic_cell(
    *,
    config: ExperimentConfig,
    design: DiagnosticDesignV1,
    route: StateRouteSpecV1,
    donor_banks: Mapping[str, DonorBankV1],
    donor_instructions: Mapping[str, tuple[str, str]],
    donor_images: Mapping[
        tuple[str, int], tuple[str, Mapping[str, np.ndarray[Any, Any]]]
    ],
    output: Path,
) -> tuple[RunManifestV4R3, list[dict[str, Any]], list[dict[str, Any]]]:
    result = _execute_alpha(config, output)
    base_manifest = run_manifest_from_mapping(
        json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    )
    if not isinstance(base_manifest, RunManifestV4R2):
        raise ValueError("diagnostic base run must produce manifest revision 2")
    normalization = NormalizationStatsV1.from_mapping(
        json.loads((output / "normalization.json").read_text(encoding="utf-8"))
    )
    parity = _parity_manifest(config, normalization, route, design.evaluation_seeds)
    engine = EngineV3(config)
    engine.normalization = normalization
    policy = engine.restore_policy(output / "checkpoint")
    traces: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    pairs = [
        {
            "pair_id": _pair_id(
                int(config.training["seed"]), seed, route.mode, arm.arm_id
            ),
            "train_seed": int(config.training["seed"]),
            "evaluation_seed": seed,
            "route": route.mode,
            "arm": arm.arm_id,
        }
        for arm in design.interventions
        for seed in design.evaluation_seeds
    ]
    pair_lookup = {
        (item["evaluation_seed"], item["arm"]): item["pair_id"] for item in pairs
    }
    donor_hashes = {
        (bank.modality, item.recipient_id, item.step_index): item.donor_content_sha256
        for bank in donor_banks.values()
        for item in bank.records
    }
    for arm in design.interventions:
        router = DiagnosticRouterV1(
            config,
            normalization,
            route=route,
            intervention=arm,
            donor_instructions=(
                donor_instructions
                if arm.kind == "prompt" and arm.operator == "shuffle"
                else None
            ),
            donor_images=(
                donor_images
                if arm.kind == "image" and arm.operator == "shuffle"
                else None
            ),
            counterfactual_transform_id=design.counterfactual_transform_id,
        )
        control_router = DiagnosticRouterV1(config, normalization, route=route)
        engine.diagnostic_router = router

        def trace(
            seed: int,
            routed: RoutedObservationV1,
            action: np.ndarray[Any, Any],
            transition: Any,
            step: int,
        ) -> None:
            canonical = control_router.route_observation(
                transition.observation, phase="eval"
            )
            episode_id = typed_episode_key(transition.observation.episode_id)
            donor_hash = (
                donor_hashes.get(
                    (
                        arm.kind,
                        episode_id,
                        transition.observation.step_index if arm.kind == "image" else None,
                    )
                )
                if routed.donor_id is not None
                else None
            )
            failures = (
                (
                    FailureRecordV1(
                        "execution", "timeout", "max_steps_v1", "automatic", None
                    ),
                )
                if transition.truncated
                else ()
            )
            row = DiagnosticTraceRowV1(
                pair_id=str(pair_lookup[(seed, arm.arm_id)]),
                train_seed=int(config.training["seed"]),
                evaluation_seed=seed,
                episode_id=episode_id,
                route=route.mode,
                arm=arm.arm_id,
                intervention_kind=arm.kind,
                step_id=step,
                canonical_prompt_sha256=canonical.prompt_spec.rendered_sha256,
                effective_prompt_sha256=routed.prompt_spec.rendered_sha256,
                expert_state_keys=routed.expert_state_keys,
                prompt_state_keys=routed.prompt_state_keys,
                camera_order=tuple(config.prompt["camera_order"]),
                donor_id=routed.donor_id,
                donor_content_sha256=donor_hash,
                transform_id=(
                    design.counterfactual_transform_id
                    if arm.operator == "counterfactual"
                    else None
                ),
                action_chunk_sha256=hashlib.sha256(
                    np.asarray(action, dtype=np.float32).tobytes(order="C")
                ).hexdigest(),
                reward=transition.reward,
                terminated=transition.terminated,
                truncated=transition.truncated,
                success=bool(transition.info.get("success", False)),
                failures=failures,
                error_stage=None,
                error_origin=None,
            )
            traces.append(row.to_dict())

        arm_metrics = engine.evaluate(
            policy,
            FakePointEnvV3(
                str(config.task["id"]),
                int(config.evaluation["max_steps"]),
                str(config.dataset["parameters"].get("instruction_variant", "constant_v1")),
            ),
            trace_callback=trace,
        )
        metrics.append({"route": route.mode, "arm": arm.arm_id, **arm_metrics})

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in traces:
        grouped.setdefault(str(row["pair_id"]), []).append(row)
    for pair in pairs:
        rows = grouped.get(str(pair["pair_id"]), [])
        if not rows:
            raise ValueError(f"diagnostic pair has no rollout trace: {pair['pair_id']}")
        pair["total_reward"] = float(sum(float(item["reward"]) for item in rows))
        pair["success"] = bool(rows[-1]["success"])
        pair["steps"] = len(rows)

    prompt_path = _write_json(
        output / "prompt_contract.json", config.to_dict()["prompt"]
    )
    route_path = _write_json(output / "route_contract.json", route.to_dict())
    design_path = _write_json(output / "diagnostic_design.json", design.to_dict())
    pair_path = _write_json(output / "pairs.json", {"schema_version": 1, "pairs": pairs})
    interventions_path = _write_json(
        output / "interventions.json",
        {"schema_version": 1, "items": [item.to_dict() for item in design.interventions]},
    )
    donor_path = _write_json(
        output / "donor_bank.json",
        {name: bank.to_dict() for name, bank in sorted(donor_banks.items())},
    )
    parity_path = _write_json(output / "parity.json", parity.to_dict())
    trace_path = _write_trace(output / "trace.jsonl", traces)
    consumption_path = _write_json(
        output / "input_consumption.json",
        {
            "schema_version": 1,
            "route": route.mode,
            "expert_state_keys": list(route.state_features if route.expert_enabled else ()),
            "prompt_state_keys": list(route.state_features if route.prompt_enabled else ()),
            "camera_order": list(config.prompt["camera_order"]),
            "metadata_allowlist": ["diagnostic"],
        },
    )
    metrics_path = _write_json(
        output / "metrics.json",
        {
            "claim_allowed": False,
            "complete": True,
            "base_final_loss": result.losses[-1],
            "arms": metrics,
        },
    )
    manifest = RunManifestV4R3(
        **{
            key: value
            for key, value in base_manifest.__dict__.items()
            if key != "metrics_sha256"
        },
        metrics_sha256=sha256_file(metrics_path),
        prompt_contract_sha256=sha256_file(prompt_path),
        route_contract_sha256=sha256_file(route_path),
        diagnostic_design_sha256=sha256_file(design_path),
        pair_set_sha256=sha256_file(pair_path),
        intervention_set_sha256=sha256_file(interventions_path),
        donor_bank_sha256=sha256_file(donor_path),
        parity_manifest_sha256=sha256_file(parity_path),
        failure_trace_sha256=sha256_file(trace_path),
        input_consumption_sha256=sha256_file(consumption_path),
        camera_order=tuple(config.prompt["camera_order"]),
        parity_verified=parity.verified,
        complete=True,
    )
    manifest.save(output / "manifest.json")
    verify_run_directory(output)
    return manifest, pairs, metrics


def _execute_diagnostic(
    design_path: Path, design: DiagnosticDesignV1, output: Path
) -> EvidenceManifestV2:
    base_path = _resolve_base_config(design_path, design)
    base = ExperimentConfig.load(base_path)
    if base.contract_revision != 2 or not base.diagnostics["enabled"]:
        raise ValueError("diagnostic design requires an enabled revision 2 base config")
    instruction_bank, donor_instructions = _instruction_donor_bank(
        base, design.evaluation_seeds, donor_seed=design.donor_seed
    )
    image_bank, donor_images = _image_donor_bank(
        base,
        design.evaluation_seeds,
        donor_instructions,
        donor_seed=design.donor_seed,
    )
    donor_banks = {"instruction": instruction_bank, "image": image_bank}
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "resolved_design.json", design.to_dict())
    source_records: list[ArtifactHashRecordV1] = []
    all_pairs: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    source_manifests: list[RunManifestV4R3] = []
    final_root = Path(design.output_dir)
    for train_seed in design.train_seeds:
        for route in design.routes:
            payload = base.to_dict()
            payload["training"]["seed"] = train_seed
            payload["evaluation"]["seed"] = design.evaluation_seeds[0]
            payload["evaluation"]["seeds"] = list(design.evaluation_seeds)
            payload["evaluation"]["episodes"] = len(design.evaluation_seeds)
            payload["routing"] = route.to_dict()
            payload["routing"].pop("schema_version")
            logical = final_root / "runs" / str(train_seed) / route.mode
            payload["artifacts"]["output_dir"] = logical.as_posix()
            config = ExperimentConfig.from_mapping(payload)
            cell = output / "runs" / str(train_seed) / route.mode
            manifest, pairs, metrics = _diagnostic_cell(
                config=config,
                design=design,
                route=route,
                donor_banks=donor_banks,
                donor_instructions=donor_instructions,
                donor_images=donor_images,
                output=cell,
            )
            source_manifests.append(manifest)
            all_pairs.extend(pairs)
            all_metrics.extend(
                {"train_seed": train_seed, **item} for item in metrics
            )
            manifest_path = cell / "manifest.json"
            source_records.append(
                ArtifactHashRecordV1(
                    manifest_path.relative_to(output).as_posix(), sha256_file(manifest_path)
                )
            )
    pair_ids = [str(item["pair_id"]) for item in all_pairs]
    expected = (
        len(design.train_seeds)
        * len(design.routes)
        * len(design.interventions)
        * len(design.evaluation_seeds)
    )
    matrix_complete = len(pair_ids) == expected and len(pair_ids) == len(set(pair_ids))
    matrix_path = _write_json(
        output / "matrix.json",
        {
            "schema_version": 1,
            "expected_pairs": expected,
            "observed_pairs": len(pair_ids),
            "complete": matrix_complete,
            "pair_ids": sorted(pair_ids),
        },
    )
    aggregate_path = _write_json(
        output / "aggregate.json",
        {"schema_version": 1, "claim_allowed": False, "cells": all_metrics},
    )
    pair_csv = output / "per_pair.csv"
    with pair_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "pair_id", "train_seed", "evaluation_seed", "route", "arm",
                "total_reward", "success", "steps",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(all_pairs, key=lambda item: str(item["pair_id"])))
    homogeneous = (
        len({item.git_sha for item in source_manifests}) == 1
        and len({item.dependency_lock_sha256 for item in source_manifests}) == 1
        and len({item.policy_id for item in source_manifests}) == 1
    )
    evidence = EvidenceManifestV2(
        design_sha256=sha256_file(output / "resolved_design.json"),
        base_config_sha256=base.sha256(),
        source_manifests=tuple(source_records),
        matrix_sha256=sha256_file(matrix_path),
        aggregate_sha256=sha256_file(aggregate_path),
        per_pair_sha256=sha256_file(pair_csv),
        expected_pairs=expected,
        observed_pairs=len(pair_ids),
        homogeneous=homogeneous,
        matrix_complete=matrix_complete,
        reduced_design=design.reduced_design,
        claim_allowed=False,
        allowed_wording=_ALLOWED_WORDING,
    )
    evidence.save(output / "evidence_manifest.json")
    verify_diagnostic_output(output)
    return evidence


def run_diagnostic(design_file: str | Path, *, overwrite: bool = False) -> Path:
    design_path = Path(design_file)
    design = _load_design(design_path)
    output = Path(design.output_dir)
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"diagnostic output is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"diagnostic output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    backup = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
    incomplete = output.with_name(f".{output.name}.incomplete.json")
    try:
        _execute_diagnostic(design_path, design, staging)
        if output.exists():
            output.replace(backup)
        staging.replace(output)
        if backup.exists():
            shutil.rmtree(backup)
        if incomplete.exists():
            incomplete.unlink()
        verify_diagnostic_output(output)
        return output
    except Exception as exc:
        _write_json(
            incomplete,
            {
                "schema_version": 1,
                "complete": False,
                "error_type": type(exc).__name__,
                "error_stage": (
                    exc.stage if isinstance(exc, DiagnosticExecutionError) else None
                ),
                "error_origin": (
                    exc.origin if isinstance(exc, DiagnosticExecutionError) else None
                ),
            },
        )
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise


def verify_diagnostic_output(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"diagnostic output does not exist: {root}")
    manifest_path = root / "evidence_manifest.json"
    evidence = EvidenceManifestV2.from_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    checks = {
        "resolved_design.json": evidence.design_sha256,
        "matrix.json": evidence.matrix_sha256,
        "aggregate.json": evidence.aggregate_sha256,
        "per_pair.csv": evidence.per_pair_sha256,
    }
    for name, expected in checks.items():
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or sha256_file(candidate) != expected:
            raise ValueError(f"diagnostic artifact hash mismatch: {name}")
    manifests: list[RunManifestV4R3] = []
    for record in evidence.source_manifests:
        candidate = (root / record.path).resolve()
        if not candidate.is_relative_to(root) or sha256_file(candidate) != record.sha256:
            raise ValueError(f"diagnostic source manifest hash mismatch: {record.path}")
        verified = verify_run_directory(candidate.parent)
        parsed = run_manifest_from_mapping(verified)
        if not isinstance(parsed, RunManifestV4R3):
            raise ValueError("diagnostic evidence requires run manifest revision 3")
        manifests.append(parsed)
    matrix = json.loads((root / "matrix.json").read_text(encoding="utf-8"))
    pair_ids = matrix.get("pair_ids")
    complete = (
        isinstance(pair_ids, list)
        and len(pair_ids) == evidence.expected_pairs
        and len(pair_ids) == len(set(pair_ids))
        and evidence.observed_pairs == evidence.expected_pairs
        and evidence.matrix_complete
    )
    if not complete:
        raise ValueError("diagnostic matrix is incomplete or contains duplicate pairs")
    homogeneous = (
        len({item.git_sha for item in manifests}) == 1
        and len({item.dependency_lock_sha256 for item in manifests}) == 1
        and len({item.policy_id for item in manifests}) == 1
    )
    if homogeneous != evidence.homogeneous:
        raise ValueError("diagnostic homogeneity result does not match source manifests")
    if evidence.claim_allowed:
        raise ValueError("Beta 1 diagnostic evidence cannot allow scientific claims")
    return evidence.to_dict()


def write_diagnostic_report(path: str | Path, output: str | Path) -> Path:
    evidence = verify_diagnostic_output(path)
    source = Path(path).resolve()
    target = Path(output)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"diagnostic report directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=True)
    aggregate = json.loads((source / "aggregate.json").read_text(encoding="utf-8"))
    _write_json(target / "summary.json", {"evidence": evidence, "aggregate": aggregate})
    shutil.copy2(source / "per_pair.csv", target / "per_pair.csv")
    wording = html.escape(str(evidence["allowed_wording"]))
    body = (
        "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
        "<title>LunaVLA v3 diagnostic report</title>"
        f"<h1>LunaVLA v3 diagnostic report</h1><p>{wording}</p>"
        f"<p>Pairs: {evidence['observed_pairs']} / {evidence['expected_pairs']}</p>"
        "<p>Scientific claims allowed: false</p></html>\n"
    )
    (target / "index.html").write_text(body, encoding="utf-8")
    return target
