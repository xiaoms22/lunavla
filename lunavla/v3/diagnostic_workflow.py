from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

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
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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


def _normalized_cell_mapping(config: ExperimentConfig) -> dict[str, Any]:
    payload = config.to_dict()
    payload["training"]["seed"] = "<design-train-seed>"
    payload["routing"]["mode"] = "<design-route>"
    payload["artifacts"]["output_dir"] = "<design-output>"
    return payload


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
    relative = (design_path.resolve().parent / design.base_config).resolve()
    repository = (_REPOSITORY_ROOT / design.base_config).resolve()
    if relative.is_relative_to(design_path.resolve().parent) and relative.is_file():
        return relative
    if repository.is_relative_to(_REPOSITORY_ROOT) and repository.is_file():
        return repository
    raise FileNotFoundError(f"diagnostic base config does not exist: {design.base_config}")


def _resolve_output(design_path: Path, design: DiagnosticDesignV1) -> Path:
    resolved_design = design_path.resolve()
    root = (
        _REPOSITORY_ROOT
        if resolved_design.is_relative_to(_REPOSITORY_ROOT)
        else resolved_design.parent
    )
    output = (root / design.output_dir).resolve()
    if not output.is_relative_to(root):
        raise ValueError("diagnostic output escapes its allowed root")
    return output


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
            values.append(
                {name: np.array(observation.images[name], copy=True) for name in cameras}
            )
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


def synthetic_thumbnail_payloads(
    config: ExperimentConfig,
    seeds: Sequence[int],
    donor_images: Mapping[
        tuple[str, int], tuple[str, Mapping[str, np.ndarray[Any, Any]]]
    ],
) -> dict[str, bytes]:
    cameras = tuple(config.prompt["camera_order"])
    if len(cameras) != 1:
        raise ValueError("Beta 1 synthetic thumbnails require exactly one camera")
    camera = cameras[0]
    variant = str(config.dataset["parameters"].get("instruction_variant", "constant_v1"))
    payloads: dict[str, bytes] = {}
    for seed in seeds:
        env = FakePointEnvV3(
            str(config.task["id"]), int(config.evaluation["max_steps"]), variant
        )
        try:
            observation = env.reset(seed=seed)
        finally:
            env.close()
        key = typed_episode_key(observation.episode_id)
        if (key, 0) not in donor_images:
            raise ValueError("synthetic thumbnail is missing an image donor")
        _, shuffled = donor_images[(key, 0)]
        for arm, array in (
            ("control", observation.images[camera]),
            ("image_shuffle", shuffled[camera]),
        ):
            value = np.asarray(array)
            if value.dtype != np.uint8 or value.shape != (16, 16, 3):
                raise ValueError("synthetic thumbnail must be 16x16 RGB uint8")
            stream = io.BytesIO()
            Image.fromarray(value, mode="RGB").save(
                stream, format="PNG", optimize=False, compress_level=9
            )
            payloads[f"seed-{seed}-{arm}.png"] = stream.getvalue()
    return payloads


def _write_synthetic_thumbnails(
    output: Path,
    config: ExperimentConfig,
    seeds: Sequence[int],
    donor_images: Mapping[
        tuple[str, int], tuple[str, Mapping[str, np.ndarray[Any, Any]]]
    ],
) -> Path:
    directory = output / "thumbnails"
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for filename, payload in sorted(
        synthetic_thumbnail_payloads(config, seeds, donor_images).items()
    ):
        target = directory / filename
        target.write_bytes(payload)
        parts = filename.removesuffix(".png").split("-", 2)
        seed = int(parts[1])
        arm = parts[2]
        records.append(
            {
                "path": f"thumbnails/{filename}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "seed": seed,
                "arm": arm,
                "step_index": 0,
                "shape": [16, 16, 3],
                "dtype": "uint8",
                "synthetic": True,
                "metadata_allowed": False,
            }
        )
    return _write_json(
        output / "thumbnails.json",
        {"schema_version": 1, "records": records},
    )


@dataclass(frozen=True)
class EvidenceManifestV2:
    design_sha256: str
    base_config_sha256: str
    base_config_file_sha256: str
    cell_contract_sha256: str
    source_manifests: tuple[ArtifactHashRecordV1, ...]
    matrix_sha256: str
    aggregate_sha256: str
    per_pair_sha256: str
    thumbnail_manifest_sha256: str | None
    expected_pairs: int
    observed_pairs: int
    homogeneous: bool
    matrix_complete: bool
    reduced_design: bool
    claim_allowed: bool
    framework_statement_allowed: bool
    release_eligible: bool
    gate_reasons: tuple[str, ...]
    allowed_wording: str | None

    def __post_init__(self) -> None:
        for name in (
            "design_sha256", "base_config_sha256", "base_config_file_sha256",
            "cell_contract_sha256", "matrix_sha256", "aggregate_sha256",
            "per_pair_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(item not in "0123456789abcdef" for item in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.thumbnail_manifest_sha256 is not None and (
            len(self.thumbnail_manifest_sha256) != 64
            or any(item not in "0123456789abcdef" for item in self.thumbnail_manifest_sha256)
        ):
            raise ValueError("thumbnail_manifest_sha256 must be null or SHA-256")
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
            "homogeneous", "matrix_complete", "reduced_design", "claim_allowed",
            "framework_statement_allowed", "release_eligible",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.claim_allowed:
            raise ValueError("Beta 1 evidence cannot open scientific claims")
        if self.matrix_complete != (self.observed_pairs == self.expected_pairs):
            raise ValueError("matrix_complete conflicts with pair counts")
        reasons = tuple(self.gate_reasons)
        allowed_reasons = {
            "reduced_design", "dirty_source", "mixed_sha", "mixed_dependency",
            "config_drift", "incomplete_matrix", "parity_failure",
            "beta1_framework_only",
        }
        if not reasons or len(reasons) != len(set(reasons)) or not set(reasons) <= allowed_reasons:
            raise ValueError("invalid evidence gate_reasons")
        if self.framework_statement_allowed != (self.allowed_wording == _ALLOWED_WORDING):
            raise ValueError("framework statement flag conflicts with allowed wording")
        if self.release_eligible and not self.framework_statement_allowed:
            raise ValueError("release eligibility requires a framework statement")
        object.__setattr__(self, "source_manifests", sources)
        object.__setattr__(self, "gate_reasons", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "contract_revision": 1,
            "format": "lunavla_v3_diagnostic_evidence",
            "design_sha256": self.design_sha256,
            "base_config_sha256": self.base_config_sha256,
            "base_config_file_sha256": self.base_config_file_sha256,
            "cell_contract_sha256": self.cell_contract_sha256,
            "source_manifests": [item.to_dict() for item in self.source_manifests],
            "matrix_sha256": self.matrix_sha256,
            "aggregate_sha256": self.aggregate_sha256,
            "per_pair_sha256": self.per_pair_sha256,
            "thumbnail_manifest_sha256": self.thumbnail_manifest_sha256,
            "expected_pairs": self.expected_pairs,
            "observed_pairs": self.observed_pairs,
            "homogeneous": self.homogeneous,
            "matrix_complete": self.matrix_complete,
            "reduced_design": self.reduced_design,
            "claim_allowed": self.claim_allowed,
            "framework_statement_allowed": self.framework_statement_allowed,
            "release_eligible": self.release_eligible,
            "gate_reasons": list(self.gate_reasons),
            "allowed_wording": self.allowed_wording,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceManifestV2":
        fields = {
            "schema_version", "contract_revision", "format", "design_sha256",
            "base_config_sha256", "base_config_file_sha256",
            "cell_contract_sha256", "source_manifests", "matrix_sha256",
            "aggregate_sha256", "per_pair_sha256", "expected_pairs",
            "thumbnail_manifest_sha256",
            "observed_pairs", "homogeneous", "matrix_complete", "reduced_design",
            "claim_allowed", "framework_statement_allowed", "release_eligible",
            "gate_reasons", "allowed_wording",
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
            base_config_file_sha256=value["base_config_file_sha256"],
            cell_contract_sha256=value["cell_contract_sha256"],
            source_manifests=tuple(
                ArtifactHashRecordV1.from_mapping(item) for item in sources
            ),
            matrix_sha256=value["matrix_sha256"],
            aggregate_sha256=value["aggregate_sha256"],
            per_pair_sha256=value["per_pair_sha256"],
            thumbnail_manifest_sha256=value["thumbnail_manifest_sha256"],
            expected_pairs=value["expected_pairs"],
            observed_pairs=value["observed_pairs"],
            homogeneous=value["homogeneous"],
            matrix_complete=value["matrix_complete"],
            reduced_design=value["reduced_design"],
            claim_allowed=value["claim_allowed"],
            framework_statement_allowed=value["framework_statement_allowed"],
            release_eligible=value["release_eligible"],
            gate_reasons=tuple(value["gate_reasons"]),
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
                        "instruction" if arm.kind == "prompt" else "image",
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
    cell_contract_path = _write_json(
        output / "cell_contract.json", _normalized_cell_mapping(config)
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
        cell_contract_sha256=sha256_file(cell_contract_path),
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
    base_config_path = _write_json(output / "base_config.json", base.to_dict())
    root_cell_contract_path = _write_json(
        output / "cell_contract.json", _normalized_cell_mapping(base)
    )
    thumbnail_manifest_path = (
        _write_synthetic_thumbnails(
            output, base, design.evaluation_seeds, donor_images
        )
        if {item.kind for item in design.interventions} == {"image"}
        else None
    )
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
    sha_homogeneous = len({item.git_sha for item in source_manifests}) == 1
    dependency_homogeneous = (
        len({item.dependency_lock_sha256 for item in source_manifests}) == 1
    )
    config_homogeneous = (
        len({item.cell_contract_sha256 for item in source_manifests}) == 1
    )
    homogeneous = (
        sha_homogeneous
        and dependency_homogeneous
        and config_homogeneous
        and len({item.policy_id for item in source_manifests}) == 1
    )
    parity_ok = all(item.parity_verified for item in source_manifests)
    dirty = any(item.git_dirty for item in source_manifests)
    gate_reasons = ["beta1_framework_only"]
    if design.reduced_design:
        gate_reasons.append("reduced_design")
    if dirty:
        gate_reasons.append("dirty_source")
    if not sha_homogeneous:
        gate_reasons.append("mixed_sha")
    if not dependency_homogeneous:
        gate_reasons.append("mixed_dependency")
    if not config_homogeneous:
        gate_reasons.append("config_drift")
    if not matrix_complete:
        gate_reasons.append("incomplete_matrix")
    if not parity_ok:
        gate_reasons.append("parity_failure")
    framework_statement_allowed = (
        matrix_complete and homogeneous and parity_ok and not dirty
    )
    evidence = EvidenceManifestV2(
        design_sha256=sha256_file(output / "resolved_design.json"),
        base_config_sha256=base.sha256(),
        base_config_file_sha256=sha256_file(base_config_path),
        cell_contract_sha256=sha256_file(root_cell_contract_path),
        source_manifests=tuple(source_records),
        matrix_sha256=sha256_file(matrix_path),
        aggregate_sha256=sha256_file(aggregate_path),
        per_pair_sha256=sha256_file(pair_csv),
        thumbnail_manifest_sha256=(
            None
            if thumbnail_manifest_path is None
            else sha256_file(thumbnail_manifest_path)
        ),
        expected_pairs=expected,
        observed_pairs=len(pair_ids),
        homogeneous=homogeneous,
        matrix_complete=matrix_complete,
        reduced_design=design.reduced_design,
        claim_allowed=False,
        framework_statement_allowed=framework_statement_allowed,
        release_eligible=framework_statement_allowed,
        gate_reasons=tuple(sorted(gate_reasons)),
        allowed_wording=(
            _ALLOWED_WORDING if framework_statement_allowed else None
        ),
    )
    evidence.save(output / "evidence_manifest.json")
    verify_diagnostic_output(output)
    return evidence


def run_diagnostic(design_file: str | Path, *, overwrite: bool = False) -> Path:
    design_path = Path(design_file).resolve()
    design = _load_design(design_path)
    output = _resolve_output(design_path, design)
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
        "base_config.json": evidence.base_config_file_sha256,
        "cell_contract.json": evidence.cell_contract_sha256,
        "matrix.json": evidence.matrix_sha256,
        "aggregate.json": evidence.aggregate_sha256,
        "per_pair.csv": evidence.per_pair_sha256,
    }
    for name, expected in checks.items():
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or sha256_file(candidate) != expected:
            raise ValueError(f"diagnostic artifact hash mismatch: {name}")
    design = DiagnosticDesignV1.from_mapping(
        json.loads((root / "resolved_design.json").read_text(encoding="utf-8"))
    )
    base_config = ExperimentConfig.from_mapping(
        json.loads((root / "base_config.json").read_text(encoding="utf-8"))
    )
    if base_config.sha256() != evidence.base_config_sha256:
        raise ValueError("base config semantic hash mismatch")
    if json.loads((root / "cell_contract.json").read_text(encoding="utf-8")) != (
        _normalized_cell_mapping(base_config)
    ):
        raise ValueError("base config normalized cell contract drifted")
    image_suite = {item.kind for item in design.interventions} == {"image"}
    if image_suite != (evidence.thumbnail_manifest_sha256 is not None):
        raise ValueError("thumbnail manifest presence does not match diagnostic suite")
    if evidence.thumbnail_manifest_sha256 is not None:
        thumbnail_manifest = root / "thumbnails.json"
        if sha256_file(thumbnail_manifest) != evidence.thumbnail_manifest_sha256:
            raise ValueError("diagnostic artifact hash mismatch: thumbnails.json")
        thumbnail_payload = json.loads(thumbnail_manifest.read_text(encoding="utf-8"))
        if set(thumbnail_payload) != {"schema_version", "records"} or thumbnail_payload[
            "schema_version"
        ] != 1:
            raise ValueError("invalid synthetic thumbnail manifest")
        records = thumbnail_payload["records"]
        expected_thumbnails = {
            (seed, arm)
            for seed in design.evaluation_seeds
            for arm in ("control", "image_shuffle")
        }
        observed_thumbnails: set[tuple[int, str]] = set()
        for record in records:
            if set(record) != {
                "path", "sha256", "seed", "arm", "step_index", "shape", "dtype",
                "synthetic", "metadata_allowed",
            }:
                raise ValueError("invalid synthetic thumbnail record")
            relative = Path(record["path"])
            candidate = (root / relative).resolve()
            if (
                not candidate.is_relative_to(root / "thumbnails")
                or sha256_file(candidate) != record["sha256"]
            ):
                raise ValueError("synthetic thumbnail hash or containment mismatch")
            with Image.open(candidate) as image:
                image.load()
                if image.mode != "RGB" or image.size != (16, 16) or image.info:
                    raise ValueError("synthetic thumbnail format or metadata is invalid")
            identity = (int(record["seed"]), str(record["arm"]))
            if (
                identity in observed_thumbnails
                or record["step_index"] != 0
                or record["shape"] != [16, 16, 3]
                or record["dtype"] != "uint8"
                or record["synthetic"] is not True
                or record["metadata_allowed"] is not False
            ):
                raise ValueError("synthetic thumbnail metadata is invalid")
            observed_thumbnails.add(identity)
        if observed_thumbnails != expected_thumbnails:
            raise ValueError("synthetic thumbnail matrix is incomplete")
    manifests: list[RunManifestV4R3] = []
    observed_pair_ids: set[str] = set()
    recomputed_pairs: dict[str, dict[str, Any]] = {}
    for record in evidence.source_manifests:
        candidate = (root / record.path).resolve()
        if not candidate.is_relative_to(root) or sha256_file(candidate) != record.sha256:
            raise ValueError(f"diagnostic source manifest hash mismatch: {record.path}")
        verified = verify_run_directory(candidate.parent)
        parsed = run_manifest_from_mapping(verified)
        if not isinstance(parsed, RunManifestV4R3):
            raise ValueError("diagnostic evidence requires run manifest revision 3")
        parity = PromptParityManifestV1.from_mapping(
            json.loads((candidate.parent / "parity.json").read_text(encoding="utf-8"))
        )
        if parsed.parity_verified != parity.verified:
            raise ValueError("run manifest parity flag does not recompute")
        cell_config = ExperimentConfig.from_mapping(
            json.loads((candidate.parent / "resolved_config.json").read_text(encoding="utf-8"))
        )
        expected_cell = _normalized_cell_mapping(cell_config)
        actual_cell = json.loads(
            (candidate.parent / "cell_contract.json").read_text(encoding="utf-8")
        )
        if actual_cell != expected_cell:
            raise ValueError("diagnostic normalized cell contract drifted")
        pairs_payload = json.loads(
            (candidate.parent / "pairs.json").read_text(encoding="utf-8")
        )
        trace_rows = [
            DiagnosticTraceRowV1.from_mapping(json.loads(line))
            for line in (candidate.parent / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        by_pair: dict[str, list[DiagnosticTraceRowV1]] = {}
        for row in trace_rows:
            by_pair.setdefault(row.pair_id, []).append(row)
        interventions = {item.arm_id: item for item in design.interventions}
        banks_payload = json.loads(
            (candidate.parent / "donor_bank.json").read_text(encoding="utf-8")
        )
        banks = {
            name: DonorBankV1.from_mapping(value) for name, value in banks_payload.items()
        }
        donor_records = {
            (bank.modality, item.recipient_id, item.step_index): item
            for bank in banks.values()
            for item in bank.records
        }
        for pair in pairs_payload["pairs"]:
            pair_id = str(pair["pair_id"])
            if pair_id in observed_pair_ids:
                raise ValueError("diagnostic matrix contains duplicate pair IDs")
            observed_pair_ids.add(pair_id)
            rows = sorted(by_pair.get(pair_id, ()), key=lambda item: item.step_id)
            if not rows or [item.step_id for item in rows] != list(range(len(rows))):
                raise ValueError("diagnostic trace steps are missing or duplicated")
            arm = interventions[str(pair["arm"])]
            for row in rows:
                if row.route != pair["route"] or row.arm != pair["arm"]:
                    raise ValueError("trace route/arm does not match pair metadata")
                if arm.kind == "prompt":
                    changed = row.canonical_prompt_sha256 != row.effective_prompt_sha256
                    if (arm.operator == "control") == changed:
                        raise ValueError("prompt intervention did not affect every expected step")
                elif row.canonical_prompt_sha256 != row.effective_prompt_sha256:
                    raise ValueError("image intervention unexpectedly changed prompt bytes")
                if arm.operator == "shuffle":
                    if row.donor_id is None or row.donor_content_sha256 is None:
                        raise ValueError("shuffle trace is missing donor provenance")
                    donor_key = (
                        "instruction" if arm.kind == "prompt" else "image",
                        row.episode_id,
                        row.step_id if arm.kind == "image" else None,
                    )
                    donor_record = donor_records.get(donor_key)
                    if (
                        donor_record is None
                        or donor_record.donor_id != row.donor_id
                        or donor_record.donor_content_sha256
                        != row.donor_content_sha256
                    ):
                        raise ValueError("shuffle donor hash does not match donor bank")
                elif row.donor_id is not None or row.donor_content_sha256 is not None:
                    raise ValueError("non-shuffle trace unexpectedly declares a donor")
            recomputed_pairs[pair_id] = {
                "pair_id": pair_id,
                "train_seed": pair["train_seed"],
                "evaluation_seed": pair["evaluation_seed"],
                "route": pair["route"],
                "arm": pair["arm"],
                "total_reward": float(sum(item.reward for item in rows)),
                "success": bool(rows[-1].success),
                "steps": len(rows),
            }
        manifests.append(parsed)
    matrix = json.loads((root / "matrix.json").read_text(encoding="utf-8"))
    pair_ids = matrix.get("pair_ids")
    expected_pair_ids = {
        _pair_id(train_seed, evaluation_seed, route.mode, arm.arm_id)
        for train_seed in design.train_seeds
        for route in design.routes
        for arm in design.interventions
        for evaluation_seed in design.evaluation_seeds
    }
    complete = (
        isinstance(pair_ids, list)
        and len(pair_ids) == evidence.expected_pairs
        and len(pair_ids) == len(set(pair_ids))
        and evidence.observed_pairs == evidence.expected_pairs
        and evidence.matrix_complete
        and set(pair_ids) == expected_pair_ids == observed_pair_ids
    )
    if not complete:
        raise ValueError("diagnostic matrix is incomplete or contains duplicate pairs")
    with (root / "per_pair.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = {row["pair_id"]: row for row in csv.DictReader(stream)}
    if set(csv_rows) != set(recomputed_pairs):
        raise ValueError("per-pair CSV does not match trace pairs")
    for pair_id, expected_row in recomputed_pairs.items():
        csv_row = csv_rows[pair_id]
        if (
            int(csv_row["train_seed"]) != expected_row["train_seed"]
            or int(csv_row["evaluation_seed"]) != expected_row["evaluation_seed"]
            or csv_row["route"] != expected_row["route"]
            or csv_row["arm"] != expected_row["arm"]
            or abs(float(csv_row["total_reward"]) - expected_row["total_reward"]) > 1e-9
            or (csv_row["success"] == "True") != expected_row["success"]
            or int(csv_row["steps"]) != expected_row["steps"]
        ):
            raise ValueError("per-pair CSV metrics do not recompute from traces")
    sha_homogeneous = len({item.git_sha for item in manifests}) == 1
    dependency_homogeneous = len({item.dependency_lock_sha256 for item in manifests}) == 1
    config_homogeneous = all(
        item.cell_contract_sha256 == evidence.cell_contract_sha256
        for item in manifests
    )
    homogeneous = (
        sha_homogeneous
        and dependency_homogeneous
        and config_homogeneous
        and len({item.policy_id for item in manifests}) == 1
    )
    if homogeneous != evidence.homogeneous:
        raise ValueError("diagnostic homogeneity result does not match source manifests")
    parity_ok = all(item.parity_verified for item in manifests)
    dirty = any(item.git_dirty for item in manifests)
    reasons = ["beta1_framework_only"]
    if design.reduced_design:
        reasons.append("reduced_design")
    if dirty:
        reasons.append("dirty_source")
    if not sha_homogeneous:
        reasons.append("mixed_sha")
    if not dependency_homogeneous:
        reasons.append("mixed_dependency")
    if not config_homogeneous:
        reasons.append("config_drift")
    if not complete:
        reasons.append("incomplete_matrix")
    if not parity_ok:
        reasons.append("parity_failure")
    if tuple(sorted(reasons)) != evidence.gate_reasons:
        raise ValueError("evidence gate reasons do not match recomputed sources")
    statement_allowed = complete and homogeneous and parity_ok and not dirty
    if (
        evidence.framework_statement_allowed != statement_allowed
        or evidence.release_eligible != statement_allowed
        or evidence.allowed_wording
        != (_ALLOWED_WORDING if statement_allowed else None)
    ):
        raise ValueError("evidence release/statement gates do not recompute")
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
    trace_rows = [
        DiagnosticTraceRowV1.from_mapping(json.loads(line))
        for trace in sorted(source.glob("runs/*/*/trace.jsonl"))
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line
    ]
    failure_counts: dict[str, int] = {}
    origin_counts: dict[str, int] = {}
    for row in trace_rows:
        for failure in row.failures:
            key = f"{failure.layer}/{failure.provenance}/{failure.label}"
            failure_counts[key] = failure_counts.get(key, 0) + 1
        if row.error_origin is not None:
            origin_counts[row.error_origin] = origin_counts.get(row.error_origin, 0) + 1
    parity_count = sum(
        PromptParityManifestV1.from_mapping(
            json.loads(path.read_text(encoding="utf-8"))
        ).verified
        for path in source.glob("runs/*/*/parity.json")
    )
    thumbnail_records: list[dict[str, Any]] = []
    if (source / "thumbnails.json").is_file():
        thumbnail_payload = json.loads(
            (source / "thumbnails.json").read_text(encoding="utf-8")
        )
        thumbnail_records = list(thumbnail_payload["records"])
        thumbnail_target = target / "thumbnails"
        thumbnail_target.mkdir(exist_ok=True)
        for record in thumbnail_records:
            shutil.copy2(source / record["path"], thumbnail_target / Path(record["path"]).name)
    summary = {
        "evidence": evidence,
        "aggregate": aggregate,
        "parity_verified_cells": parity_count,
        "failure_counts": failure_counts,
        "error_origin_counts": origin_counts,
        "thumbnails": thumbnail_records,
    }
    _write_json(target / "summary.json", summary)
    shutil.copy2(source / "per_pair.csv", target / "per_pair.csv")
    with (source / "per_pair.csv").open(encoding="utf-8", newline="") as stream:
        pair_rows = list(csv.DictReader(stream))
    wording = html.escape(
        str(evidence["allowed_wording"])
        if evidence["allowed_wording"] is not None
        else "Framework statement unavailable; inspect gate reasons."
    )
    gates = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in evidence["gate_reasons"]
    )
    failures = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{count}</td></tr>"
        for key, count in sorted(failure_counts.items())
    ) or "<tr><td>none</td><td>0</td></tr>"
    pairs = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row[name]))}</td>"
            for name in ("train_seed", "evaluation_seed", "route", "arm", "success", "steps")
        )
        + "</tr>"
        for row in pair_rows
    )
    thumbnails = "".join(
        f'<figure><img src="thumbnails/{html.escape(Path(record["path"]).name)}" '
        f'width="128" height="128" alt="synthetic {html.escape(record["arm"])} '
        f'seed {record["seed"]}"><figcaption>seed {record["seed"]}: '
        f'{html.escape(record["arm"])}</figcaption></figure>'
        for record in thumbnail_records
    )
    body = (
        "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
        "<title>LunaVLA v3 diagnostic report</title>"
        f"<h1>LunaVLA v3 diagnostic report</h1><p>{wording}</p>"
        f"<p>Pairs: {evidence['observed_pairs']} / {evidence['expected_pairs']}</p>"
        f"<p>Parity verified cells: {parity_count}</p>"
        "<p>Scientific claims allowed: false</p>"
        f"<h2>Gate reasons</h2><ul>{gates}</ul>"
        f"<h2>Failure taxonomy</h2><table><tr><th>layer/provenance/label</th>"
        f"<th>count</th></tr>{failures}</table>"
        "<h2>Per-pair results</h2><table><tr><th>train seed</th><th>eval seed</th>"
        f"<th>route</th><th>arm</th><th>success</th><th>steps</th></tr>{pairs}</table>"
        f"<h2>Synthetic thumbnails</h2>{thumbnails}</html>\n"
    )
    (target / "index.html").write_text(body, encoding="utf-8")
    return target
