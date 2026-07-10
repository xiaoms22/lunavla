"""Read-only verification for small, tracked v2 evidence snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from lunavla.config import ExperimentConfig
from lunavla.evidence import EvidenceManifest, EvidenceStatistic
from lunavla.evidence_design import EvidenceDesign
from lunavla.evidence_runner import (
    _aggregate,
    _evaluation_fixture,
    _read_pair_rows,
    _resolved_job_config,
    _validate_canonical_full_config,
    derive_plan,
    is_full_design,
)
from lunavla.manifest import RunManifest


SNAPSHOT_MANIFEST_SCHEMA_VERSION = 1
INSTRUCTION_FOLLOWING_DENIED = "Instruction-following has not yet been established."
VISUAL_CONTROL_CONTRIBUTION_DENIED = "Visual-control contribution has not yet been established."
PUBLISHED_LANGUAGE_GIT_SHA = "a546695445f6fa6e717cd560d5acf718e037940a"
PUBLISHED_LANGUAGE_EVIDENCE_SHA256 = (
    "106ea2421d37c6c374e31d01a788101e358317f76b6abc315318634e6c6fa3b8"
)
PUBLISHED_LANGUAGE_SNAPSHOT_SHA256 = (
    "a266c2fe85cd7ca83728d4ab597b7e2c5621e562bfbae66aa4f586161eb7d4f7"
)
PUBLISHED_LANGUAGE_WORKFLOW_URL = "https://github.com/xiaoms22/lunavla/actions/runs/29106885353"
PUBLISHED_VISUAL_GIT_SHA = "bf0e550a7aa3fb0bb07354cd7cb525752c56268d"
PUBLISHED_VISUAL_EVIDENCE_SHA256 = (
    "d8ff8c798a6810a09a2905dbafd6f5259ac2356623ee6060d335d660db6e9056"
)
PUBLISHED_VISUAL_SNAPSHOT_SHA256 = (
    "8ac8b2d77c46ed67ac8d78c11e11d8f8cd7312b9eabd9a2ea025c830c6391d62"
)
PUBLISHED_VISUAL_WORKFLOW_URL = "https://github.com/xiaoms22/lunavla/actions/runs/29110701437"

_SNAPSHOT_FIELDS = {
    "schema_version",
    "verification",
    "files",
    "source_evidence_manifest_sha256",
}
_VERIFICATION_FIELDS = {
    "arm_episode_count",
    "design_id",
    "design_sha256",
    "git_sha",
    "reduced_design",
    "source_count",
    "verified",
}
_REPRODUCIBILITY_FIELDS = {
    "checkpoint_sha256",
    "config_sha256",
    "data_sha256",
    "metrics_sha256",
    "original_run_id",
    "repeat_run_id",
    "required",
    "verified",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


@dataclass(frozen=True)
class PublishedLanguageEvidence:
    """Validated values that may be rendered into public documentation."""

    evidence_manifest_path: Path
    snapshot_manifest_path: Path
    evidence_manifest_sha256: str
    git_sha: str
    train_seed_count: int
    control_trials: int
    arm_wilson: tuple[EvidenceStatistic, ...]
    counterfactual_final_distance: EvidenceStatistic
    counterfactual_success: EvidenceStatistic
    failed_checks: tuple[str, ...]
    workflow_url: str
    statement: str


@dataclass(frozen=True)
class PublishedVisualEvidence:
    """Validated visual-study values that may be rendered publicly."""

    evidence_manifest_path: Path
    snapshot_manifest_path: Path
    evidence_manifest_sha256: str
    git_sha: str
    train_seed_count: int
    control_trials: int
    arm_wilson: tuple[EvidenceStatistic, ...]
    paired_final_distance: tuple[EvidenceStatistic, ...]
    failed_checks: tuple[str, ...]
    workflow_url: str
    statement: str


@dataclass(frozen=True)
class _VerifiedPublishedEvidence:
    evidence_manifest_path: Path
    snapshot_manifest_path: Path
    evidence_manifest_sha256: str
    git_sha: str
    design: EvidenceDesign
    evidence: EvidenceManifest


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    return dict(value)


def _exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ValueError(f"unknown field(s) in {name}: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing field(s) in {name}: {', '.join(missing)}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json_value(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc


def _read_json(path: Path, name: str) -> dict[str, Any]:
    return _mapping(_read_json_value(path), name)


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _snapshot_relative_path(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("snapshot file paths must be strings")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or value == "snapshot_manifest.json"
    ):
        raise ValueError(f"unsafe or non-normalized snapshot file path: {value!r}")
    return value


def _verify_file_inventory(root: Path, raw_files: object) -> dict[str, str]:
    files_payload = _mapping(raw_files, "snapshot files")
    files: dict[str, str] = {}
    for raw_path, raw_digest in files_payload.items():
        relative = _snapshot_relative_path(raw_path)
        files[relative] = _sha256(raw_digest, f"files.{relative}")
    if not files:
        raise ValueError("snapshot files cannot be empty")

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"snapshot cannot contain symbolic links: {path.relative_to(root)}")
        if path.is_file() and path != root / "snapshot_manifest.json":
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files):
        missing = sorted(set(files) - actual)
        unlisted = sorted(actual - set(files))
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if unlisted:
            detail.append(f"unlisted={unlisted}")
        raise ValueError("snapshot file inventory mismatch: " + ", ".join(detail))

    for relative, expected in sorted(files.items()):
        path = root.joinpath(*PurePosixPath(relative).parts)
        if _sha256_file(path) != expected:
            raise ValueError(f"snapshot file hash mismatch: {relative}")
    return files


def _verify_reproducibility(
    payload: Mapping[str, Any],
    *,
    root: Path,
    reproducibility_run_id: str | None,
    manifests: Mapping[str, RunManifest],
) -> bool:
    _exact_fields(payload, _REPRODUCIBILITY_FIELDS, "reproducibility")
    if _boolean(payload.get("required"), "reproducibility.required") is not True:
        raise ValueError("full evidence requires a reproducibility sentinel")
    if _boolean(payload.get("verified"), "reproducibility.verified") is not True:
        raise ValueError("reproducibility sentinel is not verified")
    if reproducibility_run_id is None:
        raise ValueError("full evidence has no reproducibility run")
    repeat_run_id = f"{reproducibility_run_id}-repeat"
    if (
        payload.get("original_run_id") != reproducibility_run_id
        or payload.get("repeat_run_id") != repeat_run_id
    ):
        raise ValueError("reproducibility run IDs differ from the derived plan")
    original = manifests[reproducibility_run_id]
    repeat_root = root / "reproducibility" / repeat_run_id
    repeat = RunManifest.load(repeat_root / "manifest.json")
    if repeat.to_dict() != original.to_dict():
        raise ValueError("repeat RunManifest differs from the original run")
    original_metrics_path = root / "runs" / reproducibility_run_id / "metrics.json"
    repeat_metrics_path = repeat_root / "metrics.json"
    if _read_json_value(repeat_metrics_path) != _read_json_value(original_metrics_path):
        raise ValueError("repeat metrics differ from the original run")
    expected_pairs = {
        "checkpoint_sha256": (original.checkpoint_sha256, repeat.checkpoint_sha256),
        "config_sha256": (original.config_sha256, repeat.config_sha256),
        "data_sha256": (original.data_sha256, repeat.data_sha256),
        "metrics_sha256": (
            _sha256_file(original_metrics_path),
            _sha256_file(repeat_metrics_path),
        ),
    }
    for name, expected in expected_pairs.items():
        values = payload.get(name)
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"reproducibility.{name} must contain the original and repeat")
        left = _sha256(values[0], f"reproducibility.{name}[0]")
        right = _sha256(values[1], f"reproducibility.{name}[1]")
        if (left, right) != expected or left != right:
            raise ValueError(f"reproducibility.{name} does not match")
    if (
        original.artifact_sha256.get("metrics.json") != expected_pairs["metrics_sha256"][0]
        or repeat.artifact_sha256.get("metrics.json") != expected_pairs["metrics_sha256"][1]
    ):
        raise ValueError("reproducibility metrics are not bound by RunManifest")
    return True


def _verify_snapshot(
    snapshot_root: str | Path,
    *,
    suite: str,
    published_git_sha: str,
    published_evidence_sha256: str,
    published_snapshot_sha256: str,
    claim_id: str,
    denied_statement: str,
) -> _VerifiedPublishedEvidence:
    """Verify a publication snapshot without changing it or any source artifact."""

    raw_root = Path(snapshot_root)
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("snapshot root must be a real directory")
    root = raw_root.resolve()
    snapshot_path = root / "snapshot_manifest.json"
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        raise ValueError("snapshot_manifest.json must be a real file")
    snapshot_digest = _sha256_file(snapshot_path)
    if snapshot_digest != published_snapshot_sha256:
        raise ValueError(
            "snapshot manifest does not match the registered official workflow artifact"
        )
    snapshot = _read_json(snapshot_path, "snapshot manifest")
    _exact_fields(snapshot, _SNAPSHOT_FIELDS, "snapshot manifest")
    if (
        _integer(snapshot["schema_version"], "snapshot schema_version")
        != SNAPSHOT_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported snapshot schema_version: {snapshot['schema_version']}")
    files = _verify_file_inventory(root, snapshot["files"])

    evidence_path = root / "evidence_manifest.json"
    evidence_digest = _sha256_file(evidence_path)
    if evidence_digest != published_evidence_sha256:
        raise ValueError(
            "EvidenceManifest does not match the registered official workflow artifact"
        )
    recorded_evidence_digest = _sha256(
        snapshot["source_evidence_manifest_sha256"],
        "source_evidence_manifest_sha256",
    )
    if files.get("evidence_manifest.json") != evidence_digest:
        raise ValueError("snapshot files do not bind evidence_manifest.json")
    if recorded_evidence_digest != evidence_digest:
        raise ValueError("source evidence manifest hash mismatch")

    evidence_payload = _read_json(evidence_path, "evidence manifest")
    evidence = EvidenceManifest.from_mapping(evidence_payload)
    verification = _mapping(snapshot["verification"], "snapshot verification")
    _exact_fields(verification, _VERIFICATION_FIELDS, "snapshot verification")
    if _boolean(verification["verified"], "verification.verified") is not True:
        raise ValueError("snapshot verification must be true")
    reduced_design = _boolean(verification["reduced_design"], "verification.reduced_design")
    source_count = _integer(
        verification["source_count"], "verification.source_count", positive=True
    )
    arm_episode_count = _integer(
        verification["arm_episode_count"],
        "verification.arm_episode_count",
        positive=True,
    )
    git_sha = verification["git_sha"]
    if not isinstance(git_sha, str) or not _GIT_SHA.fullmatch(git_sha):
        raise ValueError("verification.git_sha must be a full lowercase Git object ID")
    if git_sha != published_git_sha:
        raise ValueError("snapshot Git SHA does not match the publication registry")
    if (
        verification["design_id"] != evidence.design_id
        or verification["design_sha256"] != evidence.design_sha256
        or reduced_design != evidence.reduced_design
        or source_count != len(evidence.sources)
    ):
        raise ValueError("snapshot and evidence manifest design identity differ")
    if reduced_design or not evidence.matrix_complete:
        raise ValueError("published evidence must use the complete design")
    if not all(passed for _, passed in evidence.integrity_checks):
        raise ValueError("published evidence contains a failed integrity check")

    design = EvidenceDesign.load(root / "design.yaml")
    if design.suite != suite or not is_full_design(design):
        raise ValueError(f"snapshot does not contain the canonical full {suite} design")
    if design.design_id != evidence.design_id or design.sha256() != evidence.design_sha256:
        raise ValueError("EvidenceDesign identity/hash differs from the evidence manifest")
    plan = derive_plan(design)
    if arm_episode_count != plan.expected_arm_episodes:
        raise ValueError("snapshot arm episode count differs from the derived design")
    if source_count != plan.expected_training_runs:
        raise ValueError("snapshot source count differs from the derived design")

    base_payload = _read_json(root / "base_config.json", "base config")
    base_config = ExperimentConfig.from_mapping(base_payload)
    expected_plan = {**plan.to_dict(), "base_config_sha256": base_config.sha256()}
    if _canonical_json(_read_json(root / "plan.json", "evidence plan")) != _canonical_json(
        expected_plan
    ):
        raise ValueError("plan.json differs from the plan derived from EvidenceDesign")

    jobs = {job.run_id: job for job in plan.jobs}
    sources = {source.run_id: source for source in evidence.sources}
    if set(sources) != set(jobs):
        raise ValueError("source manifests differ from the derived training matrix")
    manifests: list[RunManifest] = []
    manifests_by_run: dict[str, RunManifest] = {}
    dependency_sets: set[bytes] = set()
    non_image_pairing_hashes: set[str] = set()
    data_hashes_by_variant: dict[str, set[str]] = {}
    split_records: set[bytes] = set()
    fixtures_by_run: dict[str, dict[int, dict[str, Any]]] = {}
    rollout_samples: dict[str, list[dict[str, Any]]] = {}
    for run_id, job in sorted(jobs.items()):
        source = sources[run_id]
        relative = f"runs/{run_id}/manifest.json"
        if files.get(relative) != source.manifest_sha256:
            raise ValueError(f"source manifest hash mismatch: {run_id}")
        manifest_path = root / relative
        _read_json(manifest_path, f"source manifest {run_id}")
        manifest = RunManifest.load(manifest_path)
        manifests.append(manifest)
        manifests_by_run[run_id] = manifest
        if manifest.run_id != run_id:
            raise ValueError(f"source run_id mismatch: {run_id}")
        if manifest.git_sha != git_sha:
            raise ValueError(f"source Git SHA mismatch: {run_id}")
        if manifest.git_dirty or manifest.source_diff_sha256 is not None:
            raise ValueError(f"dirty source cannot be published evidence: {run_id}")
        if manifest.design_id != design.design_id or manifest.design_sha256 != design.sha256():
            raise ValueError(f"source design identity mismatch: {run_id}")
        if manifest.train_seeds != [job.train_seed]:
            raise ValueError(f"source train seed mismatch: {run_id}")
        if manifest.data_seeds != [design.seeds.data]:
            raise ValueError(f"source data seed mismatch: {run_id}")
        if tuple(manifest.eval_seeds) != design.seeds.evaluation:
            raise ValueError(f"source evaluation seeds mismatch: {run_id}")
        if (
            manifest.condition.get("variant") != job.variant
            or manifest.condition.get("train_seed") != job.train_seed
            or tuple(manifest.condition.get("arm_ids", ())) != job.arm_ids
        ):
            raise ValueError(f"source condition differs from the design: {run_id}")
        expected_config = _resolved_job_config(design, base_config, job)
        _validate_canonical_full_config(
            expected_config,
            suite=design.suite,
            variant=job.variant,
        )
        if (
            manifest.config != expected_config.to_dict()
            or manifest.config_sha256 != expected_config.sha256()
        ):
            raise ValueError(f"source config differs from EvidenceDesign: {run_id}")
        run_root = root / "runs" / run_id
        resolved_path = run_root / "resolved_config.json"
        metrics_path = run_root / "metrics.json"
        donor_path = run_root / "donor_bank.json"
        artifact_paths = {
            "resolved_config.json": resolved_path,
            "metrics.json": metrics_path,
            "donor_bank.json": donor_path,
        }
        for artifact_name, artifact_path in artifact_paths.items():
            if _sha256_file(artifact_path) != manifest.artifact_sha256.get(artifact_name):
                raise ValueError(
                    f"snapshot artifact is not bound by RunManifest: {run_id}/{artifact_name}"
                )
        if _read_json(resolved_path, f"resolved config {run_id}") != expected_config.to_dict():
            raise ValueError(f"resolved_config.json differs from EvidenceDesign: {run_id}")
        if _read_json(metrics_path, f"metrics {run_id}") != manifest.metrics:
            raise ValueError(f"metrics.json differs from RunManifest: {run_id}")
        donor_hash = _sha256_file(donor_path)
        if manifest.paired_data.get("donor_bank_sha256") != donor_hash:
            raise ValueError(f"donor bank differs from paired-data contract: {run_id}")
        expected_fixture = _evaluation_fixture(expected_config, design)
        if (
            manifest.eval_fixture.get("evidence_fixture_sha256") != expected_fixture["sha256"]
            or manifest.eval_fixture.get("evidence_episodes") != expected_fixture["episodes"]
        ):
            raise ValueError(f"evaluation fixture differs from EvidenceDesign: {run_id}")
        fixtures_by_run[run_id] = {
            int(item["eval_seed"]): dict(item) for item in expected_fixture["episodes"]
        }
        expected_arms = [
            next(arm for arm in design.arms if arm.id == arm_id).to_dict() for arm_id in job.arm_ids
        ]
        if manifest.arms != expected_arms:
            raise ValueError(f"source arms differ from EvidenceDesign: {run_id}")
        expected_pairs = [
            {
                "pair_id": f"{job.train_seed}:{eval_seed}",
                "train_seed": job.train_seed,
                "eval_seed": eval_seed,
                "arms": list(job.arm_ids),
            }
            for eval_seed in design.seeds.evaluation
        ]
        if _canonical_json(manifest.pairs) != _canonical_json(expected_pairs):
            raise ValueError(f"source pairs differ from EvidenceDesign: {run_id}")
        non_image_hash = _sha256(
            manifest.paired_data.get("non_image_pairing_sha256"),
            f"paired_data.non_image_pairing_sha256 {run_id}",
        )
        non_image_pairing_hashes.add(non_image_hash)
        data_hashes_by_variant.setdefault(job.variant, set()).add(manifest.data_sha256)
        split_records.add(_canonical_json(manifest.dataset_split))
        sample_payload = _read_json_value(run_root / "rollouts.sample.json")
        if not isinstance(sample_payload, list) or len(sample_payload) != 2:
            raise ValueError(f"rollouts.sample.json must contain two entries: {run_id}")
        if any(not isinstance(item, Mapping) for item in sample_payload):
            raise TypeError(f"rollouts.sample.json entries must be objects: {run_id}")
        rollout_samples[run_id] = [dict(item) for item in sample_payload]
        dependency_sets.add(
            json.dumps(
                manifest.dependencies,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
    if len(dependency_sets) != 1:
        raise ValueError("source manifests do not share one dependency set")
    if len(non_image_pairing_hashes) != 1:
        raise ValueError("source manifests do not share one paired dataset")
    if (
        any(len(hashes) != 1 for hashes in data_hashes_by_variant.values())
        or set(data_hashes_by_variant) != {job.variant for job in plan.jobs}
        or len(split_records) != 1
    ):
        raise ValueError("source manifests do not share one data contract per variant/split")

    rows = _read_pair_rows(root / "per_pair.csv")
    if len(rows) != arm_episode_count:
        raise ValueError("per_pair.csv row count differs from snapshot verification")
    arms = {arm.id: arm for arm in design.arms}
    cells: set[tuple[int, int, str]] = set()
    rows_by_cell: dict[str, dict[str, Any]] = {}
    for row in rows:
        train_seed = int(row["train_seed"])
        eval_seed = int(row["eval_seed"])
        arm_id = str(row["arm_id"])
        row_job = jobs.get(str(row["run_id"]))
        cell = (train_seed, eval_seed, arm_id)
        if (
            row_job is None
            or train_seed != row_job.train_seed
            or eval_seed not in design.seeds.evaluation
            or arm_id not in row_job.arm_ids
            or cell in cells
        ):
            raise ValueError(f"unexpected or duplicate per-pair cell: {cell}")
        cells.add(cell)
        if row["run_id"] != row_job.run_id:
            raise ValueError(f"per-pair run_id mismatch: {cell}")
        pair_id = f"{train_seed}:{eval_seed}"
        if row["pair_id"] != pair_id or row["cell_id"] != f"{pair_id}:{arm_id}":
            raise ValueError(f"per-pair identifiers are not aligned: {cell}")
        if row["arm_mode"] != arms[arm_id].mode:
            raise ValueError(f"per-pair arm mode mismatch: {cell}")
        fixture = fixtures_by_run[row_job.run_id].get(eval_seed)
        fixture_family = None
        if fixture is not None:
            fixture_family = fixture.get("task_id", fixture.get("family"))
        if fixture_family is None or row["family"] != fixture_family:
            raise ValueError(f"per-pair task family differs from eval fixture: {cell}")
        rows_by_cell[str(row["cell_id"])] = row

    sampled_cells: set[str] = set()
    for run_id, samples in sorted(rollout_samples.items()):
        for sample in samples:
            required = {
                "cell_id",
                "pair_id",
                "arm_id",
                "seed",
                "task",
                "success",
                "final_distance",
                "first_action_mse",
                "total_reward",
                "steps",
            }
            if not required.issubset(sample):
                raise ValueError(f"rollout sample is missing evidence fields: {run_id}")
            cell_id = str(sample["cell_id"])
            if cell_id in sampled_cells:
                raise ValueError(f"duplicate rollout sample cell: {cell_id}")
            sampled_cells.add(cell_id)
            paired_row = rows_by_cell.get(cell_id)
            if paired_row is None or paired_row["run_id"] != run_id:
                raise ValueError(f"rollout sample has no matching per-pair row: {cell_id}")
            expected_values = {
                "pair_id": paired_row["pair_id"],
                "arm_id": paired_row["arm_id"],
                "seed": paired_row["eval_seed"],
                "task": (
                    paired_row["family"]
                    if suite == "language"
                    else manifests_by_run[run_id].task_id
                ),
                "success": bool(paired_row["success"]),
                "final_distance": paired_row["final_distance"],
                "first_action_mse": paired_row["first_action_mse"],
                "total_reward": paired_row["total_reward"],
                "steps": paired_row["steps"],
            }
            actual_values = {name: sample[name] for name in expected_values}
            if actual_values != expected_values:
                raise ValueError(f"rollout sample differs from per-pair row: {cell_id}")

    reproducibility = _read_json(root / "reproducibility.json", "reproducibility record")
    reproducibility_verified = _verify_reproducibility(
        reproducibility,
        root=root,
        reproducibility_run_id=plan.reproducibility_run_id,
        manifests=manifests_by_run,
    )
    recomputed = _aggregate(
        design,
        plan,
        manifests,
        rows,
        evidence.sources,
        reproducibility_verified=reproducibility_verified,
    )
    if recomputed.to_dict() != evidence.to_dict():
        raise ValueError("evidence statistics or claim gates differ from source data")
    expected_aggregate = {
        "statistics": [statistic.to_dict() for statistic in evidence.statistics],
        "claims": [claim.to_dict() for claim in evidence.claims],
    }
    if _read_json(root / "aggregate.json", "aggregate projection") != expected_aggregate:
        raise ValueError("aggregate.json differs from EvidenceManifest")

    if len(evidence.claims) != 1 or evidence.claims[0].claim_id != claim_id:
        raise ValueError(f"{suite} evidence must contain exactly one {claim_id} claim gate")
    claim = evidence.claims[0]
    if claim.allowed or claim.statement != denied_statement or not claim.failed_checks:
        raise ValueError(f"{claim_id} claim must remain closed for this snapshot")

    return _VerifiedPublishedEvidence(
        evidence_manifest_path=evidence_path,
        snapshot_manifest_path=snapshot_path,
        evidence_manifest_sha256=evidence_digest,
        git_sha=git_sha,
        design=design,
        evidence=evidence,
    )


def verify_language_snapshot(snapshot_root: str | Path) -> PublishedLanguageEvidence:
    """Verify the registered language snapshot and expose only publication-safe values."""

    verified = _verify_snapshot(
        snapshot_root,
        suite="language",
        published_git_sha=PUBLISHED_LANGUAGE_GIT_SHA,
        published_evidence_sha256=PUBLISHED_LANGUAGE_EVIDENCE_SHA256,
        published_snapshot_sha256=PUBLISHED_LANGUAGE_SNAPSHOT_SHA256,
        claim_id="instruction_following",
        denied_statement=INSTRUCTION_FOLLOWING_DENIED,
    )
    evidence = verified.evidence
    design = verified.design
    claim = evidence.claims[0]
    statistics = {statistic.statistic_id: statistic for statistic in evidence.statistics}
    arm_wilson = tuple(statistics[f"{arm.id}-success"] for arm in design.arms)
    control_trials = statistics["control-success"].sample_n
    if any(
        statistic.method != "wilson" or statistic.sample_n != control_trials
        for statistic in arm_wilson
    ):
        raise ValueError("arm Wilson statistics do not use one paired trial matrix")
    counterfactual_distance = statistics["counterfactual-all-final_distance"]
    counterfactual_success = statistics["counterfactual-all-success"]
    if (
        counterfactual_distance.sample_n != control_trials
        or counterfactual_success.sample_n != control_trials
        or counterfactual_distance.train_seed_n != len(design.seeds.train)
        or counterfactual_success.train_seed_n != len(design.seeds.train)
    ):
        raise ValueError("counterfactual paired statistics differ from the design matrix")

    return PublishedLanguageEvidence(
        evidence_manifest_path=verified.evidence_manifest_path,
        snapshot_manifest_path=verified.snapshot_manifest_path,
        evidence_manifest_sha256=verified.evidence_manifest_sha256,
        git_sha=verified.git_sha,
        train_seed_count=len(design.seeds.train),
        control_trials=control_trials,
        arm_wilson=arm_wilson,
        counterfactual_final_distance=counterfactual_distance,
        counterfactual_success=counterfactual_success,
        failed_checks=claim.failed_checks,
        workflow_url=PUBLISHED_LANGUAGE_WORKFLOW_URL,
        statement=claim.statement,
    )


def verify_visual_snapshot(snapshot_root: str | Path) -> PublishedVisualEvidence:
    """Verify the registered visual snapshot and expose only publication-safe values."""

    verified = _verify_snapshot(
        snapshot_root,
        suite="visual",
        published_git_sha=PUBLISHED_VISUAL_GIT_SHA,
        published_evidence_sha256=PUBLISHED_VISUAL_EVIDENCE_SHA256,
        published_snapshot_sha256=PUBLISHED_VISUAL_SNAPSHOT_SHA256,
        claim_id="visual_control_contribution",
        denied_statement=VISUAL_CONTROL_CONTRIBUTION_DENIED,
    )
    evidence = verified.evidence
    design = verified.design
    claim = evidence.claims[0]
    statistics = {statistic.statistic_id: statistic for statistic in evidence.statistics}
    arm_wilson = tuple(statistics[f"{arm.id}-success"] for arm in design.arms)
    control_trials = statistics["control-success"].sample_n
    if any(
        statistic.method != "wilson" or statistic.sample_n != control_trials
        for statistic in arm_wilson
    ):
        raise ValueError("visual arm Wilson statistics do not use one paired trial matrix")
    paired_final_distance = tuple(
        statistics[f"{arm_id}-{scope}-final_distance"]
        for arm_id in ("occlusion", "state_only")
        for scope in ("all", "direct_reach", "waypoint_reach")
    )
    train_seed_count = len(design.seeds.train)
    if any(
        statistic.method != "hierarchical_paired_bootstrap"
        or statistic.train_seed_n != train_seed_count
        or statistic.sample_n
        != (control_trials if statistic.scope.endswith(":all") else control_trials // 2)
        for statistic in paired_final_distance
    ):
        raise ValueError("visual paired statistics differ from the design matrix")

    return PublishedVisualEvidence(
        evidence_manifest_path=verified.evidence_manifest_path,
        snapshot_manifest_path=verified.snapshot_manifest_path,
        evidence_manifest_sha256=verified.evidence_manifest_sha256,
        git_sha=verified.git_sha,
        train_seed_count=train_seed_count,
        control_trials=control_trials,
        arm_wilson=arm_wilson,
        paired_final_distance=paired_final_distance,
        failed_checks=claim.failed_checks,
        workflow_url=PUBLISHED_VISUAL_WORKFLOW_URL,
        statement=claim.statement,
    )
