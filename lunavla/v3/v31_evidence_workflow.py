from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .artifacts import sha256_file
from .v31_evidence import (
    V31EvidenceDesignV1,
    V31EvidenceManifestV1,
    V31EvidenceRowV1,
    V31RepeatSentinelV1,
    v31_claim_passes,
    v31_expected_matrix_keys,
    v31_row_inventory_sha256,
    v31_statistics,
)


_ROOT = Path(__file__).resolve().parents[2]


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _tree_hash(root: Path) -> str:
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise ValueError("v3.1 evidence execution tree is empty")
    return _stable_hash(files)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(result) != 40:
        raise ValueError("v3.1 evidence requires a full Git SHA")
    return result


@dataclass(frozen=True)
class V31ExecutionBatchV1:
    rows: tuple[V31EvidenceRowV1, ...]
    checkpoints_sha256: str
    metrics_sha256: str
    feature_source: str
    seed_checkpoint_sha256: dict[int, str]
    seed_metrics_sha256: dict[int, str]

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("v3.1 execution batch requires rows")
        if self.feature_source not in {"real_frozen_vlm", "deterministic_fixture"}:
            raise ValueError("v3.1 execution batch feature_source is invalid")
        for value in (
            self.checkpoints_sha256,
            self.metrics_sha256,
            *self.seed_checkpoint_sha256.values(),
            *self.seed_metrics_sha256.values(),
        ):
            if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
                raise ValueError("v3.1 execution batch hashes must be SHA-256")


class V31EvidenceExecutor(Protocol):
    def execute(
        self,
        design: V31EvidenceDesignV1,
        *,
        only_train_seed: int | None,
        output_dir: Path,
    ) -> V31ExecutionBatchV1: ...


class DeterministicV31FixtureExecutor:
    """Fast contract executor. Its output can never open the scientific claim."""

    def __init__(self, *, git_sha: str | None = None) -> None:
        self.git_sha = _git_sha() if git_sha is None else git_sha
        self.dependency_hash = sha256_file(_ROOT / "requirements-v3-vlm-cpu.lock")
        self.feature_hash = _stable_hash({"source": "deterministic_fixture", "version": 1})

    @staticmethod
    def _metrics(
        seed: int, arm: str, task: str, stratum: str, episode: int
    ) -> tuple[bool, float, float]:
        digest = hashlib.sha256(f"{seed}|{arm}|{task}|{stratum}|{episode}".encode()).digest()
        noise = int.from_bytes(digest[:4], "big") / (2**32 - 1)
        arm_offset = {
            "baseline": 0.00,
            "smol_control": 0.08,
            "feature_mask": -0.02,
            "feature_shuffle": -0.04,
        }[arm]
        difficulty = 0.08 if task == "failure_recovery" else 0.04
        score = 0.55 + arm_offset - difficulty + (noise - 0.5) * 0.20
        success = score >= 0.50
        final_distance = max(0.0, 0.18 - score * 0.14)
        first_action_mse = max(0.0, 0.12 - score * 0.07)
        return success, final_distance, first_action_mse

    def execute(
        self,
        design: V31EvidenceDesignV1,
        *,
        only_train_seed: int | None,
        output_dir: Path,
    ) -> V31ExecutionBatchV1:
        output_dir.mkdir(parents=True, exist_ok=False)
        seeds = design.train_seeds if only_train_seed is None else (only_train_seed,)
        rows: list[V31EvidenceRowV1] = []
        checkpoint_records: dict[str, str] = {}
        for seed in seeds:
            for arm in design.arms:
                checkpoint = _stable_hash(
                    {
                        "executor": "deterministic_fixture",
                        "seed": seed,
                        "training_arm": "baseline" if arm == "baseline" else "smol_control",
                    }
                )
                checkpoint_records[f"seed-{seed}/{arm}"] = checkpoint
                for task in design.task_ids:
                    for stratum in design.held_out_strata:
                        for episode in range(design.episodes_per_cell):
                            success, distance, mse = self._metrics(
                                seed, arm, task, stratum, episode
                            )
                            rows.append(
                                V31EvidenceRowV1(
                                    train_seed=seed,
                                    arm=arm,
                                    task_id=task,
                                    held_out_stratum=stratum,
                                    episode_index=episode,
                                    pair_id=f"{task}:{stratum}:{episode}",
                                    git_sha=self.git_sha,
                                    dependency_lock_sha256=self.dependency_hash,
                                    feature_source_sha256=self.feature_hash,
                                    checkpoint_sha256=checkpoint,
                                    success=success,
                                    final_distance=distance,
                                    first_action_mse=mse,
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
        _write_json(output_dir / "rows.json", [row.to_dict() for row in rows_tuple])
        _write_json(output_dir / "checkpoints.json", checkpoint_records)
        _write_json(
            output_dir / "execution.json",
            {
                "schema_version": 1,
                "feature_source": "deterministic_fixture",
                "checkpoints_sha256": checkpoints_hash,
                "metrics_sha256": metrics_hash,
                "seed_checkpoint_sha256": seed_checkpoint_hashes,
                "seed_metrics_sha256": seed_metrics_hashes,
            },
        )
        return V31ExecutionBatchV1(
            rows_tuple,
            checkpoints_hash,
            metrics_hash,
            "deterministic_fixture",
            seed_checkpoint_hashes,
            seed_metrics_hashes,
        )


def _expected_keys(
    design: V31EvidenceDesignV1, only_seed: int | None
) -> set[tuple[int, str, str, str, int]]:
    keys = set(v31_expected_matrix_keys(design))
    if only_seed is not None:
        keys = {key for key in keys if key[0] == only_seed}
    return keys


def _verify_batch(
    design: V31EvidenceDesignV1,
    batch: V31ExecutionBatchV1,
    only_seed: int | None,
) -> None:
    keys = tuple(row.matrix_key for row in batch.rows)
    if len(keys) != len(set(keys)) or set(keys) != _expected_keys(design, only_seed):
        raise ValueError("v3.1 executor did not produce the exact preregistered matrix")


def _execute(
    design: V31EvidenceDesignV1,
    output: Path,
    executor: V31EvidenceExecutor,
) -> V31EvidenceManifestV1:
    output.mkdir(parents=True, exist_ok=False)
    source = executor.execute(design, only_train_seed=None, output_dir=output / "source")
    repeat = executor.execute(design, only_train_seed=11, output_dir=output / "repeat-seed-11")
    _verify_batch(design, source, None)
    _verify_batch(design, repeat, 11)
    source_seed = tuple(row for row in source.rows if row.train_seed == 11)
    source_seed_hash = v31_row_inventory_sha256(source_seed)
    repeat_hash = v31_row_inventory_sha256(repeat.rows)
    sentinel = V31RepeatSentinelV1(
        train_seed=11,
        source_rows_sha256=source_seed_hash,
        repeat_rows_sha256=repeat_hash,
        source_checkpoints_sha256=source.seed_checkpoint_sha256[11],
        repeat_checkpoints_sha256=repeat.seed_checkpoint_sha256[11],
        source_metrics_sha256=source.seed_metrics_sha256[11],
        repeat_metrics_sha256=repeat.seed_metrics_sha256[11],
        verified=(
            source_seed_hash == repeat_hash
            and source.seed_checkpoint_sha256[11] == repeat.seed_checkpoint_sha256[11]
            and source.seed_metrics_sha256[11] == repeat.seed_metrics_sha256[11]
        ),
    )
    statistics = v31_statistics(source.rows, design)
    threshold_passed = v31_claim_passes(statistics)
    design_path = _write_json(output / "design.json", design.to_dict())
    rows_path = _write_json(output / "rows.json", [row.to_dict() for row in source.rows])
    repeat_path = _write_json(output / "repeat-rows.json", [row.to_dict() for row in repeat.rows])
    sentinel_path = _write_json(output / "sentinel.json", sentinel.to_dict())
    statistics_path = _write_json(output / "statistics.json", statistics)
    git_shas = {row.git_sha for row in source.rows}
    dependencies = {row.dependency_lock_sha256 for row in source.rows}
    features = {row.feature_source_sha256 for row in source.rows}
    homogeneous = len(git_shas) == len(dependencies) == len(features) == 1
    matrix_complete = len(source.rows) == design.expected_rows
    reasons: list[str] = []
    if not matrix_complete:
        reasons.append("incomplete_matrix")
    if not homogeneous:
        reasons.append("mixed_source")
    if not sentinel.verified:
        reasons.append("sentinel_failure")
    if source.feature_source == "deterministic_fixture":
        reasons.append("fixture_source")
    if not threshold_passed:
        reasons.append("claim_threshold_not_met")
    claim_gate = {
        "schema_version": 1,
        "claim_id": "frozen_vlm_feature_contribution",
        "threshold_passed": threshold_passed,
        "claim_allowed": not reasons,
        "gate_reasons": reasons,
        "failure_statement": "冻结 VLM 特征贡献尚未建立",
    }
    claim_path = _write_json(output / "claim-gate.json", claim_gate)
    manifest = V31EvidenceManifestV1(
        design_sha256=design.sha256(),
        row_inventory_sha256=v31_row_inventory_sha256(source.rows),
        sentinel_sha256=sentinel.sha256(),
        statistics_sha256=sha256_file(statistics_path),
        claim_gate_sha256=sha256_file(claim_path),
        git_sha=next(iter(git_shas)),
        feature_source=source.feature_source,
        expected_rows=design.expected_rows,
        observed_rows=len(source.rows),
        matrix_complete=matrix_complete,
        homogeneous=homogeneous,
        sentinel_verified=sentinel.verified,
        claim_allowed=not reasons,
        release_eligible=(
            matrix_complete
            and homogeneous
            and sentinel.verified
            and source.feature_source == "real_frozen_vlm"
        ),
        gate_reasons=tuple(reasons),
    )
    manifest_path = _write_json(output / "evidence-manifest.json", manifest.to_dict())
    _write_json(
        output / "artifact-inventory.json",
        {
            "schema_version": 1,
            "files": {
                path.name: sha256_file(path)
                for path in (
                    design_path,
                    rows_path,
                    repeat_path,
                    sentinel_path,
                    statistics_path,
                    claim_path,
                    manifest_path,
                )
            },
            "trees": {
                "source": _tree_hash(output / "source"),
                "repeat-seed-11": _tree_hash(output / "repeat-seed-11"),
            },
        },
    )
    verify_v31_evidence(output)
    return manifest


def run_v31_evidence(
    design_file: str | Path,
    output_dir: str | Path,
    executor: V31EvidenceExecutor,
    *,
    overwrite: bool = False,
) -> Path:
    design = V31EvidenceDesignV1.load(design_file)
    output = Path(output_dir).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError("v3.1 evidence output exists; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    backup = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
    try:
        _execute(design, staging, executor)
        if output.exists():
            output.replace(backup)
        staging.replace(output)
        if backup.exists():
            shutil.rmtree(backup)
        verify_v31_evidence(output)
        return output
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise


def verify_v31_evidence(output_dir: str | Path) -> V31EvidenceManifestV1:
    root = Path(output_dir).resolve()
    expected = {
        "artifact-inventory.json",
        "claim-gate.json",
        "design.json",
        "evidence-manifest.json",
        "repeat-rows.json",
        "repeat-seed-11",
        "rows.json",
        "sentinel.json",
        "source",
        "statistics.json",
    }
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise ValueError("v3.1 evidence file set is incomplete or contains extras")
    inventory = json.loads((root / "artifact-inventory.json").read_text())
    if set(inventory) != {"schema_version", "files", "trees"} or inventory["schema_version"] != 1:
        raise ValueError("v3.1 artifact inventory is invalid")
    for name, digest in inventory["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError(f"v3.1 evidence artifact hash mismatch: {name}")
    if set(inventory["trees"]) != {"source", "repeat-seed-11"}:
        raise ValueError("v3.1 execution tree inventory is invalid")
    for name, digest in inventory["trees"].items():
        if _tree_hash(root / name) != digest:
            raise ValueError(f"v3.1 evidence execution tree hash mismatch: {name}")
    design = V31EvidenceDesignV1.from_mapping(json.loads((root / "design.json").read_text()))
    rows = tuple(
        V31EvidenceRowV1.from_mapping(item) for item in json.loads((root / "rows.json").read_text())
    )
    repeat = tuple(
        V31EvidenceRowV1.from_mapping(item)
        for item in json.loads((root / "repeat-rows.json").read_text())
    )
    sentinel_data = json.loads((root / "sentinel.json").read_text())
    sentinel = V31RepeatSentinelV1.from_mapping(sentinel_data)
    manifest = V31EvidenceManifestV1.from_mapping(
        json.loads((root / "evidence-manifest.json").read_text())
    )
    _verify_batch(
        design,
        V31ExecutionBatchV1(rows, "0" * 64, "0" * 64, manifest.feature_source, {}, {}),
        None,
    )
    _verify_batch(
        design,
        V31ExecutionBatchV1(repeat, "0" * 64, "0" * 64, manifest.feature_source, {}, {}),
        11,
    )
    if v31_row_inventory_sha256(rows) != manifest.row_inventory_sha256:
        raise ValueError("v3.1 row inventory hash mismatch")
    if v31_row_inventory_sha256(repeat) != sentinel.repeat_rows_sha256:
        raise ValueError("v3.1 repeat inventory hash mismatch")
    statistics = v31_statistics(rows, design)
    if statistics != json.loads((root / "statistics.json").read_text()):
        raise ValueError("v3.1 statistics do not independently recompute")
    claim = json.loads((root / "claim-gate.json").read_text())
    if set(claim) != {
        "schema_version",
        "claim_id",
        "threshold_passed",
        "claim_allowed",
        "gate_reasons",
        "failure_statement",
    }:
        raise ValueError("v3.1 claim gate fields are invalid")
    if claim["threshold_passed"] != v31_claim_passes(statistics):
        raise ValueError("v3.1 claim threshold was not independently recomputed")
    if manifest.design_sha256 != design.sha256() or manifest.sentinel_sha256 != sentinel.sha256():
        raise ValueError("v3.1 manifest does not bind design and sentinel")
    if {row.git_sha for row in rows} != {manifest.git_sha}:
        raise ValueError("v3.1 manifest Git SHA does not match rows")
    source_execution = json.loads((root / "source/execution.json").read_text())
    repeat_execution = json.loads((root / "repeat-seed-11/execution.json").read_text())
    execution_fields = {
        "schema_version",
        "feature_source",
        "checkpoints_sha256",
        "metrics_sha256",
        "seed_checkpoint_sha256",
        "seed_metrics_sha256",
    }
    if set(source_execution) != execution_fields or set(repeat_execution) != execution_fields:
        raise ValueError("v3.1 execution contract fields are invalid")
    if (
        source_execution["feature_source"] != manifest.feature_source
        or repeat_execution["feature_source"] != manifest.feature_source
    ):
        raise ValueError("v3.1 execution feature source does not match manifest")
    if (
        source_execution["seed_checkpoint_sha256"]["11"] != sentinel.source_checkpoints_sha256
        or repeat_execution["seed_checkpoint_sha256"]["11"] != sentinel.repeat_checkpoints_sha256
        or source_execution["seed_metrics_sha256"]["11"] != sentinel.source_metrics_sha256
        or repeat_execution["seed_metrics_sha256"]["11"] != sentinel.repeat_metrics_sha256
    ):
        raise ValueError("v3.1 sentinel does not bind execution seed inventories")
    expected_reasons: list[str] = []
    if not manifest.matrix_complete:
        expected_reasons.append("incomplete_matrix")
    if not manifest.homogeneous:
        expected_reasons.append("mixed_source")
    if not manifest.sentinel_verified:
        expected_reasons.append("sentinel_failure")
    if manifest.feature_source == "deterministic_fixture":
        expected_reasons.append("fixture_source")
    if not claim["threshold_passed"]:
        expected_reasons.append("claim_threshold_not_met")
    if tuple(expected_reasons) != manifest.gate_reasons:
        raise ValueError("v3.1 manifest gate reasons were not independently recomputed")
    if manifest.claim_allowed != (not expected_reasons):
        raise ValueError("v3.1 manifest claim_allowed was not independently recomputed")
    if (
        claim["claim_allowed"] != manifest.claim_allowed
        or tuple(claim["gate_reasons"]) != manifest.gate_reasons
    ):
        raise ValueError("v3.1 claim gate does not match evidence manifest")
    return manifest
