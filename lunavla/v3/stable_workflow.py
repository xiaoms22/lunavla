from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .artifacts import sha256_file
from .stable_contracts import (
    StableEvidenceDesignV1,
    StableEvidenceRowV1,
    StableEvidenceSummaryV1,
    StableRepeatSentinelV1,
    build_stable_evidence_summary,
    stable_row_inventory_sha256,
    verify_stable_evidence_bundle,
    wilson_interval,
)


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise NotADirectoryError(f"stable execution tree does not exist: {root}")
    payload = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not payload:
        raise ValueError("stable execution tree must not be empty")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StableExecutionBatchV1:
    rows: tuple[StableEvidenceRowV1, ...]
    checkpoint_inventory_sha256: str
    metrics_inventory_sha256: str

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if not rows or any(not isinstance(row, StableEvidenceRowV1) for row in rows):
            raise ValueError("stable execution batch requires evidence rows")
        for name in ("checkpoint_inventory_sha256", "metrics_inventory_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(item not in "0123456789abcdef" for item in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        object.__setattr__(self, "rows", rows)


class StableStudyExecutor(Protocol):
    def execute(
        self,
        design: StableEvidenceDesignV1,
        *,
        only_train_seed: int | None,
        output_dir: Path,
    ) -> StableExecutionBatchV1: ...


def _statistics(rows: Sequence[StableEvidenceRowV1]) -> dict[str, Any]:
    groups: dict[tuple[str, int | None, str | None, str | None], list[bool]] = {}
    for row in rows:
        key = (row.policy, row.task_id, row.route, row.intervention)
        groups.setdefault(key, []).append(row.success)
    records: list[dict[str, Any]] = []
    for (policy, task_id, route, intervention), values in sorted(
        groups.items(), key=lambda item: json.dumps(item[0], separators=(",", ":"))
    ):
        successes = sum(values)
        low, high = wilson_interval(successes, len(values))
        records.append(
            {
                "policy": policy,
                "task_id": task_id,
                "route": route,
                "intervention": intervention,
                "successes": successes,
                "trials": len(values),
                "success_rate": successes / len(values),
                "wilson_95": [low, high],
            }
        )
    return {
        "schema_version": 1,
        "descriptive_only": True,
        "performance_claim_allowed": False,
        "groups": records,
    }


def _claim_gate() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "claim_allowed": False,
        "allowed_scope": "deterministic_lunavla_teaching_fixture",
        "reason": "v3.0 publishes complete descriptive evidence without a preregistered superiority claim",
    }


def _verify_execution_batches(
    design: StableEvidenceDesignV1,
    source: StableExecutionBatchV1,
    repeat: StableExecutionBatchV1,
) -> None:
    expected = {
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
    }
    source_keys = tuple(row.matrix_key for row in source.rows)
    if len(source_keys) != len(set(source_keys)) or set(source_keys) != expected:
        raise ValueError("stable executor did not produce the exact frozen source matrix")
    expected_repeat = {
        key for key in expected if key[1] == design.repeat_train_seed
    }
    repeat_keys = tuple(row.matrix_key for row in repeat.rows)
    if len(repeat_keys) != len(set(repeat_keys)) or set(repeat_keys) != expected_repeat:
        raise ValueError("stable executor did not produce the exact repeat-seed matrix")


def _execute_study(
    design: StableEvidenceDesignV1,
    output: Path,
    executor: StableStudyExecutor,
) -> StableEvidenceSummaryV1:
    design.validate_stable_matrix()
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"
    repeat_dir = output / "repeat-seed-11"
    source = executor.execute(design, only_train_seed=None, output_dir=source_dir)
    repeat = executor.execute(
        design,
        only_train_seed=design.repeat_train_seed,
        output_dir=repeat_dir,
    )
    _verify_execution_batches(design, source, repeat)
    source_tree_sha256 = _tree_sha256(source_dir)
    repeat_tree_sha256 = _tree_sha256(repeat_dir)
    source_seed_rows = tuple(
        row for row in source.rows if row.train_seed == design.repeat_train_seed
    )
    source_inventory = stable_row_inventory_sha256(source_seed_rows)
    repeat_inventory = stable_row_inventory_sha256(repeat.rows)
    sentinel = StableRepeatSentinelV1(
        study_id=design.study_id,
        train_seed=design.repeat_train_seed,
        source_row_inventory_sha256=source_inventory,
        repeat_row_inventory_sha256=repeat_inventory,
        source_checkpoint_sha256=source.checkpoint_inventory_sha256,
        repeat_checkpoint_sha256=repeat.checkpoint_inventory_sha256,
        source_metrics_sha256=source.metrics_inventory_sha256,
        repeat_metrics_sha256=repeat.metrics_inventory_sha256,
        verified=(
            source_inventory == repeat_inventory
            and source.checkpoint_inventory_sha256
            == repeat.checkpoint_inventory_sha256
            and source.metrics_inventory_sha256 == repeat.metrics_inventory_sha256
        ),
    )
    design_path = _write_json(output / "design.json", design.to_dict())
    rows_path = _write_json(
        output / "rows.json", [row.to_dict() for row in source.rows]
    )
    repeat_path = _write_json(
        output / "repeat-rows.json", [row.to_dict() for row in repeat.rows]
    )
    sentinel_path = _write_json(output / "sentinel.json", sentinel.to_dict())
    statistics_path = _write_json(output / "statistics.json", _statistics(source.rows))
    claim_gate_path = _write_json(output / "claim-gate.json", _claim_gate())
    git_shas = {row.git_sha for row in source.rows}
    if len(git_shas) != 1:
        raise ValueError("stable execution source rows contain mixed Git SHAs")
    summary = build_stable_evidence_summary(
        design,
        source.rows,
        sentinel,
        expected_git_sha=next(iter(git_shas)),
        statistics_sha256=sha256_file(statistics_path),
        claim_gate_sha256=sha256_file(claim_gate_path),
    )
    _write_json(output / "summary.json", summary.to_dict())
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
                    claim_gate_path,
                )
            },
            "trees": {
                "source": source_tree_sha256,
                "repeat-seed-11": repeat_tree_sha256,
            },
        },
    )
    verify_stable_study(output)
    return summary


def run_stable_study(
    design_file: str | Path,
    output_root: str | Path,
    executor: StableStudyExecutor,
    *,
    overwrite: bool = False,
) -> Path:
    design = StableEvidenceDesignV1.load(design_file)
    root = Path(output_root).resolve()
    output = root / design.study_id
    if output.exists() and not overwrite:
        raise FileExistsError(f"stable evidence output already exists: {output}")
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{design.study_id}.staging-{uuid.uuid4().hex}"
    backup = root / f".{design.study_id}.previous-{uuid.uuid4().hex}"
    try:
        _execute_study(design, staging, executor)
        if output.exists():
            output.replace(backup)
        staging.replace(output)
        if backup.exists():
            shutil.rmtree(backup)
        verify_stable_study(output)
        return output
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise


def verify_stable_study(output_dir: str | Path) -> StableEvidenceSummaryV1:
    root = Path(output_dir).resolve()
    expected = {
        "artifact-inventory.json",
        "claim-gate.json",
        "design.json",
        "repeat-rows.json",
        "rows.json",
        "sentinel.json",
        "source",
        "statistics.json",
        "summary.json",
        "repeat-seed-11",
    }
    if not root.is_dir() or {item.name for item in root.iterdir()} != expected:
        raise ValueError("stable evidence study file set is incomplete or contains extras")
    inventory = json.loads((root / "artifact-inventory.json").read_text(encoding="utf-8"))
    if (
        set(inventory) != {"schema_version", "files", "trees"}
        or inventory["schema_version"] != 1
    ):
        raise ValueError("stable artifact inventory is invalid")
    files = inventory["files"]
    if set(files) != {
        "claim-gate.json",
        "design.json",
        "repeat-rows.json",
        "rows.json",
        "sentinel.json",
        "statistics.json",
    }:
        raise ValueError("stable artifact inventory paths are invalid")
    for name, expected_hash in files.items():
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or sha256_file(candidate) != expected_hash:
            raise ValueError(f"stable evidence artifact hash mismatch: {name}")
    if set(inventory["trees"]) != {"source", "repeat-seed-11"}:
        raise ValueError("stable execution tree inventory paths are invalid")
    for name, expected_hash in inventory["trees"].items():
        if _tree_sha256(root / name) != expected_hash:
            raise ValueError(f"stable execution tree hash mismatch: {name}")
    design = StableEvidenceDesignV1.from_mapping(
        json.loads((root / "design.json").read_text(encoding="utf-8"))
    )
    rows = tuple(
        StableEvidenceRowV1.from_mapping(item)
        for item in json.loads((root / "rows.json").read_text(encoding="utf-8"))
    )
    repeat_rows = tuple(
        StableEvidenceRowV1.from_mapping(item)
        for item in json.loads((root / "repeat-rows.json").read_text(encoding="utf-8"))
    )
    sentinel = StableRepeatSentinelV1.from_mapping(
        json.loads((root / "sentinel.json").read_text(encoding="utf-8"))
    )
    summary = StableEvidenceSummaryV1.from_mapping(
        json.loads((root / "summary.json").read_text(encoding="utf-8"))
    )
    if stable_row_inventory_sha256(repeat_rows) != sentinel.repeat_row_inventory_sha256:
        raise ValueError("stable repeat row inventory does not match sentinel")
    if sha256_file(root / "statistics.json") != summary.statistics_sha256:
        raise ValueError("stable statistics hash does not match summary")
    if sha256_file(root / "claim-gate.json") != summary.claim_gate_sha256:
        raise ValueError("stable claim gate hash does not match summary")
    verify_stable_evidence_bundle(design, rows, sentinel, summary)
    return summary
