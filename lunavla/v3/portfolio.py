from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactHashRecordV1, sha256_file
from .stable_contracts import (
    StableEvidenceDesignV1,
    StableEvidenceRowV1,
    StableEvidenceSummaryV1,
    StableRepeatSentinelV1,
    validate_stable_design_set,
    verify_stable_evidence_bundle,
)
from .stable_workflow import verify_stable_study


PORTFOLIO_STATEMENT = (
    "Built a tamper-evident 1,550-row CPU teaching-fixture evidence workflow with "
    "five-seed paired analysis and fail-closed claims."
)
PORTFOLIO_LIMITATIONS = (
    "Results apply only to deterministic LunaVLA teaching fixtures.",
    "The bundle is not a PushT, LIBERO, real-robot, GPU, or production benchmark.",
    "SmolVLA remains a conformance-only adapter with pretrained weights disabled.",
)
_STUDIES = {
    "fixture_policy_ladder",
    "fixture_state_routes",
    "fixture_prompt_interventions",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_PRIVACY_RULES = {
    "private_home_path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "private_project_name": re.compile(r"(?i)\b(?:mozbrain|spirit-ai)\b"),
    "internal_url": re.compile(r"(?i)https?://[^\s\"']*(?:\.internal|\.corp)(?:[/:\s\"']|$)"),
    "github_token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "generic_secret": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})\b"),
    "contact_email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    return _write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _records(value: object, name: str) -> tuple[ArtifactHashRecordV1, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    records = tuple(ArtifactHashRecordV1.from_mapping(item) for item in value)
    if not records or len({item.path for item in records}) != len(records):
        raise ValueError(f"{name} must contain unique non-empty records")
    return tuple(sorted(records, key=lambda item: item.path))


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


@dataclass(frozen=True)
class PortfolioBundleV1:
    git_sha: str
    evidence_summaries: tuple[ArtifactHashRecordV1, ...]
    total_rows: int
    capability_statements: tuple[str, ...]
    limitations: tuple[str, ...]
    privacy_scan_sha256: str
    files: tuple[ArtifactHashRecordV1, ...]
    source_homogeneous: bool
    release_eligible: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PortfolioBundleV1 schema_version must be integer 1")
        if not _GIT_SHA.fullmatch(self.git_sha):
            raise ValueError("portfolio git_sha must be a full lowercase Git SHA")
        if isinstance(self.total_rows, bool) or self.total_rows != 1_550:
            raise ValueError("portfolio bundle must bind exactly 1,550 evidence rows")
        summaries = tuple(sorted(self.evidence_summaries, key=lambda item: item.path))
        files = tuple(sorted(self.files, key=lambda item: item.path))
        if {item.path for item in summaries} != {
            f"{study}/summary.json" for study in _STUDIES
        }:
            raise ValueError("portfolio requires all three evidence summaries")
        if {item.path for item in files} != {
            "config-differences.json", "index.html", "portfolio.json",
            "portfolio.md", "privacy-scan.json",
        }:
            raise ValueError("portfolio output file set is incomplete")
        if not _SHA256.fullmatch(self.privacy_scan_sha256):
            raise ValueError("privacy_scan_sha256 must be a lowercase SHA-256")
        if self.source_homogeneous is not True or self.release_eligible is not True:
            raise ValueError("portfolio export requires homogeneous release-eligible evidence")
        if any(not isinstance(item, ArtifactHashRecordV1) for item in (*summaries, *files)):
            raise TypeError("portfolio artifact records must use ArtifactHashRecordV1")
        statements = _string_tuple(self.capability_statements, "capability_statements")
        limitations = _string_tuple(self.limitations, "limitations")
        if statements != (PORTFOLIO_STATEMENT,):
            raise ValueError("portfolio capability wording must be fixed")
        if limitations != PORTFOLIO_LIMITATIONS:
            raise ValueError("portfolio limitations must be complete and fixed")
        object.__setattr__(self, "evidence_summaries", summaries)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "capability_statements", statements)
        object.__setattr__(self, "limitations", limitations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "git_sha": self.git_sha,
            "evidence_summaries": [item.to_dict() for item in self.evidence_summaries],
            "total_rows": self.total_rows,
            "capability_statements": list(self.capability_statements),
            "limitations": list(self.limitations),
            "privacy_scan_sha256": self.privacy_scan_sha256,
            "files": [item.to_dict() for item in self.files],
            "source_homogeneous": self.source_homogeneous,
            "release_eligible": self.release_eligible,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PortfolioBundleV1":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("PortfolioBundleV1 requires exact fields")
        payload = dict(value)
        payload["evidence_summaries"] = _records(
            payload["evidence_summaries"], "evidence_summaries"
        )
        payload["files"] = _records(payload["files"], "files")
        for field in ("capability_statements", "limitations"):
            payload[field] = _string_tuple(payload[field], field)
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


def _load_study(root: Path, study: str) -> tuple[
    StableEvidenceDesignV1,
    tuple[StableEvidenceRowV1, ...],
    StableRepeatSentinelV1,
    StableEvidenceSummaryV1,
]:
    directory = root / study
    expected = {"design.json", "rows.json", "sentinel.json", "summary.json"}
    if not directory.is_dir():
        raise ValueError(f"portfolio evidence study {study} has an invalid file set")
    names = {item.name for item in directory.iterdir()}
    if "artifact-inventory.json" in names:
        verify_stable_study(directory)
    elif names != expected:
        raise ValueError(f"portfolio evidence study {study} has an invalid file set")
    design = StableEvidenceDesignV1.from_mapping(
        json.loads((directory / "design.json").read_text(encoding="utf-8"))
    )
    raw_rows = json.loads((directory / "rows.json").read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise TypeError("portfolio evidence rows must be a list")
    rows = tuple(StableEvidenceRowV1.from_mapping(item) for item in raw_rows)
    sentinel = StableRepeatSentinelV1.from_mapping(
        json.loads((directory / "sentinel.json").read_text(encoding="utf-8"))
    )
    summary = StableEvidenceSummaryV1.from_mapping(
        json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    )
    verify_stable_evidence_bundle(design, rows, sentinel, summary)
    if design.study_id != study or not summary.release_eligible:
        raise ValueError("portfolio evidence is claim-closed or has the wrong study identity")
    return design, rows, sentinel, summary


def _privacy_scan(files: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    checked: list[str] = []
    for label, path in files:
        checked.append(label)
        text = path.read_text(encoding="utf-8")
        for rule, pattern in _PRIVACY_RULES.items():
            if pattern.search(text):
                findings.append({"file": label, "rule": rule})
    return {
        "schema_version": 1,
        "passed": not findings,
        "findings": findings,
        "rules": sorted(_PRIVACY_RULES),
        "checked_files": sorted(checked),
    }


def _source_files(source: Path) -> tuple[tuple[str, Path], ...]:
    allowed = (
        "artifact-inventory.json",
        "claim-gate.json",
        "design.json",
        "repeat-rows.json",
        "rows.json",
        "sentinel.json",
        "statistics.json",
        "summary.json",
    )
    files: list[tuple[str, Path]] = []
    for study in sorted(_STUDIES):
        directory = source / study
        names = {item.name for item in directory.iterdir()} if directory.is_dir() else set()
        selected = allowed if "artifact-inventory.json" in names else (
            "design.json", "rows.json", "sentinel.json", "summary.json"
        )
        for name in selected:
            path = directory / name
            if not path.is_file():
                raise ValueError(f"portfolio evidence study {study} is missing {name}")
            files.append((f"{study}/{name}", path))
    return tuple(files)


def build_portfolio(
    evidence_root: str | Path, output_root: str | Path, *, overwrite: bool = False
) -> Path:
    source = Path(evidence_root).resolve()
    if not source.is_dir() or {item.name for item in source.iterdir()} != _STUDIES:
        raise ValueError("portfolio evidence root must contain exactly the three frozen studies")
    source_files = _source_files(source)
    source_scan = _privacy_scan(source_files)
    if not source_scan["passed"]:
        raise ValueError("portfolio source evidence privacy scan failed")
    loaded = {study: _load_study(source, study) for study in sorted(_STUDIES)}
    designs = tuple(value[0] for value in loaded.values())
    rows_by_study = validate_stable_design_set(designs)
    summaries = tuple(value[3] for value in loaded.values())
    git_shas = {summary.git_sha for summary in summaries}
    dependency_locks: dict[str, set[str]] = {}
    upstream_identities: dict[str, set[str]] = {}
    for _, rows, _, _ in loaded.values():
        for row in rows:
            dependency_locks.setdefault(row.policy, set()).add(
                row.dependency_lock_sha256
            )
            upstream_identities.setdefault(row.policy, set()).add(
                row.upstream_identity_sha256
            )
    if (
        len(git_shas) != 1
        or any(len(values) != 1 for values in dependency_locks.values())
        or any(len(values) != 1 for values in upstream_identities.values())
    ):
        raise ValueError("portfolio evidence contains mixed source provenance")
    output = Path(output_root).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("portfolio output must not overlap the evidence source")
    if output.exists() and not output.is_dir():
        raise ValueError("portfolio output must be a directory")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("portfolio output already exists; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    backup = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
    try:
        staging.mkdir()
        differences = {
            study: {
                "policies": list(value[0].policies),
                "tasks": list(value[0].task_ids),
                "routes": list(value[0].routes),
                "interventions": list(value[0].interventions),
                "rows": rows_by_study[study],
            }
            for study, value in loaded.items()
        }
        portfolio = {
            "schema_version": 1,
            "title": "LunaVLA v3 CPU teaching evidence",
            "git_sha": next(iter(git_shas)),
            "total_rows": sum(rows_by_study.values()),
            "capability_statements": [PORTFOLIO_STATEMENT],
            "resume_bullet_templates": [PORTFOLIO_STATEMENT],
            "limitations": list(PORTFOLIO_LIMITATIONS),
            "studies": rows_by_study,
        }
        json_path = _write_json(staging / "portfolio.json", portfolio)
        differences_path = _write_json(staging / "config-differences.json", differences)
        markdown = "# LunaVLA v3 CPU teaching evidence\n\n" + PORTFOLIO_STATEMENT + "\n\n"
        markdown += "## Verified studies\n\n" + "\n".join(
            f"- `{study}`: {rows_by_study[study]} rows" for study in sorted(rows_by_study)
        )
        markdown += "\n\n## Limitations\n\n" + "\n".join(
            f"- {item}" for item in PORTFOLIO_LIMITATIONS
        ) + "\n"
        markdown_path = _write(staging / "portfolio.md", markdown)
        html_path = _write(
            staging / "index.html",
            "<!doctype html><meta charset=\"utf-8\"><title>LunaVLA v3 evidence</title>"
            f"<h1>LunaVLA v3 CPU teaching evidence</h1><p>{html.escape(PORTFOLIO_STATEMENT)}</p>"
            + "<ul>"
            + "".join(
                f"<li><code>{html.escape(study)}</code>: {rows_by_study[study]} rows</li>"
                for study in sorted(rows_by_study)
            )
            + "</ul><h2>Limitations</h2><ul>"
            + "".join(f"<li>{html.escape(item)}</li>" for item in PORTFOLIO_LIMITATIONS)
            + "</ul>\n",
        )
        generated_files = (
            ("config-differences.json", differences_path),
            ("index.html", html_path),
            ("portfolio.json", json_path),
            ("portfolio.md", markdown_path),
        )
        scan = _privacy_scan((*source_files, *generated_files))
        if not scan["passed"]:
            raise ValueError("portfolio privacy scan failed")
        privacy_path = _write_json(staging / "privacy-scan.json", scan)
        files = tuple(
            ArtifactHashRecordV1(path.name, sha256_file(path))
            for path in (differences_path, html_path, json_path, markdown_path, privacy_path)
        )
        evidence = tuple(
            ArtifactHashRecordV1(
                f"{study}/summary.json", sha256_file(source / study / "summary.json")
            )
            for study in sorted(_STUDIES)
        )
        manifest = PortfolioBundleV1(
            git_sha=next(iter(git_shas)),
            evidence_summaries=evidence,
            total_rows=1_550,
            capability_statements=(PORTFOLIO_STATEMENT,),
            limitations=PORTFOLIO_LIMITATIONS,
            privacy_scan_sha256=sha256_file(privacy_path),
            files=files,
            source_homogeneous=True,
            release_eligible=True,
        )
        _write_json(staging / "portfolio_manifest.json", manifest.to_dict())
        verify_portfolio(staging)
        if output.exists():
            output.replace(backup)
        staging.replace(output)
        if backup.exists():
            shutil.rmtree(backup)
        return output
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise


def verify_portfolio(output_root: str | Path) -> PortfolioBundleV1:
    root = Path(output_root)
    expected = {
        "config-differences.json", "index.html", "portfolio.json", "portfolio.md",
        "portfolio_manifest.json", "privacy-scan.json",
    }
    if not root.is_dir() or {item.name for item in root.iterdir()} != expected:
        raise ValueError("portfolio output file set is incomplete or contains extras")
    manifest = PortfolioBundleV1.from_mapping(
        json.loads((root / "portfolio_manifest.json").read_text(encoding="utf-8"))
    )
    for record in manifest.files:
        if sha256_file(root / record.path) != record.sha256:
            raise ValueError(f"portfolio file hash mismatch: {record.path}")
    if sha256_file(root / "privacy-scan.json") != manifest.privacy_scan_sha256:
        raise ValueError("portfolio privacy scan hash mismatch")
    scan = json.loads((root / "privacy-scan.json").read_text(encoding="utf-8"))
    if scan.get("passed") is not True or scan.get("findings") != []:
        raise ValueError("portfolio privacy scan is not clean")
    generated_scan = _privacy_scan(
        (
            ("config-differences.json", root / "config-differences.json"),
            ("index.html", root / "index.html"),
            ("portfolio.json", root / "portfolio.json"),
            ("portfolio.md", root / "portfolio.md"),
        )
    )
    if not generated_scan["passed"]:
        raise ValueError("portfolio generated files fail privacy verification")
    portfolio = json.loads((root / "portfolio.json").read_text(encoding="utf-8"))
    if portfolio.get("capability_statements") != [PORTFOLIO_STATEMENT]:
        raise ValueError("portfolio capability wording mismatch")
    if portfolio.get("resume_bullet_templates") != [PORTFOLIO_STATEMENT]:
        raise ValueError("portfolio resume wording mismatch")
    if portfolio.get("limitations") != list(PORTFOLIO_LIMITATIONS):
        raise ValueError("portfolio limitations mismatch")
    if portfolio.get("git_sha") != manifest.git_sha or portfolio.get("total_rows") != 1_550:
        raise ValueError("portfolio identity or row count mismatch")
    if portfolio.get("studies") != {
        "fixture_policy_ladder": 200,
        "fixture_prompt_interventions": 750,
        "fixture_state_routes": 600,
    }:
        raise ValueError("portfolio study matrix mismatch")
    return manifest
