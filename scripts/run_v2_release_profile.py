from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from lunavla.evidence import EvidenceManifest
from lunavla.evidence_design import EvidenceDesign
from lunavla.evidence_runner import (
    EvidenceVerification,
    derive_plan,
    is_full_design,
    verify_evidence,
)
from lunavla.manifest import MANIFEST_SCHEMA_VERSION, RunManifest
from lunavla.lerobot_integration import IntegrationManifest
from lunavla.published_evidence import (
    PUBLISHED_LANGUAGE_EVIDENCE_SHA256,
    PUBLISHED_LANGUAGE_GIT_SHA,
    PUBLISHED_LANGUAGE_WORKFLOW_URL,
    PUBLISHED_VISUAL_EVIDENCE_SHA256,
    PUBLISHED_VISUAL_GIT_SHA,
    PUBLISHED_VISUAL_WORKFLOW_URL,
    verify_language_snapshot,
    verify_visual_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "release-assets"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ALPHA_CONFIGS = (
    Path("configs/v2/numpy_baseline.yaml"),
    Path("configs/v2/transformer_chunk_cpu.yaml"),
    Path("configs/v2/transformer_visual_cpu.yaml"),
    Path("configs/v2/transformer_visual_state_only_cpu.yaml"),
)
EVIDENCE_PROFILE_DESIGNS = {
    "language": Path("configs/v2/evidence/language_alpha2.yaml"),
    "vision": Path("configs/v2/evidence/visual_beta1.yaml"),
}
EVIDENCE_PROFILE_SUITES = {"language": "language", "vision": "visual"}
EVIDENCE_ARCHIVE_NAMES = {
    "language": "lunavla-v2-language-evidence.tar.gz",
    "vision": "lunavla-v2-vision-evidence.tar.gz",
}
RC_ARCHIVE_NAME = "lunavla-v2-rc-evidence.tar.gz"
RC_INTEGRITY_NAME = "rc-release-integrity.json"
RC_PACKAGE_VERSION = "2.0.0rc1"
STABLE_ARCHIVE_NAME = "lunavla-v2-stable-evidence.tar.gz"
STABLE_PACKAGE_VERSION = "2.0.0"
STABLE_TAG = "v2.0.0"
STABLE_DEFAULT_BRANCH = "main"
STABLE_SIGNER_WORKFLOW = "xiaoms22/lunavla/.github/workflows/v2-release-dispatch.yml"
STABLE_SOURCE_REF = "refs/heads/main"
STABLE_OUTPUTS = {
    "language": (
        Path("outputs/stable-release/language"),
        Path("results/v2/stable-release/language"),
    ),
    "vision": (
        Path("outputs/stable-release/vision"),
        Path("results/v2/stable-release/vision"),
    ),
}
PUBLISHED_SNAPSHOT_ROOTS = (
    Path("results/v2/language-alpha2"),
    Path("results/v2/visual-beta1"),
)
RC_CONTRACT_SOURCES = (
    Path("docs/v2/public_api_contract.json"),
    Path("docs/v2/contracts/config-design-schema.json"),
    Path("docs/v2/artifact_contracts.json"),
    Path("docs/v2/contract_freeze.md"),
    Path("docs/v2/compatibility.md"),
    Path("docs/v2/evidence_contract.md"),
    Path("docs/v2/MODEL_CARD.md"),
    Path("docs/v2/DATA_CARD.md"),
    Path("SECURITY.md"),
)


@dataclass(frozen=True)
class ControlledEvidence:
    profile: str
    design_path: Path
    design: EvidenceDesign
    output_root: Path
    snapshot_root: Path
    manifest: EvidenceManifest
    verification: EvidenceVerification
    runs: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class StableIntegration:
    manifest_path: Path
    manifest_sha256: str
    bundle_path: Path
    bundle_sha256: str
    verification_path: Path
    verification_sha256: str
    verified_attestation_count: int


@dataclass(frozen=True)
class StableEvidence:
    canonical_design_path: Path
    controlled: ControlledEvidence
    metadata: dict[str, Path]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean LunaVLA v2 release evidence profile."
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=("alpha", "language", "vision", "rc", "stable"),
    )
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument(
        "--integration-manifest",
        type=Path,
        help="stable-only real LeRobot integration manifest from this workflow",
    )
    parser.add_argument(
        "--integration-attestation-bundle",
        type=Path,
        help="stable-only GitHub provenance bundle for the integration manifest",
    )
    return parser.parse_args(argv)


def run(command: Sequence[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        list(command),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_asset_record(path: Path) -> dict[str, str]:
    """Return the stable release-relative identity of one existing asset."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"release asset must be a real file: {path}")
    try:
        relative = path.relative_to(RELEASE_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"release asset is outside release-assets/: {path}") from exc
    return {"path": relative, "sha256": sha256_file(path)}


def _real_tree_files(root: Path, *, label: str) -> tuple[Path, ...]:
    """Return every regular file below a real directory and reject links."""

    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"{label} must be a real directory: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"{label} cannot contain symlinks: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise RuntimeError(f"{label} contains a non-regular entry: {path}")
    return tuple(files)


def canonical_origin(remote: str) -> str:
    """Normalize Git's equivalent HTTPS/SSH spelling without broadening ownership."""

    value = remote.strip()
    if value.endswith(".git"):
        value = value[:-4]
    return value.rstrip("/")


def verify_source(expected_sha: str) -> None:
    if not SHA256_PATTERN.fullmatch(expected_sha):
        raise ValueError("--expected-sha must be a 40-character lowercase Git SHA")
    actual = run(("git", "rev-parse", "HEAD"), capture=True)
    if actual != expected_sha:
        raise RuntimeError(f"checked-out Git SHA {actual} does not match {expected_sha}")
    status = run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        capture=True,
    )
    if status:
        raise RuntimeError("release evidence requires a clean Git checkout")
    remote = canonical_origin(run(("git", "remote", "get-url", "origin"), capture=True))
    if remote not in {
        "https://github.com/xiaoms22/lunavla",
        "git@github.com:xiaoms22/lunavla",
    }:
        raise RuntimeError(f"refusing release evidence from unexpected origin: {remote}")


def verify_stable_main_tip(expected_sha: str) -> None:
    """Require the stable candidate to be the fetched tip of the default branch."""

    remote_main = run(
        ("git", "rev-parse", "--verify", "refs/remotes/origin/main^{commit}"),
        capture=True,
    )
    if remote_main != expected_sha:
        raise RuntimeError(
            "stable evidence requires expected_sha to equal the fetched origin/main tip"
        )
    if validated_project_version() != STABLE_PACKAGE_VERSION:
        raise RuntimeError(f"stable evidence requires package version {STABLE_PACKAGE_VERSION}")


def _tree_sha256(root: Path) -> str:
    """Hash a real directory's paths and contents, rejecting symlink ambiguity."""

    digest = hashlib.sha256()
    for path in _real_tree_files(root, label="snapshot root"):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def published_snapshot_hashes() -> dict[str, str]:
    """Capture tracked publication snapshots so stable generation cannot alter them."""

    return {path.as_posix(): _tree_sha256(ROOT / path) for path in PUBLISHED_SNAPSHOT_ROOTS}


def _require_ignored_destination(path: Path) -> None:
    result = subprocess.run(
        ("git", "check-ignore", "--quiet", "--", path.relative_to(ROOT).as_posix()),
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"stable generated destination must be ignored by Git: {path}")


def _stable_execution_design(
    profile: str,
) -> tuple[Path, Path, EvidenceDesign, Path, Path]:
    """Copy a canonical design while isolating all post-merge generated files."""

    canonical_path, canonical = evidence_profile_design(profile)
    output_relative, snapshot_relative = STABLE_OUTPUTS[profile]
    output_root = ROOT / output_relative
    snapshot_root = ROOT / snapshot_relative
    for destination in (output_root, snapshot_root):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"stable generated destination already exists: {destination.relative_to(ROOT)}"
            )
        _require_ignored_destination(destination)
    payload = canonical.to_dict()
    payload["output"] = {
        "run_root": output_relative.as_posix(),
        "snapshot_root": snapshot_relative.as_posix(),
    }
    execution = EvidenceDesign.from_mapping(payload)
    if not is_full_design(execution):
        raise RuntimeError(f"stable {profile} execution design is not canonical")
    generated_path = RELEASE_ROOT / "stable-designs" / f"{profile}.yaml"
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(
        json.dumps(execution.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return canonical_path, generated_path, execution, output_root, snapshot_root


def prepare_release_root() -> None:
    if RELEASE_ROOT.exists() or (ROOT / "outputs" / "v2").exists():
        raise FileExistsError("release-assets/ and outputs/v2/ must not exist before a release run")
    RELEASE_ROOT.mkdir(parents=True)


def evidence_profile_design(profile: str) -> tuple[Path, EvidenceDesign]:
    """Load the one canonical full design assigned to a release profile."""

    try:
        relative = EVIDENCE_PROFILE_DESIGNS[profile]
    except KeyError as exc:
        raise ValueError(f"profile {profile!r} has no controlled EvidenceDesign") from exc
    path = ROOT / relative
    design = EvidenceDesign.load(path)
    if design.suite != EVIDENCE_PROFILE_SUITES[profile]:
        raise RuntimeError(
            f"profile {profile!r} requires suite {EVIDENCE_PROFILE_SUITES[profile]!r}, "
            f"not {design.suite!r}"
        )
    if not is_full_design(design):
        raise RuntimeError(f"profile {profile!r} requires the exact canonical full design")
    return path, design


def prepare_evidence_release(profile: str) -> tuple[Path, EvidenceDesign, Path, Path]:
    design_path, design = evidence_profile_design(profile)
    output_root = ROOT / design.output.run_root
    snapshot_root = ROOT / design.output.snapshot_root
    existing = [
        path
        for path in (RELEASE_ROOT, output_root, snapshot_root)
        if path.exists() or path.is_symlink()
    ]
    if existing:
        rendered = ", ".join(path.relative_to(ROOT).as_posix() for path in existing)
        raise FileExistsError(
            f"controlled release destinations must not exist before a release run: {rendered}"
        )
    RELEASE_ROOT.mkdir(parents=True)
    return design_path, design, output_root, snapshot_root


def prepare_stable_release() -> dict[str, str]:
    """Create only the ignored stable root and bind pre-existing tracked snapshots."""

    if RELEASE_ROOT.exists() or RELEASE_ROOT.is_symlink():
        raise FileExistsError("release-assets/ must not exist before a stable release run")
    for output_root, snapshot_root in STABLE_OUTPUTS.values():
        for relative in (output_root, snapshot_root):
            path = ROOT / relative
            if path.exists() or path.is_symlink():
                raise FileExistsError(
                    f"stable generated destination must not exist: {relative.as_posix()}"
                )
    before = published_snapshot_hashes()
    RELEASE_ROOT.mkdir(parents=True)
    return before


def alpha_quality_gate() -> None:
    commands: tuple[tuple[str, ...], ...] = (
        ("uv", "lock", "--check"),
        (sys.executable, "scripts/lock_v2_cpu.py", "--check"),
        (sys.executable, "scripts/validate_configs.py"),
        (sys.executable, "scripts/check_repo_quality.py"),
        (sys.executable, "scripts/render_readme_results.py", "--check"),
        ("ruff", "check", "."),
        ("mypy", "dataset", "model", "trainer", "lunavla", "eval_vla.py"),
        (sys.executable, "-m", "pytest", "-q"),
    )
    for command in commands:
        run(command)


def alpha_runs(expected_sha: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config_path in ALPHA_CONFIGS:
        run(
            (
                sys.executable,
                "-m",
                "lunavla.cli",
                "train",
                config_path.as_posix(),
                "--require-device",
                "cpu",
                "--require-clean",
            )
        )
        config = json.loads(
            run(
                (
                    sys.executable,
                    "-c",
                    ("import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))"),
                    config_path.as_posix(),
                ),
                capture=True,
            )
        )
        output_dir = ROOT / str(config["artifacts"]["output_dir"])
        run((sys.executable, "-m", "lunavla.cli", "verify-run", str(output_dir)))
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["git_sha"] != expected_sha:
            raise RuntimeError(f"manifest source mismatch: {manifest_path}")
        if manifest["git_dirty"] or manifest["source_diff_sha256"] is not None:
            raise RuntimeError(f"dirty manifest is not release evidence: {manifest_path}")
        rows.append(
            {
                "config": config_path.as_posix(),
                "output_dir": output_dir.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "policy_id": manifest["policy_id"],
                "task_id": manifest["task_id"],
                "claim_allowed": False,
            }
        )
    return rows


def _release_run_row(
    *,
    output_root: Path,
    source_run_id: str,
    expected_sha: str,
) -> dict[str, object]:
    run_dir = output_root / "runs" / source_run_id
    manifest_path = run_dir / "manifest.json"
    manifest = RunManifest.verify_run_dir(run_dir)
    failures = {
        "schema_version": manifest.schema_version == MANIFEST_SCHEMA_VERSION,
        "git_sha": manifest.git_sha == expected_sha,
        "source_clean": not manifest.git_dirty and manifest.source_diff_sha256 is None,
        "training_device": manifest.config.get("training", {}).get("device") == "cpu",
        "policy_device": manifest.config.get("policy", {}).get("device") == "cpu",
        "recorded_device": manifest.runtime_determinism.get("device") == "cpu",
    }
    failed = [name for name, passed in failures.items() if not passed]
    if failed:
        raise RuntimeError(
            f"controlled evidence source {source_run_id!r} failed release checks: "
            + ", ".join(failed)
        )
    return {
        "run_id": source_run_id,
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "policy_id": manifest.policy_id,
        "task_id": manifest.task_id,
        "device": "cpu",
    }


def validate_controlled_evidence(
    *,
    profile: str,
    expected_sha: str,
    design_path: Path,
    design: EvidenceDesign,
    output_root: Path,
    snapshot_root: Path,
) -> ControlledEvidence:
    """Recompute the aggregate and enforce release-specific provenance gates."""

    verification = verify_evidence(output_root)
    manifest_path = output_root / "evidence_manifest.json"
    manifest = EvidenceManifest.load(manifest_path)
    plan = derive_plan(design)
    checks = {
        "canonical_full_design": is_full_design(design),
        "design_identity": verification.design_id == design.design_id
        and verification.design_sha256 == design.sha256(),
        "full_matrix": not verification.reduced_design
        and not manifest.reduced_design
        and manifest.matrix_complete,
        "source_count": verification.source_count == plan.expected_training_runs,
        "arm_episode_count": verification.arm_episode_count == plan.expected_arm_episodes,
        "exact_git_sha": verification.git_sha == expected_sha,
        "integrity": all(passed for _, passed in manifest.integrity_checks),
        "snapshot": snapshot_root.is_dir() and not snapshot_root.is_symlink(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"controlled {profile} evidence failed release checks: " + ", ".join(failed)
        )
    expected_claim = (
        "instruction_following" if profile == "language" else "visual_control_contribution"
    )
    if tuple(claim.claim_id for claim in manifest.claims) != (expected_claim,):
        raise RuntimeError(f"controlled {profile} evidence has an unexpected claim contract")
    runs = tuple(
        _release_run_row(
            output_root=output_root,
            source_run_id=source.run_id,
            expected_sha=expected_sha,
        )
        for source in manifest.sources
    )
    source_hashes = {source.run_id: source.manifest_sha256 for source in manifest.sources}
    if any(row["manifest_sha256"] != source_hashes[str(row["run_id"])] for row in runs):
        raise RuntimeError("release run manifest hashes differ from EvidenceManifest sources")
    return ControlledEvidence(
        profile=profile,
        design_path=design_path,
        design=design,
        output_root=output_root,
        snapshot_root=snapshot_root,
        manifest=manifest,
        verification=verification,
        runs=runs,
    )


def execute_controlled_evidence(
    *,
    profile: str,
    expected_sha: str,
    design_path: Path,
    design: EvidenceDesign,
    output_root: Path,
    snapshot_root: Path,
) -> ControlledEvidence:
    """Run, verify, snapshot, and re-verify the full declared matrix via the public CLI."""

    run(
        (
            sys.executable,
            "-m",
            "lunavla.cli",
            "evidence-run",
            design_path.relative_to(ROOT).as_posix(),
        )
    )
    run(
        (
            sys.executable,
            "-m",
            "lunavla.cli",
            "evidence-verify",
            output_root.relative_to(ROOT).as_posix(),
        )
    )
    run(
        (
            sys.executable,
            "-m",
            "lunavla.cli",
            "evidence-snapshot",
            output_root.relative_to(ROOT).as_posix(),
            "--out",
            snapshot_root.relative_to(ROOT).as_posix(),
        )
    )
    run(
        (
            sys.executable,
            "-m",
            "lunavla.cli",
            "evidence-verify",
            output_root.relative_to(ROOT).as_posix(),
        )
    )
    return validate_controlled_evidence(
        profile=profile,
        expected_sha=expected_sha,
        design_path=design_path,
        design=design,
        output_root=output_root,
        snapshot_root=snapshot_root,
    )


def build_distributions() -> list[Path]:
    destination = RELEASE_ROOT / "dist"
    run((sys.executable, "-m", "build", "--outdir", str(destination)))
    distributions = sorted(path for path in destination.iterdir() if path.is_file())
    if not distributions:
        raise RuntimeError("package build produced no distributions")
    run(("twine", "check", *(str(path) for path in distributions)))
    version = validated_project_version()
    expected_names = {
        f"lunavla-{version}-py3-none-any.whl",
        f"lunavla-{version}.tar.gz",
    }
    actual_names = {path.name for path in distributions}
    if actual_names != expected_names:
        raise RuntimeError(
            f"release distributions do not match project version {version}: {sorted(actual_names)}"
        )
    return distributions


def installed_requirements() -> str:
    """Return a deterministic freeze without assuming pip exists in the uv venv."""

    versions: dict[str, tuple[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        value = (name, distribution.version)
        previous = versions.get(canonical)
        if previous is not None and previous[1] != value[1]:
            raise RuntimeError(
                f"conflicting installed versions for {canonical}: {previous[1]} and {value[1]}"
            )
        versions[canonical] = value
    if not versions:
        raise RuntimeError("the release environment contains no distributions")
    return "".join(f"{versions[key][0]}=={versions[key][1]}\n" for key in sorted(versions))


def validated_project_version() -> str:
    """Require source, package metadata, and public API versions to agree."""

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = project.get("project", {}).get("version")
    installed_version = importlib.metadata.version("lunavla")
    public_version = getattr(importlib.import_module("lunavla"), "__version__", None)
    versions = {
        "pyproject.toml": project_version,
        "installed metadata": installed_version,
        "lunavla.__version__": public_version,
    }
    if not all(isinstance(value, str) and value for value in versions.values()):
        raise RuntimeError(f"invalid LunaVLA version contract: {versions}")
    if len(set(versions.values())) != 1:
        raise RuntimeError(f"inconsistent LunaVLA version contract: {versions}")
    return installed_version


def sbom_command(requirements: Path, output: Path) -> tuple[str, ...]:
    return (
        "cyclonedx-py",
        "requirements",
        str(requirements),
        "--output-reproducible",
        "--output-file",
        str(output),
    )


def write_environment_and_sbom() -> None:
    requirements = RELEASE_ROOT / "environment-requirements.txt"
    requirements.write_text(installed_requirements(), encoding="utf-8")
    run(sbom_command(requirements, RELEASE_ROOT / "sbom.json"))


def _verified_subject_count(payload: object, expected_sha256: str) -> int:
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("GitHub returned no verified integration attestations")
    matched = 0
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeError(f"attestation verification result {index} is not an object")
        verification = item.get("verificationResult")
        if not isinstance(verification, dict):
            raise RuntimeError("attestation result is missing verificationResult")
        statement = verification.get("statement")
        if not isinstance(statement, dict):
            raise RuntimeError("attestation result is missing its verified statement")
        subjects = statement.get("subject")
        if not isinstance(subjects, list):
            raise RuntimeError("attestation verified statement has no subject list")
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            digest = subject.get("digest")
            if isinstance(digest, dict) and digest.get("sha256") == expected_sha256:
                matched += 1
                break
    if matched == 0:
        raise RuntimeError("verified attestation does not bind the integration manifest SHA-256")
    return matched


def verify_stable_integration(
    *,
    expected_sha: str,
    manifest_source: Path,
    bundle_source: Path,
) -> StableIntegration:
    """Verify the strict manifest and GitHub provenance identity, then copy both."""

    raw_sources = (("manifest", Path(manifest_source)), ("bundle", Path(bundle_source)))
    for label, source in raw_sources:
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"stable integration {label} must be a real file: {source}")
    manifest_source = Path(manifest_source).resolve()
    bundle_source = Path(bundle_source).resolve()
    manifest = IntegrationManifest.load(manifest_source)
    if manifest.git_sha != expected_sha:
        raise RuntimeError("integration manifest Git SHA differs from the stable candidate")
    manifest_sha256 = sha256_file(manifest_source)
    verification_text = run(
        (
            "gh",
            "attestation",
            "verify",
            str(manifest_source),
            "--bundle",
            str(bundle_source),
            "--repo",
            "xiaoms22/lunavla",
            "--signer-workflow",
            STABLE_SIGNER_WORKFLOW,
            "--source-digest",
            expected_sha,
            "--source-ref",
            STABLE_SOURCE_REF,
            "--deny-self-hosted-runners",
            "--format",
            "json",
        ),
        capture=True,
    )
    try:
        verification_payload = json.loads(verification_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GitHub attestation verification did not return JSON") from exc
    verified_count = _verified_subject_count(verification_payload, manifest_sha256)

    destination = RELEASE_ROOT / "integration"
    destination.mkdir()
    manifest_path = destination / "integration_manifest.json"
    bundle_path = destination / "integration-attestation-bundle.jsonl"
    verification_path = destination / "integration-attestation-verification.json"
    shutil.copyfile(manifest_source, manifest_path)
    shutil.copyfile(bundle_source, bundle_path)
    verification_path.write_text(
        json.dumps(
            verification_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return StableIntegration(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        bundle_path=bundle_path,
        bundle_sha256=sha256_file(bundle_path),
        verification_path=verification_path,
        verification_sha256=sha256_file(verification_path),
        verified_attestation_count=verified_count,
    )


def write_candidate(
    *,
    profile: str,
    expected_sha: str,
    runs: Iterable[dict[str, object]],
    distributions: Sequence[Path],
) -> Path:
    path = RELEASE_ROOT / "release-candidate.json"
    payload = {
        "schema_version": 1,
        "profile": profile,
        "expected_sha": expected_sha,
        "package_version": validated_project_version(),
        "authoritative_device": "cpu",
        "modality_effect_claims": False,
        "runs": list(runs),
        "distributions": [
            {
                "path": item.relative_to(RELEASE_ROOT).as_posix(),
                "sha256": sha256_file(item),
            }
            for item in distributions
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_evidence_metadata(evidence: ControlledEvidence) -> dict[str, Path]:
    """Materialize reviewable design, aggregate, verification, and claim records."""

    prefix = evidence.profile
    design_path = RELEASE_ROOT / f"{prefix}-evidence-design.yaml"
    manifest_path = RELEASE_ROOT / f"{prefix}-evidence-manifest.json"
    verification_path = RELEASE_ROOT / f"{prefix}-evidence-verification.json"
    claims_path = RELEASE_ROOT / f"{prefix}-claim-summary.json"
    shutil.copyfile(evidence.design_path, design_path)
    shutil.copyfile(evidence.output_root / "evidence_manifest.json", manifest_path)
    verification_path.write_text(
        json.dumps(
            evidence.verification.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    claim_payload = {
        "schema_version": 1,
        "profile": evidence.profile,
        "design_id": evidence.design.design_id,
        "design_sha256": evidence.design.sha256(),
        "claim_allowed": any(claim.allowed for claim in evidence.manifest.claims),
        "claims": [claim.to_dict() for claim in evidence.manifest.claims],
    }
    claims_path.write_text(
        json.dumps(claim_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "design": design_path,
        "manifest": manifest_path,
        "verification": verification_path,
        "claims": claims_path,
    }


def write_evidence_file_checksums(
    evidence: ControlledEvidence,
    metadata: dict[str, Path],
    distributions: Sequence[Path],
) -> Path:
    """Hash every complete-evidence and review-snapshot file before archiving."""

    release_inputs = (
        RELEASE_ROOT / "environment-requirements.txt",
        RELEASE_ROOT / "sbom.json",
        *metadata.values(),
        *distributions,
    )
    candidates = sorted(
        (
            *(
                path
                for root in (evidence.output_root, evidence.snapshot_root)
                for path in root.rglob("*")
                if path.is_file() and not path.is_symlink()
            ),
            *release_inputs,
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    if not candidates:
        raise RuntimeError("controlled evidence produced no files to checksum")
    target = RELEASE_ROOT / f"{evidence.profile}-evidence-files.SHA256SUMS"
    target.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in candidates
        ),
        encoding="utf-8",
    )
    return target


def write_evidence_candidate(
    *,
    expected_sha: str,
    evidence: ControlledEvidence,
    metadata: dict[str, Path],
    distributions: Sequence[Path],
) -> Path:
    """Write a candidate whose claim text is copied from the verified aggregate."""

    path = RELEASE_ROOT / "release-candidate.json"
    claim_allowed = any(claim.allowed for claim in evidence.manifest.claims)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "profile": evidence.profile,
        "expected_sha": expected_sha,
        "package_version": validated_project_version(),
        "authoritative_device": "cpu",
        "modality_effect_claims": claim_allowed,
        "claims": [claim.to_dict() for claim in evidence.manifest.claims],
        "evidence": {
            "design_id": evidence.design.design_id,
            "design_sha256": evidence.design.sha256(),
            "design_path": metadata["design"].relative_to(RELEASE_ROOT).as_posix(),
            "evidence_manifest_path": metadata["manifest"].relative_to(RELEASE_ROOT).as_posix(),
            "evidence_manifest_sha256": sha256_file(metadata["manifest"]),
            "verification_path": metadata["verification"].relative_to(RELEASE_ROOT).as_posix(),
            "claim_summary_path": metadata["claims"].relative_to(RELEASE_ROOT).as_posix(),
            "evidence_files_sha256_path": metadata["file_checksums"]
            .relative_to(RELEASE_ROOT)
            .as_posix(),
            "evidence_files_sha256": sha256_file(metadata["file_checksums"]),
            "full_output_path": evidence.output_root.relative_to(ROOT).as_posix(),
            "review_snapshot_path": evidence.snapshot_root.relative_to(ROOT).as_posix(),
            "source_count": evidence.verification.source_count,
            "arm_episode_count": evidence.verification.arm_episode_count,
            "matrix_complete": evidence.manifest.matrix_complete,
            "reduced_design": evidence.manifest.reduced_design,
        },
        "runs": list(evidence.runs),
        "distributions": [
            {
                "path": item.relative_to(RELEASE_ROOT).as_posix(),
                "sha256": sha256_file(item),
            }
            for item in distributions
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_evidence_archive() -> Path:
    archive = RELEASE_ROOT / "lunavla-v2-alpha-evidence.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as stream:
        stream.add(ROOT / "outputs" / "v2", arcname="outputs/v2")
        for name in ("release-candidate.json", "environment-requirements.txt", "sbom.json"):
            stream.add(RELEASE_ROOT / name, arcname=f"release-assets/{name}")
        stream.add(RELEASE_ROOT / "dist", arcname="release-assets/dist")
    return archive


def write_controlled_evidence_archive(
    evidence: ControlledEvidence,
    metadata: dict[str, Path],
) -> Path:
    archive = RELEASE_ROOT / EVIDENCE_ARCHIVE_NAMES[evidence.profile]
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as stream:
        stream.add(
            evidence.output_root,
            arcname=evidence.output_root.relative_to(ROOT).as_posix(),
        )
        stream.add(
            evidence.snapshot_root,
            arcname=evidence.snapshot_root.relative_to(ROOT).as_posix(),
        )
        release_files = (
            RELEASE_ROOT / "release-candidate.json",
            RELEASE_ROOT / "environment-requirements.txt",
            RELEASE_ROOT / "sbom.json",
            *metadata.values(),
        )
        for path in release_files:
            stream.add(path, arcname=f"release-assets/{path.name}")
        stream.add(RELEASE_ROOT / "dist", arcname="release-assets/dist")
    return archive


def write_checksums() -> Path:
    target = RELEASE_ROOT / "SHA256SUMS"
    candidates = tuple(
        path
        for path in _real_tree_files(RELEASE_ROOT, label="release asset tree")
        if path != target
    )
    target.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(RELEASE_ROOT).as_posix()}\n"
            for path in candidates
        ),
        encoding="utf-8",
    )
    return target


def verify_rc_published_evidence() -> list[dict[str, object]]:
    """Re-verify the two registered studies without treating closed claims as failures."""

    language = verify_language_snapshot(ROOT / "results/v2/language-alpha2")
    visual = verify_visual_snapshot(ROOT / "results/v2/visual-beta1")
    rows = (
        (
            "language",
            language,
            PUBLISHED_LANGUAGE_GIT_SHA,
            PUBLISHED_LANGUAGE_EVIDENCE_SHA256,
            PUBLISHED_LANGUAGE_WORKFLOW_URL,
        ),
        (
            "visual",
            visual,
            PUBLISHED_VISUAL_GIT_SHA,
            PUBLISHED_VISUAL_EVIDENCE_SHA256,
            PUBLISHED_VISUAL_WORKFLOW_URL,
        ),
    )
    result: list[dict[str, object]] = []
    for suite, published, git_sha, evidence_sha256, workflow_url in rows:
        if published.git_sha != git_sha or published.evidence_manifest_sha256 != evidence_sha256:
            raise RuntimeError(f"registered {suite} publication identity changed")
        result.append(
            {
                "suite": suite,
                "source_git_sha": git_sha,
                "evidence_manifest_sha256": evidence_sha256,
                "snapshot_manifest_sha256": sha256_file(published.snapshot_manifest_path),
                "workflow_url": workflow_url,
                "claim_allowed": False,
                "statement": published.statement,
            }
        )
    return result


def write_rc_contract_files() -> list[dict[str, str]]:
    """Copy the frozen machine-readable descriptors and human contract boundary."""

    destination = RELEASE_ROOT / "contracts"
    destination.mkdir()
    records: list[dict[str, str]] = []
    for relative in RC_CONTRACT_SOURCES:
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"RC contract source must be a real file: {relative}")
        target = destination / relative.name
        if target.exists():
            raise RuntimeError(f"duplicate RC contract asset name: {target.name}")
        shutil.copyfile(source, target)
        records.append(
            {
                "source_path": relative.as_posix(),
                "release_path": target.relative_to(RELEASE_ROOT).as_posix(),
                "sha256": sha256_file(target),
            }
        )
    return records


def write_rc_candidate(
    *,
    expected_sha: str,
    published_evidence: Sequence[dict[str, object]],
    contracts: Sequence[dict[str, str]],
    distributions: Sequence[Path],
) -> Path:
    path = RELEASE_ROOT / "release-candidate.json"
    environment = RELEASE_ROOT / "environment-requirements.txt"
    sbom = RELEASE_ROOT / "sbom.json"
    payload = {
        "schema_version": 3,
        "profile": "rc",
        "expected_sha": expected_sha,
        "package_version": validated_project_version(),
        "authoritative_device": "cpu",
        "contract_freeze": True,
        "modality_effect_claims": False,
        "published_evidence": list(published_evidence),
        "contracts": list(contracts),
        "release_assets": {
            "environment_requirements": release_asset_record(environment),
            "sbom": release_asset_record(sbom),
            "rc_evidence_archive": {
                "path": RC_ARCHIVE_NAME,
                "sha256_record": RC_INTEGRITY_NAME,
            },
            "integrity_manifest": {"path": RC_INTEGRITY_NAME},
        },
        "distributions": [release_asset_record(item) for item in distributions],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_rc_archive() -> Path:
    archive = RELEASE_ROOT / RC_ARCHIVE_NAME
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as stream:
        for name in (
            "release-candidate.json",
            "environment-requirements.txt",
            "sbom.json",
        ):
            stream.add(RELEASE_ROOT / name, arcname=f"release-assets/{name}")
        stream.add(RELEASE_ROOT / "contracts", arcname="release-assets/contracts")
        stream.add(RELEASE_ROOT / "dist", arcname="release-assets/dist")
    return archive


def write_rc_integrity(expected_sha: str, archive: Path) -> Path:
    """Bind the completed RC archive without introducing a self-reference."""

    paths = (
        RELEASE_ROOT / "release-candidate.json",
        RELEASE_ROOT / "environment-requirements.txt",
        RELEASE_ROOT / "sbom.json",
        archive,
        *sorted((RELEASE_ROOT / "contracts").iterdir()),
        *sorted((RELEASE_ROOT / "dist").iterdir()),
    )
    target = RELEASE_ROOT / RC_INTEGRITY_NAME
    payload = {
        "schema_version": 1,
        "profile": "rc",
        "expected_sha": expected_sha,
        "hash_layer": "post-archive",
        "assets": [release_asset_record(path) for path in paths],
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def write_stable_evidence_metadata(
    *,
    canonical_design_path: Path,
    evidence: ControlledEvidence,
) -> StableEvidence:
    metadata = write_evidence_metadata(evidence)
    canonical_target = RELEASE_ROOT / f"{evidence.profile}-canonical-evidence-design.yaml"
    shutil.copyfile(canonical_design_path, canonical_target)
    metadata["canonical_design"] = canonical_target
    return StableEvidence(
        canonical_design_path=canonical_design_path,
        controlled=evidence,
        metadata=metadata,
    )


def write_stable_file_checksums(
    *,
    studies: Sequence[StableEvidence],
    integration: StableIntegration,
    contracts: Sequence[dict[str, str]],
    distributions: Sequence[Path],
) -> Path:
    candidates: list[Path] = [
        RELEASE_ROOT / "environment-requirements.txt",
        RELEASE_ROOT / "sbom.json",
        integration.manifest_path,
        integration.bundle_path,
        integration.verification_path,
        *distributions,
    ]
    for contract in contracts:
        candidates.append(RELEASE_ROOT / contract["release_path"])
    for study in studies:
        candidates.extend(study.metadata.values())
        for root in (study.controlled.output_root, study.controlled.snapshot_root):
            candidates.extend(
                _real_tree_files(root, label=f"stable {study.controlled.profile} tree")
            )
    unique = sorted(set(candidates), key=lambda path: path.relative_to(ROOT).as_posix())
    if not unique:
        raise RuntimeError("stable evidence produced no files")
    for path in unique:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"stable evidence checksum input must be a real file: {path}")
    target = RELEASE_ROOT / "stable-evidence-files.SHA256SUMS"
    target.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in unique),
        encoding="utf-8",
    )
    return target


def write_stable_archive(
    *,
    studies: Sequence[StableEvidence],
    integration: StableIntegration,
) -> Path:
    archive = RELEASE_ROOT / STABLE_ARCHIVE_NAME
    for study in studies:
        for root in (study.controlled.output_root, study.controlled.snapshot_root):
            _real_tree_files(root, label=f"stable {study.controlled.profile} archive tree")
    _real_tree_files(RELEASE_ROOT / "contracts", label="stable contract archive tree")
    _real_tree_files(RELEASE_ROOT / "dist", label="stable distribution archive tree")
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as stream:
        for study in studies:
            for root in (study.controlled.output_root, study.controlled.snapshot_root):
                stream.add(root, arcname=root.relative_to(ROOT).as_posix())
            for path in study.metadata.values():
                stream.add(path, arcname=f"release-assets/{path.name}")
        stream.add(RELEASE_ROOT / "contracts", arcname="release-assets/contracts")
        stream.add(RELEASE_ROOT / "dist", arcname="release-assets/dist")
        stream.add(
            integration.manifest_path,
            arcname="release-assets/integration/integration_manifest.json",
        )
        stream.add(
            integration.bundle_path,
            arcname="release-assets/integration/integration-attestation-bundle.jsonl",
        )
        stream.add(
            integration.verification_path,
            arcname="release-assets/integration/integration-attestation-verification.json",
        )
        for name in (
            "environment-requirements.txt",
            "sbom.json",
            "stable-evidence-files.SHA256SUMS",
        ):
            stream.add(RELEASE_ROOT / name, arcname=f"release-assets/{name}")
    return archive


def _stable_study_record(study: StableEvidence) -> dict[str, object]:
    evidence = study.controlled
    claims = [claim.to_dict() for claim in evidence.manifest.claims]
    return {
        "profile": evidence.profile,
        "design_id": evidence.design.design_id,
        "canonical_design_path": study.metadata["canonical_design"]
        .relative_to(RELEASE_ROOT)
        .as_posix(),
        "canonical_design_sha256": sha256_file(study.metadata["canonical_design"]),
        "execution_design_path": study.metadata["design"].relative_to(RELEASE_ROOT).as_posix(),
        "execution_design_sha256": sha256_file(study.metadata["design"]),
        "design_sha256": evidence.design.sha256(),
        "evidence_manifest_path": study.metadata["manifest"].relative_to(RELEASE_ROOT).as_posix(),
        "evidence_manifest_sha256": sha256_file(study.metadata["manifest"]),
        "verification_path": study.metadata["verification"].relative_to(RELEASE_ROOT).as_posix(),
        "verification_sha256": sha256_file(study.metadata["verification"]),
        "claim_summary_path": study.metadata["claims"].relative_to(RELEASE_ROOT).as_posix(),
        "claim_summary_sha256": sha256_file(study.metadata["claims"]),
        "full_output_path": evidence.output_root.relative_to(ROOT).as_posix(),
        "review_snapshot_path": evidence.snapshot_root.relative_to(ROOT).as_posix(),
        "source_count": evidence.verification.source_count,
        "arm_episode_count": evidence.verification.arm_episode_count,
        "matrix_complete": evidence.manifest.matrix_complete,
        "reduced_design": evidence.manifest.reduced_design,
        "claims": claims,
        "runs": list(evidence.runs),
    }


def write_stable_candidate(
    *,
    expected_sha: str,
    studies: Sequence[StableEvidence],
    integration: StableIntegration,
    contracts: Sequence[dict[str, str]],
    distributions: Sequence[Path],
    archive: Path,
    evidence_checksums: Path,
) -> Path:
    if validated_project_version() != STABLE_PACKAGE_VERSION:
        raise RuntimeError("stable candidate package version does not match the stable tag")
    study_records = [_stable_study_record(study) for study in studies]
    if [record["profile"] for record in study_records] != ["language", "vision"]:
        raise RuntimeError("stable evidence must contain language then vision")
    expected_matrix = {
        "language": (5, 480),
        "vision": (10, 480),
    }
    for record in study_records:
        profile = str(record["profile"])
        actual = (int(record["source_count"]), int(record["arm_episode_count"]))
        if actual != expected_matrix[profile]:
            raise RuntimeError(
                f"stable {profile} evidence must contain exactly "
                f"{expected_matrix[profile][0]} runs and {expected_matrix[profile][1]} arm-episodes"
            )
        if record["matrix_complete"] is not True or record["reduced_design"] is not False:
            raise RuntimeError(f"stable {profile} evidence must be a complete full design")
    total_sources = sum(int(record["source_count"]) for record in study_records)
    total_episodes = sum(int(record["arm_episode_count"]) for record in study_records)
    if (total_sources, total_episodes) != (15, 960):
        raise RuntimeError("stable evidence must contain exactly 15 runs and 960 arm-episodes")
    claims = [
        claim
        for record in study_records
        for claim in record["claims"]  # type: ignore[union-attr]
    ]
    claim_ids = [claim.get("claim_id") for claim in claims if isinstance(claim, dict)]
    if claim_ids != ["instruction_following", "visual_control_contribution"]:
        raise RuntimeError("stable evidence claim set differs from the frozen design")
    if integration.verified_attestation_count < 1:
        raise RuntimeError("stable integration requires a verified GitHub attestation")
    path = RELEASE_ROOT / "release-candidate.json"
    payload = {
        "schema_version": 4,
        "profile": "stable",
        "expected_sha": expected_sha,
        "package_version": STABLE_PACKAGE_VERSION,
        "expected_tag": STABLE_TAG,
        "default_branch": STABLE_DEFAULT_BRANCH,
        "authoritative_device": "cpu",
        "gpu_required": False,
        "publish_pypi": False,
        "contract_freeze": True,
        "modality_effect_claims": any(
            bool(claim.get("allowed")) for claim in claims if isinstance(claim, dict)
        ),
        "claims": claims,
        "evidence": {
            "post_merge_rerun": True,
            "source_count": total_sources,
            "arm_episode_count": total_episodes,
            "studies": study_records,
        },
        "integration": {
            "manifest_path": integration.manifest_path.relative_to(RELEASE_ROOT).as_posix(),
            "manifest_sha256": integration.manifest_sha256,
            "attestation_bundle_path": integration.bundle_path.relative_to(RELEASE_ROOT).as_posix(),
            "attestation_bundle_sha256": integration.bundle_sha256,
            "attestation_verification_path": integration.verification_path.relative_to(
                RELEASE_ROOT
            ).as_posix(),
            "attestation_verification_sha256": integration.verification_sha256,
            "verified_attestation_count": integration.verified_attestation_count,
            "signer_workflow": STABLE_SIGNER_WORKFLOW,
            "source_ref": STABLE_SOURCE_REF,
            "source_digest": expected_sha,
            "deny_self_hosted_runners": True,
            "claim_allowed": False,
        },
        "contracts": list(contracts),
        "assets": {
            "combined_evidence_archive": {
                "path": archive.relative_to(RELEASE_ROOT).as_posix(),
                "sha256": sha256_file(archive),
            },
            "sbom": {
                "path": "sbom.json",
                "sha256": sha256_file(RELEASE_ROOT / "sbom.json"),
            },
            "environment_requirements": {
                "path": "environment-requirements.txt",
                "sha256": sha256_file(RELEASE_ROOT / "environment-requirements.txt"),
            },
            "evidence_files_sha256": {
                "path": evidence_checksums.relative_to(RELEASE_ROOT).as_posix(),
                "sha256": sha256_file(evidence_checksums),
            },
            "sha256sums_path": "SHA256SUMS",
        },
        "distributions": [
            {
                "path": item.relative_to(RELEASE_ROOT).as_posix(),
                "sha256": sha256_file(item),
            }
            for item in distributions
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _checksum_rows(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"malformed checksum row in {path}: {line!r}")
        if relative in rows:
            raise RuntimeError(f"duplicate checksum path in {path}: {relative}")
        rows[relative] = digest
    return rows


def _verify_checksum_files(path: Path, *, base: Path) -> None:
    """Verify normalized checksum rows against real files below one base."""

    base = base.resolve()
    for relative, expected in _checksum_rows(path).items():
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or "\\" in relative
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            raise RuntimeError(f"unsafe checksum path in {path}: {relative!r}")
        candidate = base / relative_path
        try:
            candidate.resolve().relative_to(base)
        except ValueError as exc:
            raise RuntimeError(f"checksum path escapes its base: {relative!r}") from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"checksum path is not a real file: {relative!r}")
        if sha256_file(candidate) != expected:
            raise RuntimeError(f"checksum mismatch for {relative!r}")


def verify_stable_asset_set(expected_sha: str) -> None:
    """Recompute every top-level asset binding after candidate assembly."""

    candidate_path = RELEASE_ROOT / "release-candidate.json"
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    evidence_payload = payload.get("evidence", {})
    integration_payload = payload.get("integration", {})
    claims = payload.get("claims", [])
    study_records = evidence_payload.get("studies", [])
    claim_ids = (
        [claim.get("claim_id") for claim in claims if isinstance(claim, dict)]
        if isinstance(claims, list)
        else []
    )
    release_contract = {
        "profile": payload.get("profile") == "stable",
        "sha": payload.get("expected_sha") == expected_sha,
        "version": payload.get("package_version") == STABLE_PACKAGE_VERSION,
        "tag": payload.get("expected_tag") == STABLE_TAG,
        "branch": payload.get("default_branch") == STABLE_DEFAULT_BRANCH,
        "device": payload.get("authoritative_device") == "cpu",
        "contract_freeze": payload.get("contract_freeze") is True,
        "pypi": payload.get("publish_pypi") is False,
        "gpu": payload.get("gpu_required") is False,
        "post_merge": evidence_payload.get("post_merge_rerun") is True,
        "matrix": evidence_payload.get("source_count") == 15
        and evidence_payload.get("arm_episode_count") == 960,
        "studies": isinstance(study_records, list)
        and len(study_records) == 2
        and all(isinstance(record, dict) for record in study_records)
        and [record.get("profile") for record in study_records if isinstance(record, dict)]
        == ["language", "vision"]
        and [
            (record.get("source_count"), record.get("arm_episode_count"))
            for record in study_records
            if isinstance(record, dict)
        ]
        == [(5, 480), (10, 480)]
        and all(
            record.get("matrix_complete") is True and record.get("reduced_design") is False
            for record in study_records
            if isinstance(record, dict)
        ),
        "claims": claim_ids == ["instruction_following", "visual_control_contribution"]
        and isinstance(payload.get("modality_effect_claims"), bool)
        and payload.get("modality_effect_claims")
        == any(bool(claim.get("allowed")) for claim in claims if isinstance(claim, dict)),
        "integration": integration_payload.get("source_digest") == expected_sha
        and integration_payload.get("source_ref") == STABLE_SOURCE_REF
        and integration_payload.get("signer_workflow") == STABLE_SIGNER_WORKFLOW
        and integration_payload.get("deny_self_hosted_runners") is True
        and integration_payload.get("claim_allowed") is False
        and isinstance(integration_payload.get("verified_attestation_count"), int)
        and not isinstance(integration_payload.get("verified_attestation_count"), bool)
        and integration_payload["verified_attestation_count"] >= 1,
    }
    failed = [name for name, passed in release_contract.items() if not passed]
    if failed:
        raise RuntimeError("stable candidate contract failed: " + ", ".join(failed))

    hash_records = [
        *payload["distributions"],
        payload["assets"]["combined_evidence_archive"],
        payload["assets"]["sbom"],
        payload["assets"]["environment_requirements"],
        payload["assets"]["evidence_files_sha256"],
        {
            "path": payload["integration"]["manifest_path"],
            "sha256": payload["integration"]["manifest_sha256"],
        },
        {
            "path": payload["integration"]["attestation_bundle_path"],
            "sha256": payload["integration"]["attestation_bundle_sha256"],
        },
        {
            "path": payload["integration"]["attestation_verification_path"],
            "sha256": payload["integration"]["attestation_verification_sha256"],
        },
    ]
    for contract in payload["contracts"]:
        hash_records.append(
            {
                "path": contract["release_path"],
                "sha256": contract["sha256"],
            }
        )
    for study in study_records:
        for path_field, hash_field in (
            ("canonical_design_path", "canonical_design_sha256"),
            ("execution_design_path", "execution_design_sha256"),
            ("evidence_manifest_path", "evidence_manifest_sha256"),
            ("verification_path", "verification_sha256"),
            ("claim_summary_path", "claim_summary_sha256"),
        ):
            hash_records.append(
                {
                    "path": study[path_field],
                    "sha256": study[hash_field],
                }
            )
    for record in hash_records:
        path = RELEASE_ROOT / record["path"]
        if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"stable candidate asset hash mismatch: {record['path']}")

    _verify_checksum_files(
        RELEASE_ROOT / payload["assets"]["evidence_files_sha256"]["path"],
        base=ROOT,
    )
    checksums = _checksum_rows(RELEASE_ROOT / "SHA256SUMS")
    checksum_path = RELEASE_ROOT / "SHA256SUMS"
    actual_files = {
        path.relative_to(RELEASE_ROOT).as_posix(): sha256_file(path)
        for path in _real_tree_files(RELEASE_ROOT, label="stable release asset tree")
        if path != checksum_path
    }
    if checksums != actual_files:
        raise RuntimeError("SHA256SUMS does not exactly cover the stable release asset set")


def run_alpha(expected_sha: str) -> None:
    prepare_release_root()
    try:
        alpha_quality_gate()
        runs = alpha_runs(expected_sha)
        distributions = build_distributions()
        write_environment_and_sbom()
        write_candidate(
            profile="alpha",
            expected_sha=expected_sha,
            runs=runs,
            distributions=distributions,
        )
        write_evidence_archive()
        write_checksums()
    except Exception:
        shutil.rmtree(RELEASE_ROOT, ignore_errors=True)
        raise


def run_rc(expected_sha: str) -> None:
    prepare_release_root()
    try:
        if validated_project_version() != RC_PACKAGE_VERSION:
            raise RuntimeError(f"RC evidence requires package version {RC_PACKAGE_VERSION}")
        alpha_quality_gate()
        published_evidence = verify_rc_published_evidence()
        distributions = build_distributions()
        write_environment_and_sbom()
        contracts = write_rc_contract_files()
        write_rc_candidate(
            expected_sha=expected_sha,
            published_evidence=published_evidence,
            contracts=contracts,
            distributions=distributions,
        )
        archive = write_rc_archive()
        write_rc_integrity(expected_sha, archive)
        write_checksums()
    except Exception:
        shutil.rmtree(RELEASE_ROOT, ignore_errors=True)
        raise


def run_stable(
    expected_sha: str,
    *,
    integration_manifest: Path,
    integration_attestation_bundle: Path,
) -> None:
    tracked_before = prepare_stable_release()
    generated_roots = [ROOT / relative for pair in STABLE_OUTPUTS.values() for relative in pair]
    try:
        verify_stable_main_tip(expected_sha)
        alpha_quality_gate()
        controlled: list[tuple[Path, ControlledEvidence]] = []
        for profile in ("language", "vision"):
            canonical_path, generated_path, design, output_root, snapshot_root = (
                _stable_execution_design(profile)
            )
            evidence = execute_controlled_evidence(
                profile=profile,
                expected_sha=expected_sha,
                design_path=generated_path,
                design=design,
                output_root=output_root,
                snapshot_root=snapshot_root,
            )
            controlled.append((canonical_path, evidence))
        if published_snapshot_hashes() != tracked_before:
            raise RuntimeError("stable evidence modified a tracked publication snapshot")

        distributions = build_distributions()
        write_environment_and_sbom()
        contracts = write_rc_contract_files()
        integration = verify_stable_integration(
            expected_sha=expected_sha,
            manifest_source=integration_manifest,
            bundle_source=integration_attestation_bundle,
        )
        studies = [
            write_stable_evidence_metadata(
                canonical_design_path=canonical_path,
                evidence=evidence,
            )
            for canonical_path, evidence in controlled
        ]
        evidence_checksums = write_stable_file_checksums(
            studies=studies,
            integration=integration,
            contracts=contracts,
            distributions=distributions,
        )
        archive = write_stable_archive(studies=studies, integration=integration)
        write_stable_candidate(
            expected_sha=expected_sha,
            studies=studies,
            integration=integration,
            contracts=contracts,
            distributions=distributions,
            archive=archive,
            evidence_checksums=evidence_checksums,
        )
        write_checksums()
        verify_stable_asset_set(expected_sha)
        if published_snapshot_hashes() != tracked_before:
            raise RuntimeError("stable asset assembly modified a tracked publication snapshot")
    except Exception:
        shutil.rmtree(RELEASE_ROOT, ignore_errors=True)
        for root in generated_roots:
            shutil.rmtree(root, ignore_errors=True)
        raise


def run_controlled_profile(profile: str, expected_sha: str) -> None:
    design_path, design, output_root, snapshot_root = prepare_evidence_release(profile)
    try:
        alpha_quality_gate()
        evidence = execute_controlled_evidence(
            profile=profile,
            expected_sha=expected_sha,
            design_path=design_path,
            design=design,
            output_root=output_root,
            snapshot_root=snapshot_root,
        )
        distributions = build_distributions()
        write_environment_and_sbom()
        metadata = write_evidence_metadata(evidence)
        metadata["file_checksums"] = write_evidence_file_checksums(
            evidence,
            metadata,
            distributions,
        )
        write_evidence_candidate(
            expected_sha=expected_sha,
            evidence=evidence,
            metadata=metadata,
            distributions=distributions,
        )
        write_controlled_evidence_archive(evidence, metadata)
        write_checksums()
    except Exception:
        shutil.rmtree(RELEASE_ROOT, ignore_errors=True)
        shutil.rmtree(output_root, ignore_errors=True)
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_sha = str(args.expected_sha)
    verify_source(expected_sha)
    integration_manifest = args.integration_manifest
    integration_bundle = args.integration_attestation_bundle
    if args.profile == "stable":
        if integration_manifest is None or integration_bundle is None:
            raise ValueError(
                "stable profile requires --integration-manifest and "
                "--integration-attestation-bundle"
            )
    elif integration_manifest is not None or integration_bundle is not None:
        raise ValueError("integration inputs are accepted only by the stable profile")
    if args.profile == "alpha":
        run_alpha(expected_sha)
    elif args.profile in EVIDENCE_PROFILE_DESIGNS:
        run_controlled_profile(args.profile, expected_sha)
    elif args.profile == "rc":
        run_rc(expected_sha)
    else:
        assert integration_manifest is not None
        assert integration_bundle is not None
        run_stable(
            expected_sha,
            integration_manifest=integration_manifest,
            integration_attestation_bundle=integration_bundle,
        )
    print(
        json.dumps(
            {
                "profile": args.profile,
                "expected_sha": expected_sha,
                "release_assets": str(RELEASE_ROOT.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
