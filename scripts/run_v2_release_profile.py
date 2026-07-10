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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean LunaVLA v2 release evidence profile.")
    parser.add_argument(
        "--profile",
        required=True,
        choices=("alpha", "language", "vision", "stable"),
    )
    parser.add_argument("--expected-sha", required=True)
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
    remote = canonical_origin(
        run(("git", "remote", "get-url", "origin"), capture=True)
    )
    if remote not in {
        "https://github.com/xiaoms22/lunavla",
        "git@github.com:xiaoms22/lunavla",
    }:
        raise RuntimeError(f"refusing release evidence from unexpected origin: {remote}")


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
                    (
                        "import json,sys,yaml; "
                        "print(json.dumps(yaml.safe_load(open(sys.argv[1]))))"
                    ),
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
        "arm_episode_count": verification.arm_episode_count
        == plan.expected_arm_episodes,
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
        "instruction_following"
        if profile == "language"
        else "visual_control_contribution"
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
    if any(
        row["manifest_sha256"] != source_hashes[str(row["run_id"])] for row in runs
    ):
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
            f"release distributions do not match project version {version}: "
            f"{sorted(actual_names)}"
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
                f"conflicting installed versions for {canonical}: "
                f"{previous[1]} and {value[1]}"
            )
        versions[canonical] = value
    if not versions:
        raise RuntimeError("the release environment contains no distributions")
    return "".join(
        f"{versions[key][0]}=={versions[key][1]}\n" for key in sorted(versions)
    )


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
            f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n"
            for path in candidates
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
            "evidence_manifest_path": metadata["manifest"]
            .relative_to(RELEASE_ROOT)
            .as_posix(),
            "evidence_manifest_sha256": sha256_file(metadata["manifest"]),
            "verification_path": metadata["verification"]
            .relative_to(RELEASE_ROOT)
            .as_posix(),
            "claim_summary_path": metadata["claims"]
            .relative_to(RELEASE_ROOT)
            .as_posix(),
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
    candidates = sorted(
        path
        for path in RELEASE_ROOT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    target = RELEASE_ROOT / "SHA256SUMS"
    target.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(RELEASE_ROOT).as_posix()}\n"
            for path in candidates
        ),
        encoding="utf-8",
    )
    return target


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
    if args.profile == "alpha":
        run_alpha(expected_sha)
    elif args.profile in EVIDENCE_PROFILE_DESIGNS:
        run_controlled_profile(args.profile, expected_sha)
    else:
        raise RuntimeError(
            f"profile {args.profile!r} is gated until its stable release contract lands"
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
