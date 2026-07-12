from __future__ import annotations

import argparse
import copy
import gzip
import io
import importlib
import json
import subprocess
import tarfile
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from lunavla.v3.artifacts import ArtifactHashRecordV1, sha256_file
from lunavla.v3.release_contracts import (
    ALPHA3_PACKAGE_VERSION,
    ALPHA3_TAG,
    Alpha3ReleaseCandidateV1,
    WeightLicenseStatusV1,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_API = ROOT / "docs/v3/public_api_contract.json"
CORE_LOCK = ROOT / "requirements-v3-core-cpu.lock"
DIFFUSION_LOCK = ROOT / "requirements-v3-diffusion-cpu.lock"
SMOLVLA_LOCK = ROOT / "requirements-v3-smolvla-cpu.lock"
LICENSE_STATUS = ROOT / "docs/v3/release/smolvla-license-status.json"
DISPATCHER = ROOT / ".github/workflows/v3-code-release-dispatch.yml"
EVIDENCE_ARCHIVE = "lunavla-v3-alpha3-code-evidence.tar.gz"
FORBIDDEN_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
REQUIRED_CHECKS = {
    "CodeQL Python",
    "v3-contracts",
    "v3-data",
    "v3-diffusion-cpu",
    "v3-engine-cpu",
    "v3-secret-scan",
    "v3-smolvla-adapter",
    "v3-v2-compat",
}


def normalize_sdist(path: str | Path, *, source_date_epoch: int) -> Path:
    """Rewrite a setuptools sdist with deterministic archive metadata."""

    if (
        not isinstance(source_date_epoch, int)
        or isinstance(source_date_epoch, bool)
        or source_date_epoch < 0
    ):
        raise ValueError("source_date_epoch must be a non-negative integer")
    source = Path(path).resolve()
    if not source.is_file() or not source.name.endswith(".tar.gz"):
        raise ValueError("sdist must be an existing .tar.gz file")
    target = source.with_name(f".{source.name}.normalized.tmp")
    try:
        with tarfile.open(source, mode="r:gz") as incoming:
            members = sorted(incoming.getmembers(), key=lambda member: member.name)
            names: set[str] = set()
            for member in members:
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"sdist member escapes the archive root: {member.name}")
                if member.name in names:
                    raise ValueError(f"sdist contains a duplicate member: {member.name}")
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"sdist contains an unsupported member type: {member.name}")
                names.add(member.name)
            with target.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, mtime=source_date_epoch
                ) as compressed:
                    with tarfile.open(
                        fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                    ) as outgoing:
                        for member in members:
                            normalized = copy.copy(member)
                            normalized.uid = 0
                            normalized.gid = 0
                            normalized.uname = ""
                            normalized.gname = ""
                            normalized.mtime = source_date_epoch
                            normalized.pax_headers = {}
                            normalized.mode = 0o755 if member.isdir() else 0o644
                            payload = incoming.extractfile(member) if member.isfile() else None
                            outgoing.addfile(normalized, payload)
        target.replace(source)
    finally:
        target.unlink(missing_ok=True)
    return source


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON document must contain a mapping: {path}")
    return dict(value)


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def require_clean_git(expected_sha: str) -> None:
    if _git("rev-parse", "HEAD") != expected_sha:
        raise RuntimeError("code release source SHA does not match the checkout")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("code release requires a clean Git checkout")


def project_version() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    package = project.get("version")
    public = getattr(importlib.import_module("lunavla"), "__version__", None)
    if package != public or package != ALPHA3_PACKAGE_VERSION:
        raise RuntimeError("Alpha 3 package version sources must both equal 3.0.0a3")
    return package


def _asset_records(root: Path) -> tuple[ArtifactHashRecordV1, ...]:
    records: list[ArtifactHashRecordV1] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if path.name in {"release-candidate.json", "SHA256SUMS", "release-finalization.json"}:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or "checkpoint" in relative.lower():
            raise ValueError(f"code-only release contains a forbidden model artifact: {relative}")
        records.append(ArtifactHashRecordV1(relative, sha256_file(path)))
    return tuple(records)


def verify_test_manifest(path: str | Path, *, expected_git_sha: str) -> dict[str, Any]:
    value = _json(path)
    expected_fields = {
        "schema_version",
        "git_sha",
        "checks",
        "pretrained_enabled",
        "weight_network_accessed",
        "claim_allowed",
    }
    if set(value) != expected_fields:
        raise ValueError("test manifest fields do not match the code-only contract")
    if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
        raise ValueError("test manifest schema_version must be integer 1")
    if value["git_sha"] != expected_git_sha:
        raise ValueError("test manifest does not bind the release Git SHA")
    checks = value["checks"]
    if not isinstance(checks, Mapping) or set(checks) != REQUIRED_CHECKS:
        raise ValueError("test manifest must contain the exact Alpha 3 required checks")
    if any(result != "success" for result in checks.values()):
        raise ValueError("every Alpha 3 required check must succeed")
    if value["pretrained_enabled"] is not False or value["weight_network_accessed"] is not False:
        raise ValueError("Alpha 3 code release forbids pretrained weights and weight network access")
    if value["claim_allowed"] is not False:
        raise ValueError("Alpha 3 code release cannot open scientific claims")
    return value


def write_evidence_bundle(asset_root: str | Path) -> Path:
    root = Path(asset_root).resolve()
    target = root / EVIDENCE_ARCHIVE
    if target.exists():
        raise FileExistsError(f"evidence archive already exists: {target}")
    names = (
        "environment-requirements.txt",
        "provenance.jsonl",
        "sbom.json",
        "smolvla-conformance-status.json",
        "test-manifest.json",
    )
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name in names:
                    source = root / name
                    if not source.is_file() or source.is_symlink():
                        raise FileNotFoundError(f"code evidence input is missing: {name}")
                    payload = source.read_bytes()
                    info = tarfile.TarInfo(f"release-assets/{name}")
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(payload))
    return target


def build_candidate(
    *,
    expected_git_sha: str,
    required_checks_path: str | Path,
    asset_root: str | Path,
    output_path: str | Path,
) -> Alpha3ReleaseCandidateV1:
    require_clean_git(expected_git_sha)
    project_version()
    root = Path(asset_root).resolve()
    verify_test_manifest(root / "test-manifest.json", expected_git_sha=expected_git_sha)
    status = WeightLicenseStatusV1.from_mapping(_json(LICENSE_STATUS))
    conformance = _json(root / "smolvla-conformance-status.json")
    if conformance != {
        "schema_version": 1,
        "repo_id": status.repo_id,
        "revision": status.revision,
        "license_status": "unverified",
        "spdx_license": "NOASSERTION",
        "pretrained_enabled": False,
        "conformance_only": True,
        "weight_accessed": False,
    }:
        raise ValueError("SmolVLA conformance status does not match the fail-closed source record")
    candidate = Alpha3ReleaseCandidateV1(
        expected_tag=ALPHA3_TAG,
        git_sha=expected_git_sha,
        package_version=project_version(),
        public_api_sha256=sha256_file(PUBLIC_API),
        core_lock_sha256=sha256_file(CORE_LOCK),
        diffusion_lock_sha256=sha256_file(DIFFUSION_LOCK),
        smolvla_lock_sha256=sha256_file(SMOLVLA_LOCK),
        weight_license_status_sha256=sha256_file(LICENSE_STATUS),
        required_checks_sha256=sha256_file(required_checks_path),
        dispatcher_sha256=sha256_file(DISPATCHER),
        assets=_asset_records(root),
    )
    candidate.save(output_path)
    return candidate


def verify_candidate(path: str | Path, asset_root: str | Path) -> Alpha3ReleaseCandidateV1:
    candidate = Alpha3ReleaseCandidateV1.from_mapping(_json(path))
    root = Path(asset_root).resolve()
    if candidate.assets != _asset_records(root):
        raise ValueError("Alpha 3 asset inventory differs from the candidate")
    expected = {
        "public_api_sha256": sha256_file(PUBLIC_API),
        "core_lock_sha256": sha256_file(CORE_LOCK),
        "diffusion_lock_sha256": sha256_file(DIFFUSION_LOCK),
        "smolvla_lock_sha256": sha256_file(SMOLVLA_LOCK),
        "weight_license_status_sha256": sha256_file(LICENSE_STATUS),
        "dispatcher_sha256": sha256_file(DISPATCHER),
    }
    for field, digest in expected.items():
        if getattr(candidate, field) != digest:
            raise ValueError(f"Alpha 3 candidate {field} drifted")
    verify_test_manifest(root / "test-manifest.json", expected_git_sha=candidate.git_sha)
    return candidate


def finalize_release(
    *, candidate_path: str | Path, asset_root: str | Path, tag: str, expected_git_sha: str
) -> Path:
    if tag != ALPHA3_TAG:
        raise ValueError(f"code-only Alpha 3 tag must be {ALPHA3_TAG}")
    require_clean_git(expected_git_sha)
    candidate = verify_candidate(candidate_path, asset_root)
    if candidate.git_sha != expected_git_sha:
        raise ValueError("candidate and requested release SHA differ")
    subprocess.run(["git", "verify-tag", tag], cwd=ROOT, check=True)
    if _git("rev-list", "-n", "1", tag) != expected_git_sha:
        raise ValueError("signed tag does not point to the release SHA")
    root = Path(asset_root).resolve()
    sums = root / "SHA256SUMS"
    expected = {item.path: item.sha256 for item in candidate.assets}
    expected["release-candidate.json"] = sha256_file(candidate_path)
    actual: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in actual:
            raise ValueError("SHA256SUMS contains malformed or duplicate rows")
        actual[name] = digest
    if actual != expected:
        raise ValueError("SHA256SUMS does not exactly cover the code release assets")
    result = root / "release-finalization.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tag": tag,
                "git_sha": expected_git_sha,
                "package_version": ALPHA3_PACKAGE_VERSION,
                "candidate_sha256": sha256_file(candidate_path),
                "sha256sums_sha256": sha256_file(sums),
                "claim_allowed": False,
                "pypi_published": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_v3_code_release.py")
    commands = parser.add_subparsers(dest="command", required=True)
    normalize = commands.add_parser("normalize-sdist")
    normalize.add_argument("sdist")
    normalize.add_argument("--source-date-epoch", required=True, type=int)
    bundle = commands.add_parser("bundle")
    bundle.add_argument("--asset-root", required=True)
    build = commands.add_parser("build-candidate")
    build.add_argument("--expected-git-sha", required=True)
    build.add_argument("--required-checks", required=True)
    build.add_argument("--asset-root", required=True)
    build.add_argument("--out", required=True)
    verify = commands.add_parser("verify-candidate")
    verify.add_argument("candidate")
    verify.add_argument("--asset-root", required=True)
    finalize = commands.add_parser("finalize-release")
    finalize.add_argument("candidate")
    finalize.add_argument("--asset-root", required=True)
    finalize.add_argument("--tag", required=True)
    finalize.add_argument("--expected-git-sha", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "normalize-sdist":
        result = normalize_sdist(
            arguments.sdist, source_date_epoch=arguments.source_date_epoch
        )
        print(json.dumps({"normalized": str(result)}))
        return 0
    if arguments.command == "bundle":
        print(json.dumps({"archive": str(write_evidence_bundle(arguments.asset_root))}))
        return 0
    if arguments.command == "build-candidate":
        candidate = build_candidate(
            expected_git_sha=arguments.expected_git_sha,
            required_checks_path=arguments.required_checks,
            asset_root=arguments.asset_root,
            output_path=arguments.out,
        )
        print(json.dumps({"candidate_sha256": candidate.sha256()}))
        return 0
    if arguments.command == "verify-candidate":
        candidate = verify_candidate(arguments.candidate, arguments.asset_root)
        print(json.dumps({"valid": True, "git_sha": candidate.git_sha}))
        return 0
    if arguments.command == "finalize-release":
        result = finalize_release(
            candidate_path=arguments.candidate,
            asset_root=arguments.asset_root,
            tag=arguments.tag,
            expected_git_sha=arguments.expected_git_sha,
        )
        print(json.dumps({"valid": True, "finalization": str(result)}))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
