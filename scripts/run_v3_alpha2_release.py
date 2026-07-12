from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from lunavla.v3 import ExperimentConfig
from lunavla.v3.artifacts import ArtifactHashRecordV1, sha256_file
from lunavla.v3.engine import EngineV3, dataset_for_config
from lunavla.v3.policy import ModelSourceContractV1, PolicyBatchV3
from lunavla.v3.release_contracts import (
    SMOLVLA_VALIDATION_PACKAGE_VERSION,
    SMOLVLA_VALIDATION_TAG,
    SmolVLAValidationCandidateV1,
    GpuValidationManifestV1,
    LicenseReviewV1,
)


ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = ROOT / "configs/v3/smolvla_pretrained_gpu.yaml"
GPU_LOCK = ROOT / "requirements-v3-smolvla-gpu-cu128.lock"
DISPATCHER = ROOT / ".github/workflows/v3-alpha2-release-dispatch.yml"
MAX_MODEL_BYTES = 1_100 * 1024 * 1024
REQUIRED_MODEL_FILES = {
    "config.json",
    "model.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
}


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON document must contain a mapping: {path}")
    return dict(value)


def _write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def require_clean_git(expected_sha: str) -> None:
    if _git("rev-parse", "HEAD") != expected_sha:
        raise RuntimeError("release source SHA does not match the clean checkout")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Alpha 2 release operations require a clean Git checkout")


def project_version() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    package = project.get("version")
    public = getattr(importlib.import_module("lunavla"), "__version__", None)
    if package != public:
        raise RuntimeError(
            f"package version sources disagree: pyproject={package!r}, lunavla={public!r}"
        )
    if not isinstance(package, str):
        raise TypeError("project.version must be a string")
    return package


def _contained(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or path == Path(".") or ".." in path.parts or "\\" in relative:
        raise ValueError(f"release path must be contained: {relative!r}")
    candidate = (root.resolve() / path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"release path escapes its root: {relative!r}")
    return candidate


def _model_source(config: ExperimentConfig) -> ModelSourceContractV1:
    if config.policy["type"] != "lerobot_smolvla":
        raise ValueError("Alpha 2 release operations require policy.type=lerobot_smolvla")
    parameters = config.policy["parameters"]
    return ModelSourceContractV1(
        repo_id=parameters["repo_id"],
        revision=parameters["revision"],
        file_hashes=parameters["file_hashes"],
        license_status=parameters["license_status"],
        pretrained_enabled=parameters["pretrained_enabled"],
    )


def validate_license(
    review_path: str | Path,
    *,
    expected_sha256: str,
    evidence_path: str | Path,
    config_path: str | Path = GPU_CONFIG,
    enable_pretrained_gate: bool,
) -> LicenseReviewV1:
    path = Path(review_path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("license review file hash does not match the immutable request")
    review = LicenseReviewV1.from_mapping(_json(path))
    if sha256_file(evidence_path) != review.evidence_sha256:
        raise ValueError("public license evidence bytes do not match LicenseReviewV1")
    config = ExperimentConfig.load(config_path)
    source = _model_source(config)
    if review.repo_id != source.repo_id or review.revision != source.revision:
        raise ValueError("license review does not match the configured model source")
    if not enable_pretrained_gate:
        raise RuntimeError("enable_pretrained_gate must be explicitly confirmed")
    parameters = config.policy["parameters"]
    if source.license_status != "verified":
        raise RuntimeError("configured model-weight license remains unverified")
    if not source.pretrained_enabled:
        raise RuntimeError("configured pretrained gate remains disabled")
    if parameters.get("conformance_only") is not False:
        raise RuntimeError("configured SmolVLA adapter remains conformance-only")
    if set(source.file_hashes) != REQUIRED_MODEL_FILES:
        raise RuntimeError("verified SmolVLA source must pin every required model and processor file")
    return review


def verify_model_snapshot(snapshot: str | Path, config: ExperimentConfig) -> tuple[ArtifactHashRecordV1, ...]:
    root = Path(snapshot).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"model snapshot does not exist: {root}")
    source = _model_source(config)
    records: list[ArtifactHashRecordV1] = []
    total = 0
    for relative, expected in source.file_hashes.items():
        candidate = _contained(root, relative)
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError(f"model snapshot file is missing or linked: {relative}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ValueError(f"model snapshot hash mismatch: {relative}")
        total += candidate.stat().st_size
        records.append(ArtifactHashRecordV1(relative, actual))
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_files != set(source.file_hashes):
        raise ValueError("model snapshot contains unreviewed or missing files")
    if total > MAX_MODEL_BYTES:
        raise ValueError("model snapshot exceeds the reviewed download limit")
    return tuple(sorted(records, key=lambda item: item.path))


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("cannot hash an empty checkpoint tree")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_hash = bytes.fromhex(sha256_file(item))
        digest.update(file_hash)
    return digest.hexdigest()


def _labels() -> tuple[str, ...]:
    labels = tuple(item.strip() for item in os.environ.get("RUNNER_LABELS", "").split(",") if item.strip())
    required = {"self-hosted", "linux", "x64", "gpu", "lunavla-v3"}
    if not required.issubset(labels):
        raise RuntimeError("runner does not expose the required LunaVLA GPU labels")
    return labels


def _gpu_identity() -> tuple[str, str, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError("Alpha 2 GPU smoke requires exactly one NVIDIA GPU")
    values = [item.strip() for item in output[0].split(",")]
    if len(values) != 3 or any(not item for item in values):
        raise RuntimeError("nvidia-smi returned malformed GPU identity")
    return values[0], hashlib.sha256(values[1].encode("utf-8")).hexdigest(), values[2]


def _action_sha256(values: npt.NDArray[np.generic]) -> str:
    array = np.asarray(values, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def gpu_smoke(
    *,
    config_path: str | Path,
    review_path: str | Path,
    evidence_path: str | Path,
    expected_license_sha256: str,
    expected_git_sha: str,
    snapshot_dir: str | Path,
    output_path: str | Path,
    enable_pretrained_gate: bool,
) -> GpuValidationManifestV1:
    require_clean_git(expected_git_sha)
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("Alpha 2 GPU smoke requires Linux x86_64")
    review = validate_license(
        review_path,
        expected_sha256=expected_license_sha256,
        evidence_path=evidence_path,
        config_path=config_path,
        enable_pretrained_gate=enable_pretrained_gate,
    )
    if project_version() != SMOLVLA_VALIDATION_PACKAGE_VERSION:
        raise RuntimeError(f"GPU smoke requires package version {SMOLVLA_VALIDATION_PACKAGE_VERSION}")
    config = ExperimentConfig.load(config_path)
    records = verify_model_snapshot(snapshot_dir, config)

    import torch
    import torchvision

    if torch.__version__ != "2.11.0+cu128" or torchvision.__version__ != "0.26.0+cu128":
        raise RuntimeError("GPU smoke dependency versions do not match the CUDA 12.8 lock")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Alpha 2 GPU smoke requires exactly one visible CUDA device")
    labels = _labels()
    gpu_name, gpu_uuid_sha256, driver_version = _gpu_identity()

    bundle = dataset_for_config(config)
    engine = EngineV3(config)
    policy, losses = engine.train(bundle.source("train"))
    if engine.policy_spec is None or engine.normalization is None:
        raise RuntimeError("GPU smoke engine did not establish policy contracts")
    samples = EngineV3._samples(
        tuple(bundle.source("train").load()),
        history=engine.policy_spec.history,
        chunk_size=engine.policy_spec.chunk_size,
    )
    checkpoint = Path(output_path).resolve().parent / ".smolvla-gpu-checkpoint"
    if checkpoint.exists():
        raise FileExistsError("GPU smoke checkpoint staging directory already exists")
    try:
        policy.save_checkpoint(checkpoint, metadata={"git_sha": expected_git_sha})
        checkpoint_sha256 = _tree_sha256(checkpoint)
        first = engine.registry.restore(
            checkpoint, config, engine.policy_spec, engine.normalization
        )
        second = engine.registry.restore(
            checkpoint, config, engine.policy_spec, engine.normalization
        )
        batch = PolicyBatchV3(tuple(samples[: config.training["batch_size"]]), device="cuda")
        torch.manual_seed(config.training["seed"])
        first_result = first.train_step(batch, learning_rate=config.training["learning_rate"], step=1)
        first.reset(config.training["seed"])
        first_action = first.predict_chunk(samples[0]).values
        torch.manual_seed(config.training["seed"])
        second_result = second.train_step(batch, learning_rate=config.training["learning_rate"], step=1)
        second.reset(config.training["seed"])
        second_action = second.predict_chunk(samples[0]).values
        if not math.isclose(first_result.loss, second_result.loss, rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError("SmolVLA resumed loss is outside the GPU tolerance")
        if not np.allclose(first_action, second_action, rtol=1e-5, atol=1e-6):
            raise RuntimeError("SmolVLA resumed action is outside the GPU tolerance")
        if first_result.gradient_norm is None or not np.isfinite(first_result.gradient_norm):
            raise RuntimeError("SmolVLA GPU optimizer step did not report a finite gradient norm")
        manifest = GpuValidationManifestV1(
            git_sha=expected_git_sha,
            package_version=project_version(),
            license_review_sha256=review.sha256(),
            model_source_sha256=engine.policy_spec.model_source.sha256(),
            dependency_lock_sha256=sha256_file(GPU_LOCK),
            dispatcher_sha256=sha256_file(DISPATCHER),
            runner_labels=labels,
            runner_os="Linux",
            runner_arch="X64",
            gpu_count=1,
            gpu_name=gpu_name,
            gpu_uuid_sha256=gpu_uuid_sha256,
            driver_version=driver_version,
            cuda_runtime="12.8",
            torch_version=torch.__version__,
            torchvision_version=torchvision.__version__,
            downloaded_files=records,
            model_bytes=sum(_contained(Path(snapshot_dir), item.path).stat().st_size for item in records),
            train_seed=config.training["seed"],
            loss_before=losses[0],
            loss_after=first_result.loss,
            gradient_norm=first_result.gradient_norm,
            checkpoint_sha256=checkpoint_sha256,
            restored_action_sha256=_action_sha256(second_action),
            resume_rtol=1e-5,
            resume_atol=1e-6,
            optimizer_step_verified=True,
            resume_verified=True,
            inference_verified=True,
        )
        manifest.save(output_path)
        return manifest
    finally:
        if checkpoint.exists():
            shutil.rmtree(checkpoint)


def _asset_records(root: Path) -> tuple[ArtifactHashRecordV1, ...]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return tuple(
        ArtifactHashRecordV1(path.relative_to(root).as_posix(), sha256_file(path))
        for path in files
        if path.name not in {"release-candidate.json", "SHA256SUMS"}
    )


def build_candidate(
    *,
    expected_git_sha: str,
    gpu_manifest_path: str | Path,
    gpu_attestation_path: str | Path,
    required_checks_path: str | Path,
    asset_root: str | Path,
    output_path: str | Path,
) -> SmolVLAValidationCandidateV1:
    require_clean_git(expected_git_sha)
    if project_version() != SMOLVLA_VALIDATION_PACKAGE_VERSION:
        raise RuntimeError(f"Alpha 2 candidate requires package version {SMOLVLA_VALIDATION_PACKAGE_VERSION}")
    root = Path(asset_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"release asset root does not exist: {root}")
    gpu_manifest = GpuValidationManifestV1.from_mapping(_json(gpu_manifest_path))
    if gpu_manifest.git_sha != expected_git_sha:
        raise ValueError("GPU manifest does not bind the release source SHA")
    candidate = SmolVLAValidationCandidateV1(
        expected_tag=SMOLVLA_VALIDATION_TAG,
        git_sha=expected_git_sha,
        package_version=project_version(),
        gpu_manifest_sha256=sha256_file(gpu_manifest_path),
        gpu_attestation_sha256=sha256_file(gpu_attestation_path),
        required_checks_sha256=sha256_file(required_checks_path),
        dispatcher_sha256=sha256_file(DISPATCHER),
        assets=_asset_records(root),
    )
    candidate.save(output_path)
    return candidate


def verify_candidate(path: str | Path, asset_root: str | Path) -> SmolVLAValidationCandidateV1:
    candidate = SmolVLAValidationCandidateV1.from_mapping(_json(path))
    root = Path(asset_root).resolve()
    actual = _asset_records(root)
    if actual != candidate.assets:
        raise ValueError("release candidate asset tree does not match its hash inventory")
    manifest_path = _contained(root, "gpu-validation-manifest.json")
    attestation_path = _contained(root, "gpu-attestation-bundle.jsonl")
    if sha256_file(manifest_path) != candidate.gpu_manifest_sha256:
        raise ValueError("release candidate GPU manifest hash mismatch")
    if sha256_file(attestation_path) != candidate.gpu_attestation_sha256:
        raise ValueError("release candidate GPU attestation hash mismatch")
    manifest = GpuValidationManifestV1.from_mapping(_json(manifest_path))
    if manifest.git_sha != candidate.git_sha:
        raise ValueError("release candidate and GPU manifest bind different Git SHAs")
    return candidate


def write_evidence_bundle(asset_root: str | Path) -> Path:
    root = Path(asset_root).resolve()
    target = root / "lunavla-v3-alpha2-evidence.tar.gz"
    if target.exists():
        raise FileExistsError(f"evidence bundle already exists: {target}")
    names = (
        "gpu-validation-manifest.json",
        "gpu-attestation-bundle.jsonl",
        "environment-requirements.txt",
        "sbom.json",
    )
    with tarfile.open(target, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name in names:
            source = _contained(root, name)
            if not source.is_file():
                raise FileNotFoundError(f"evidence input is missing: {name}")
            archive.add(source, arcname=f"release-assets/{name}", recursive=False)
    return target


def finalize_release(
    *,
    candidate_path: str | Path,
    asset_root: str | Path,
    tag: str,
    expected_git_sha: str,
) -> Path:
    if tag != SMOLVLA_VALIDATION_TAG:
        raise ValueError(f"Alpha 2 release tag must be {SMOLVLA_VALIDATION_TAG}")
    require_clean_git(expected_git_sha)
    candidate = verify_candidate(candidate_path, asset_root)
    if candidate.git_sha != expected_git_sha:
        raise ValueError("release candidate does not bind the requested Git SHA")
    subprocess.run(["git", "verify-tag", tag], cwd=ROOT, check=True)
    target = _git("rev-list", "-n", "1", tag)
    if target != expected_git_sha:
        raise ValueError("signed release tag does not point to the expected Git SHA")
    root = Path(asset_root).resolve()
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError("finalized release requires SHA256SUMS")
    expected = {
        item.path: item.sha256
        for item in candidate.assets
    } | {"release-candidate.json": sha256_file(candidate_path)}
    actual: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in actual:
            raise ValueError("SHA256SUMS contains malformed or duplicate rows")
        actual[name] = digest
    if actual != expected:
        raise ValueError("SHA256SUMS does not exactly cover the release asset set")
    return _write_json(
        root / "release-finalization.json",
        {
            "schema_version": 1,
            "tag": tag,
            "git_sha": expected_git_sha,
            "package_version": SMOLVLA_VALIDATION_PACKAGE_VERSION,
            "candidate_sha256": sha256_file(candidate_path),
            "sha256sums_sha256": sha256_file(sums),
            "claim_allowed": False,
            "pypi_published": False,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_v3_alpha2_release.py")
    commands = parser.add_subparsers(dest="command", required=True)
    license_parser = commands.add_parser("validate-license")
    license_parser.add_argument("review")
    license_parser.add_argument("--expected-sha256", required=True)
    license_parser.add_argument("--evidence", required=True)
    license_parser.add_argument("--config", default=str(GPU_CONFIG))
    license_parser.add_argument("--enable-pretrained-gate", action="store_true")
    gpu = commands.add_parser("gpu-smoke")
    gpu.add_argument("--config", default=str(GPU_CONFIG))
    gpu.add_argument("--review", required=True)
    gpu.add_argument("--evidence", required=True)
    gpu.add_argument("--expected-license-sha256", required=True)
    gpu.add_argument("--expected-git-sha", required=True)
    gpu.add_argument("--snapshot-dir", required=True)
    gpu.add_argument("--out", required=True)
    gpu.add_argument("--enable-pretrained-gate", action="store_true")
    verify_gpu = commands.add_parser("verify-gpu")
    verify_gpu.add_argument("manifest")
    build = commands.add_parser("build-candidate")
    build.add_argument("--expected-git-sha", required=True)
    build.add_argument("--gpu-manifest", required=True)
    build.add_argument("--gpu-attestation", required=True)
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
    if arguments.command == "validate-license":
        review = validate_license(
            arguments.review,
            expected_sha256=arguments.expected_sha256,
            evidence_path=arguments.evidence,
            config_path=arguments.config,
            enable_pretrained_gate=arguments.enable_pretrained_gate,
        )
        print(json.dumps({"valid": True, "license_review_sha256": review.sha256()}))
        return 0
    if arguments.command == "gpu-smoke":
        manifest = gpu_smoke(
            config_path=arguments.config,
            review_path=arguments.review,
            evidence_path=arguments.evidence,
            expected_license_sha256=arguments.expected_license_sha256,
            expected_git_sha=arguments.expected_git_sha,
            snapshot_dir=arguments.snapshot_dir,
            output_path=arguments.out,
            enable_pretrained_gate=arguments.enable_pretrained_gate,
        )
        print(json.dumps({"valid": True, "manifest_sha256": manifest.sha256()}))
        return 0
    if arguments.command == "verify-gpu":
        manifest = GpuValidationManifestV1.from_mapping(_json(arguments.manifest))
        print(json.dumps({"valid": True, "git_sha": manifest.git_sha}))
        return 0
    if arguments.command == "build-candidate":
        candidate = build_candidate(
            expected_git_sha=arguments.expected_git_sha,
            gpu_manifest_path=arguments.gpu_manifest,
            gpu_attestation_path=arguments.gpu_attestation,
            required_checks_path=arguments.required_checks,
            asset_root=arguments.asset_root,
            output_path=arguments.out,
        )
        print(json.dumps({"valid": True, "candidate_sha256": candidate.sha256()}))
        return 0
    if arguments.command == "verify-candidate":
        candidate = verify_candidate(arguments.candidate, arguments.asset_root)
        print(json.dumps({"valid": True, "git_sha": candidate.git_sha}))
        return 0
    if arguments.command == "finalize-release":
        path = finalize_release(
            candidate_path=arguments.candidate,
            asset_root=arguments.asset_root,
            tag=arguments.tag,
            expected_git_sha=arguments.expected_git_sha,
        )
        print(json.dumps({"valid": True, "finalization": str(path)}))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
