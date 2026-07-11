from __future__ import annotations

from pathlib import Path

import yaml


def test_v3_workflow_targets_integration_and_main() -> None:
    payload = yaml.load(
        Path(".github/workflows/v3-ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    triggers = payload["on"]
    assert set(triggers["pull_request"]["branches"]) == {"main", "v3"}
    assert triggers["push"]["branches"] == ["v3"]
    names = {job["name"] for job in payload["jobs"].values()}
    assert names == {
        "v3-contracts",
        "v3-data",
        "v3-engine-cpu",
        "v3-diffusion-cpu",
        "v3-v2-compat",
        "v3-secret-scan",
        "v3-smolvla-adapter",
    }


def test_v3_cpu_lock_is_an_exact_reviewed_alias() -> None:
    lines = Path("requirements-v3-core-cpu.lock").read_text(encoding="utf-8").splitlines()
    assert lines[-1] == "-r requirements-v2-core-cpu.lock"


def test_v3_cpu_job_enforces_hashes_and_rejects_accelerator_packages() -> None:
    workflow = Path(".github/workflows/v3-ci.yml").read_text(encoding="utf-8")
    assert "uv==0.11.26" in workflow
    assert "uv pip sync requirements-v3-core-cpu.lock" in workflow
    assert "--require-hashes --strict --only-binary :all: --torch-backend cpu" in workflow
    assert 'torch.__version__ == "2.11.0+cpu"' in workflow
    assert 'torchvision.__version__ == "0.26.0+cpu"' in workflow
    assert "forbidden_cpu_packages(installed)" in workflow
    assert "validate-config configs/v3/act_fake_libero_cpu.yaml" in workflow
    assert "uv pip sync requirements-v3-diffusion-cpu.lock" in workflow
    assert 'metadata.version("lerobot") == "0.6.0"' in workflow
    assert "validate-config configs/v3/diffusion_fake_libero_cpu.yaml" in workflow
    assert "uv pip sync requirements-v3-smolvla-cpu.lock" in workflow
    assert "validate-config configs/v3/smolvla_conformance_cpu.yaml" in workflow


def test_smolvla_release_dispatcher_is_manual_self_hosted_and_fail_closed() -> None:
    path = Path(".github/workflows/v3-alpha2-release-dispatch.yml")
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(payload["on"]) == {"workflow_dispatch"}
    assert set(payload["on"]["workflow_dispatch"]["inputs"]) == {
        "phase",
        "source_ref",
        "expected_sha",
        "expected_license_review_sha256",
        "enable_pretrained_gate",
        "gpu_run_id",
        "tag",
    }
    job = payload["jobs"]["gpu-gate"]
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "gpu", "lunavla-v3"]
    workflow = path.read_text(encoding="utf-8")
    assert "validate-license" in workflow
    assert "--enable-pretrained-gate" in workflow
    assert "requirements-v3-smolvla-gpu-cu128.lock" in workflow
    assert "requirements-v3-release-cpu.lock" in workflow
    assert "gpu-validation-manifest.json" in workflow
    assert "actions/attest-build-provenance@v2" in workflow
    assert "git verify-tag" in workflow
    assert "verification']['verified'] is True" in workflow
    assert "gh release create" in workflow and "--draft --prerelease" in workflow
    assert "nvidia-smi" in workflow


def test_gpu_and_release_locks_pin_authoritative_platforms() -> None:
    gpu = Path("requirements-v3-smolvla-gpu-cu128.lock").read_text(encoding="utf-8")
    assert "torch==2.11.0+cu128" in gpu
    assert "torchvision==0.26.0+cu128" in gpu
    assert "nvidia-cuda-runtime-cu12==12.8.90" in gpu
    assert "triton==3.6.0" in gpu
    release = Path("requirements-v3-release-cpu.lock").read_text(encoding="utf-8")
    assert "torch==2.11.0+cpu" in release
    assert "torchvision==0.26.0+cpu" in release
    assert "cyclonedx-bom==7" in release
    assert "nvidia-" not in release and "triton==" not in release
