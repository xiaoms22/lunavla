from __future__ import annotations

from pathlib import Path

import yaml


def test_v3_workflow_targets_integration_and_main() -> None:
    payload = yaml.load(
        Path(".github/workflows/v3-ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    triggers = payload["on"]
    assert set(triggers["pull_request"]["branches"]) == {"main", "v3", "v3-next"}
    assert triggers["push"]["branches"] == ["v3", "v3-next"]
    names = {job["name"] for job in payload["jobs"].values()}
    assert names == {
        "v3-contracts",
        "v3-data",
        "v3-engine-cpu",
        "v3-diagnostics-cpu",
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
    assert "uv pip sync requirements-v3-core-cpu.lock" in workflow
    assert "diagnostic-run" in workflow
    assert "diagnostic-verify" in workflow
    assert "diagnostic-report" in workflow


def test_v31_smolvla_dispatcher_is_manual_self_hosted_and_fail_closed() -> None:
    path = Path(".github/workflows/v3-alpha2-release-dispatch.yml")
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(payload["on"]) == {"workflow_dispatch"}
    assert set(payload["on"]["workflow_dispatch"]["inputs"]) == {
        "phase",
        "source_ref",
        "expected_sha",
        "expected_license_review_sha256",
        "runner_role",
        "expected_container_image_sha256",
        "enable_pretrained_gate",
        "gpu_run_id",
        "tag",
    }
    job = payload["jobs"]["gpu-gate"]
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "gpu", "lunavla-v3"]
    preflight = payload["jobs"]["runner-preflight"]
    assert preflight["runs-on"] == ["self-hosted", "linux", "x64", "gpu", "lunavla-v3"]
    assert "environment" not in preflight
    workflow = path.read_text(encoding="utf-8")
    assert "run_v3_runner_preflight.py qualify" in workflow
    assert "LUNAVLA_EPHEMERAL_RUNNER" in workflow
    assert "LUNAVLA_CONTAINER_IMAGE_SHA256" in workflow
    assert "license_status'] == 'unverified'" in workflow
    assert "spdx_license'] == 'NOASSERTION'" in workflow
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
    assert "v3.1.0-alpha.1" in workflow


def test_alpha3_code_release_dispatcher_is_hosted_and_weight_free() -> None:
    path = Path(".github/workflows/v3-code-release-dispatch.yml")
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(payload["on"]) == {"workflow_dispatch"}
    assert set(payload["jobs"]) == {"candidate", "finalize"}
    assert payload["jobs"]["candidate"]["runs-on"] == "ubuntu-latest"
    assert payload["jobs"]["finalize"]["runs-on"] == "ubuntu-latest"
    workflow = path.read_text(encoding="utf-8")
    assert "run_v3_code_release.py" in workflow
    assert "smolvla-conformance-status.json" in workflow
    assert "weight_network_accessed': False" in workflow
    assert "v3.0.0-alpha.3" in workflow
    assert workflow.count("python -m build --no-isolation") == 2
    assert workflow.count("normalize-sdist") == 2
    assert '"$root/required-checks.json"' in workflow
    assert "RELEASE_SIGNER_PRINCIPAL" in workflow
    assert "RELEASE_SIGNER_PUBLIC_KEY" in workflow
    assert 'gpg.ssh.allowedSignersFile "$RUNNER_TEMP/allowed_signers"' in workflow
    assert "verification']['verified'] is True" in workflow
    assert workflow.count('echo "$PWD/.venv/bin" >> "$GITHUB_PATH"') == 2
    assert workflow.count('export PATH="$PWD/.venv/bin:$PATH"') == 1
    assert "self-hosted" not in workflow

    signer = Path("docs/v3/release/allowed_signers").read_text(encoding="utf-8")
    assert signer.rstrip().endswith("/7U2ePLaQn")


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
    assert "setuptools==80.10.2" in release
    assert "nvidia-" not in release and "triton==" not in release
