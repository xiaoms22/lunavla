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
    assert "generate_v3_diagnostic_image_fixtures.py --check" in workflow
    assert "diagnostic_image_ci_design.yaml" in workflow
    assert "outputs/v3/diagnostic-image-ci" in workflow


def test_smolvla_gpu_workflow_is_manual_self_hosted_and_fail_closed() -> None:
    path = Path(".github/workflows/v3-smolvla-gpu.yml")
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(payload["on"]) == {"workflow_dispatch"}
    job = payload["jobs"]["smolvla-gpu-gate"]
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "gpu"]
    workflow = path.read_text(encoding="utf-8")
    assert '"license_status: verified" in config' in workflow
    assert '"pretrained_enabled: true" in config' in workflow
    assert "nvidia-smi" in workflow
    assert "from_pretrained" not in workflow
