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
        "v3-v2-compat",
        "v3-secret-scan",
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
