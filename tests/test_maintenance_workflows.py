from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> dict[str, object]:
    path = ROOT / ".github" / "workflows" / name
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def test_fast_ci_maintains_main_and_v1x_without_collecting_v2_tests() -> None:
    payload = _workflow("ci.yml")
    triggers = payload["on"]
    assert set(triggers["push"]["branches"]) == {"main", "v1.x"}
    assert set(triggers["pull_request"]["branches"]) == {"main", "v1.x"}

    test_job = payload["jobs"]["tests"]
    assert test_job["strategy"]["matrix"]["python-version"] == ["3.10", "3.11", "3.12"]
    steps = test_job["steps"]
    test_step = next(step for step in steps if step.get("name") == "Run test suite with coverage gate")
    command = str(test_step["run"])
    assert "python -m pytest tests " in command
    assert "tests_v2" not in command

    workflow_text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "requirements-v2" not in workflow_text
    assert "--extra v2" not in workflow_text
    assert "ruff check scripts/run_v2_release_profile.py" in workflow_text
    assert "mypy dataset model trainer lunavla scripts/run_v2_release_profile.py" in workflow_text


def test_v2_cpu_ci_targets_main_and_v2_without_renaming_required_checks() -> None:
    payload = _workflow("v2-ci.yml")
    triggers = payload["on"]
    expected_branches = {"main", "v2"}
    assert set(triggers["push"]["branches"]) == expected_branches
    assert set(triggers["pull_request"]["branches"]) == expected_branches
    assert "workflow_dispatch" in triggers

    jobs = payload["jobs"]
    required_names = {
        "v2-contracts",
        "v2-torch-cpu",
        "v2-lerobot-adapter",
        "v1.1-compat",
        "v2-secret-scan",
    }
    assert required_names <= {job["name"] for job in jobs.values()}

    setup_versions = {
        str(step["with"]["python-version"])
        for job in jobs.values()
        for step in job["steps"]
        if step.get("uses") == "actions/setup-python@v5"
    }
    assert setup_versions == {"3.12"}
    contract_commands = "\n".join(
        str(step.get("run", "")) for step in jobs["contracts"]["steps"]
    )
    assert "ruff check scripts/run_v2_release_profile.py" in contract_commands
    assert "mypy lunavla scripts/run_v2_release_profile.py" in contract_commands


def test_v2_cpu_jobs_keep_dependency_profiles_and_test_suites_separate() -> None:
    payload = _workflow("v2-ci.yml")
    jobs = payload["jobs"]

    commands = {
        job_id: "\n".join(
            str(step.get("run", ""))
            for step in job["steps"]
            if isinstance(step, dict)
        )
        for job_id, job in jobs.items()
    }
    assert 'pytest tests_v2 -m "not torch and not lerobot"' in commands["contracts"]
    assert "requirements-v2-core-cpu.lock" in commands["torch-cpu"]
    assert "requirements-v2-cpu.lock" not in commands["torch-cpu"]
    assert "pytest tests_v2 -m torch" in commands["torch-cpu"]
    assert "requirements-v2-cpu.lock" in commands["lerobot-adapter"]
    assert "pytest tests_v2 -m lerobot" in commands["lerobot-adapter"]
    assert "requirements-dev.txt" in commands["v11-compat"]
    assert "pytest tests " in commands["v11-compat"]
    assert "tests_v2" not in commands["v11-compat"]


def test_gpu_training_stays_manual_and_does_not_block_pull_requests() -> None:
    payload = _workflow("v2-gpu.yml")
    assert set(payload["on"]) == {"workflow_dispatch"}
    assert payload["jobs"]["gpu-training"]["name"] == "v2-gpu-training"


def test_codeql_covers_maintained_and_v2_integration_branches() -> None:
    payload = _workflow("codeql.yml")
    triggers = payload["on"]
    expected = {"main", "v1.x", "v2", "v3"}
    assert set(triggers["push"]["branches"]) == expected
    assert set(triggers["pull_request"]["branches"]) == expected
