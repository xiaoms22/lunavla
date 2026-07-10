from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/v2-lerobot-integration-dispatch.yml"


def _load() -> dict[str, object]:
    payload = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def test_lerobot_dispatcher_is_manual_and_manifest_only() -> None:
    payload = _load()
    assert set(payload["on"]) == {"workflow_dispatch"}
    dispatch = payload["on"]["workflow_dispatch"]
    assert set(dispatch["inputs"]) == {"source_ref", "expected_sha"}
    assert payload["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }

    steps = payload["jobs"]["integration"]["steps"]
    attestation = next(
        step for step in steps if step.get("uses") == "actions/attest-build-provenance@v2"
    )
    upload = next(
        step for step in steps if step.get("uses") == "actions/upload-artifact@v4"
    )
    manifest = (
        "${{ runner.temp }}/lunavla-lerobot-integration/integration_manifest.json"
    )
    assert attestation["with"]["subject-path"] == manifest
    assert upload["with"]["path"] == manifest


def test_lerobot_dispatcher_validates_before_source_checkout() -> None:
    payload = _load()
    steps = payload["jobs"]["integration"]["steps"]
    shell = "\n".join(str(step.get("run", "")) for step in steps)

    assert "${{ inputs." not in shell
    assert "^[0-9a-f]{40}$" in shell
    assert "refs/pull/*" in shell
    assert "git check-ref-format --branch" in shell
    assert "git rev-parse --verify HEAD^{commit}" in shell
    assert "requirements-v2-integration-cpu.lock" in shell
    assert "uv==0.11.26" in shell
    assert "--require-hashes" in shell
    assert "--torch-backend cpu" in shell
    assert "forbidden_cpu_packages" in shell
    assert "scripts/run_v2_lerobot_integration.py run" in shell
    assert "scripts/run_v2_lerobot_integration.py verify" in shell

    checkout_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses") == "actions/checkout@v4"
    )
    assert checkout_index == 1
    assert "scripts/" not in str(steps[0].get("run", ""))

    checkout = steps[checkout_index]
    assert checkout["with"] == {
        "ref": "${{ inputs.source_ref }}",
        "fetch-depth": "1",
        "persist-credentials": "false",
    }
