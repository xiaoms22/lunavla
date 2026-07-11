from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/v2-release-dispatch.yml"


def _load() -> dict[str, object]:
    payload = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def test_v2_dispatcher_is_manual_and_read_mostly() -> None:
    payload = _load()
    assert set(payload["on"]) == {"workflow_dispatch"}
    dispatch = payload["on"]["workflow_dispatch"]
    inputs = dispatch["inputs"]
    assert set(inputs) == {"source_ref", "expected_sha", "profile"}
    assert inputs["profile"]["options"] == [
        "alpha",
        "language",
        "vision",
        "rc",
        "stable",
    ]
    assert inputs["source_ref"]["default"] == "main"
    assert payload["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }


def test_v2_dispatcher_does_not_interpolate_inputs_in_shell() -> None:
    payload = _load()
    steps = payload["jobs"]["dispatch"]["steps"]
    shell = "\n".join(str(step.get("run", "")) for step in steps)
    assert "${{ inputs." not in shell
    assert "^[0-9a-f]{40}$" in shell
    assert "refs/pull/*" in shell
    assert "git rev-parse --verify HEAD^{commit}" in shell
    assert "scripts/run_v2_release_profile.py" in shell
    assert '--profile "$PROFILE"' in shell
    assert '--expected-sha "$EXPECTED_SHA"' in shell

    checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v4")
    assert checkout["with"]["persist-credentials"] == "false"
    assert checkout["with"]["ref"] == "${{ inputs.source_ref }}"


def test_v2_dispatcher_attests_every_rc_release_layer() -> None:
    payload = _load()
    steps = payload["jobs"]["dispatch"]["steps"]
    attestation = next(
        step for step in steps if step.get("uses") == "actions/attest-build-provenance@v2"
    )
    subjects = set(attestation["with"]["subject-path"].splitlines())
    assert subjects >= {
        "release-assets/dist/*",
        "release-assets/lunavla-v2-*-evidence.tar.gz",
        "release-assets/SHA256SUMS",
        "release-assets/sbom.json",
        "release-assets/release-candidate.json",
        "release-assets/environment-requirements.txt",
    }


def test_stable_dispatch_is_main_only_and_requires_same_workflow_integration() -> None:
    payload = _load()
    integration = payload["jobs"]["stable-integration"]
    assert integration["if"] == "${{ inputs.profile == 'stable' }}"
    integration_steps = integration["steps"]
    integration_shell = "\n".join(str(step.get("run", "")) for step in integration_steps)
    assert 'test "$SOURCE_REF" = "main"' in integration_shell
    assert "refs/remotes/origin/main^{commit}" in integration_shell
    assert "scripts/run_v2_lerobot_integration.py run" in integration_shell
    integration_checkout = next(
        step for step in integration_steps if step.get("uses") == "actions/checkout@v4"
    )
    assert integration_checkout["with"] == {
        "ref": "main",
        "fetch-depth": "1",
        "persist-credentials": "false",
    }
    attestation = next(
        step
        for step in integration_steps
        if step.get("uses") == "actions/attest-build-provenance@v2"
    )
    assert attestation["id"] == "attest-integration"
    assert attestation["with"]["subject-path"].endswith("integration_manifest.json")
    upload = next(
        step for step in integration_steps if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert "stable-integration-${{ inputs.expected_sha }}" == upload["with"]["name"]

    dispatch = payload["jobs"]["dispatch"]
    assert "needs.stable-integration.result == 'success'" in dispatch["if"]
    steps = dispatch["steps"]
    download = next(step for step in steps if step.get("uses") == "actions/download-artifact@v4")
    assert download["if"] == "${{ inputs.profile == 'stable' }}"
    build = next(step for step in steps if step.get("name") == "Build and verify release evidence")
    assert "--integration-manifest" in build["run"]
    assert "--integration-attestation-bundle" in build["run"]
    assert "stable-integration-inputs" in build["run"]
    dispatch_shell = "\n".join(str(step.get("run", "")) for step in steps)
    assert 'test "$SOURCE_REF" = "main"' in dispatch_shell
    assert "refs/remotes/origin/main^{commit}" in dispatch_shell


def test_stable_upload_is_release_assets_only_and_attests_final_contract_files() -> None:
    steps = _load()["jobs"]["dispatch"]["steps"]
    provenance = next(
        step
        for step in steps
        if step.get("name") == "Attest release distributions and evidence bundle"
    )
    subjects = provenance["with"]["subject-path"]
    for path in (
        "release-assets/release-candidate.json",
        "release-assets/sbom.json",
        "release-assets/environment-requirements.txt",
        "release-assets/SHA256SUMS",
        "release-assets/lunavla-v2-*-evidence.tar.gz",
    ):
        assert path in subjects
    stable_upload = next(
        step for step in steps if step.get("name") == "Upload stable release candidate"
    )
    assert stable_upload["if"] == "${{ inputs.profile == 'stable' }}"
    assert stable_upload["with"]["path"] == "release-assets/"


def test_stable_workflow_is_hash_locked_cpu_only_and_never_publishes_to_pypi() -> None:
    payload = _load()
    all_steps = [step for job in payload["jobs"].values() for step in job["steps"]]
    shell = "\n".join(str(step.get("run", "")) for step in all_steps)
    assert "requirements-v2-integration-cpu.lock" in shell
    assert "requirements-v2-release-cpu.lock" in shell
    assert shell.count("--require-hashes") == 2
    assert shell.count("--torch-backend cpu") == 2
    assert shell.count("--no-build-isolation --editable .") == 2
    assert shell.count('torch.__version__ == "2.11.0+cpu"') == 2
    assert shell.count("torch.version.cuda is None") == 2
    assert "forbidden_cpu_packages" in shell
    forbidden_uses = {
        "pypa/gh-action-pypi-publish",
        "pypa/gh-action-pypi-publish@release/v1",
    }
    assert not forbidden_uses.intersection(str(step.get("uses", "")) for step in all_steps)
    assert "twine upload" not in shell
