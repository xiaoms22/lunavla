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
