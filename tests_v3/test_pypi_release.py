from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts import run_v3_pypi_release as release


def test_publisher_provenance_requires_exact_github_identity() -> None:
    payload = {
        "attestation_bundles": [
            {
                "publisher": {
                    "kind": "GitHub",
                    "repository": "xiaoms22/lunavla",
                    "workflow": "v3-pypi-release.yml",
                    "environment": "pypi",
                },
                "attestations": [{"version": 1}],
            }
        ]
    }
    assert release._publisher_from_provenance(payload) == release.PUBLISHER
    payload["attestation_bundles"][0]["publisher"]["workflow"] = "other.yml"
    with pytest.raises(ValueError, match="exact trusted publisher"):
        release._publisher_from_provenance(payload)


def test_pypi_preflight_rejects_existing_stable_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release,
        "_get_json",
        lambda *_args, **_kwargs: {"releases": {"3.0.0": [{"filename": "existing"}]}},
    )
    with pytest.raises(ValueError, match="overwrite is forbidden"):
        release.verify_version_absent("https://pypi.example")


def test_pypi_workflow_is_manual_pinned_and_does_not_rebuild() -> None:
    path = Path(".github/workflows/v3-pypi-release.yml")
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(payload["on"]) == {"workflow_dispatch"}
    assert set(payload["jobs"]) == {"publish", "finalize"}
    publish = payload["jobs"]["publish"]
    assert publish["environment"] == "pypi"
    assert publish["permissions"] == {
        "actions": "read",
        "contents": "read",
        "id-token": "write",
    }
    workflow = path.read_text(encoding="utf-8")
    assert "pypa/gh-action-pypi-publish@cef221092ed1bacb1cc03d23a2d87d1d172e277b" in workflow
    assert "skip-existing: false" in workflow
    assert "python -m build" not in workflow
    assert "git verify-tag" in workflow
    assert "v3.0.0" in workflow
    for line in workflow.splitlines():
        if "uses:" in line:
            revision = line.rsplit("@", 1)[-1]
            assert len(revision) == 40
