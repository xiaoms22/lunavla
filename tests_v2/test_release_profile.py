from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.run_v2_release_profile import (
    SHA256_PATTERN,
    canonical_origin,
    installed_requirements,
    sbom_command,
    sha256_file,
    validated_project_version,
)


def test_release_sha_contract_is_exact() -> None:
    assert SHA256_PATTERN.fullmatch("a" * 40)
    assert not SHA256_PATTERN.fullmatch("A" * 40)
    assert not SHA256_PATTERN.fullmatch("a" * 39)
    assert not SHA256_PATTERN.fullmatch("../" + "a" * 40)


def test_release_sha256_file(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"lunavla-v2\n")
    assert sha256_file(source) == hashlib.sha256(b"lunavla-v2\n").hexdigest()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "https://github.com/xiaoms22/lunavla.git",
            "https://github.com/xiaoms22/lunavla",
        ),
        (
            "https://github.com/xiaoms22/lunavla/",
            "https://github.com/xiaoms22/lunavla",
        ),
        ("git@github.com:xiaoms22/lunavla.git", "git@github.com:xiaoms22/lunavla"),
    ],
)
def test_release_origin_normalization(raw: str, expected: str) -> None:
    assert canonical_origin(raw) == expected


def test_installed_requirements_is_sorted_and_contains_project() -> None:
    rows = installed_requirements().splitlines()
    canonical = [row.split("==", maxsplit=1)[0].lower().replace("_", "-") for row in rows]
    assert canonical == sorted(canonical)
    assert any(row.lower().startswith("lunavla==") for row in rows)


def test_release_version_contract_matches_all_sources() -> None:
    assert validated_project_version() == "2.0.0a1"


def test_release_version_contract_fails_closed_on_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    monkeypatch.setattr(release.importlib.metadata, "version", lambda _: "2.0.0a0")
    with pytest.raises(RuntimeError, match="inconsistent LunaVLA version contract"):
        release.validated_project_version()


def test_sbom_command_uses_cyclonedx_7_output_contract(tmp_path: Path) -> None:
    command = sbom_command(tmp_path / "requirements.txt", tmp_path / "sbom.json")
    assert command[:2] == ("cyclonedx-py", "requirements")
    assert "--output-reproducible" in command
    assert "--output-file" in command
    assert "--outfile" not in command
    assert command[-1].endswith("sbom.json")


def test_non_alpha_profiles_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_v2_release_profile as release

    monkeypatch.setattr(release, "verify_source", lambda expected_sha: None)
    with pytest.raises(RuntimeError, match="gated until"):
        release.main(["--profile", "vision", "--expected-sha", "a" * 40])
