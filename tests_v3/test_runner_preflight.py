from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lunavla.v3 import RunnerQualificationManifestV1
from scripts import run_v3_runner_preflight as preflight


SHA = "a" * 64
GIT_SHA = "b" * 40


def test_preflight_qualify_writes_only_fail_closed_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LUNAVLA_EPHEMERAL_RUNNER", "true")
    monkeypatch.setenv("LUNAVLA_CONTAINER_IMAGE_SHA256", SHA)
    monkeypatch.setenv("RUNNER_LABELS", "self-hosted,linux,x64,gpu,lunavla-v3")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setattr(preflight, "_require_clean_git", lambda _sha: None)
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(preflight.platform, "python_version", lambda: "3.12.10")
    monkeypatch.setattr(preflight, "_container_isolated", lambda: True)
    monkeypatch.setattr(preflight, "_private_mounts", lambda: ())
    monkeypatch.setattr(preflight, "_weight_accessed", lambda: False)
    monkeypatch.setattr(
        preflight,
        "_gpu_identity",
        lambda: ("NVIDIA A100-SXM4-80GB", SHA, "570.124.06"),
    )
    monkeypatch.setattr(preflight, "_os_version", lambda: "Ubuntu 22.04.5 LTS")
    monkeypatch.setattr(preflight, "_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(preflight, "_verify_network", lambda: preflight.NETWORK_HOSTS)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _path: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(preflight.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(preflight, "sha256_file", lambda _path: SHA)
    fake_torch = SimpleNamespace(
        __version__="2.11.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "torchvision", SimpleNamespace(__version__="0.26.0+cu128")
    )

    path = tmp_path / "runner-qualification-manifest.json"
    manifest = preflight.qualify(
        role="authoritative",
        expected_git_sha=GIT_SHA,
        container_image_sha256=SHA,
        runner_name="private-runner-name",
        output_path=path,
    )

    assert manifest.weight_accessed is False
    assert manifest.release_eligible is False
    assert manifest.claim_allowed is False
    assert "private-runner-name" not in path.read_text(encoding="utf-8")
    assert preflight.verify(path) == manifest


def test_preflight_verify_rejects_tampered_manifest(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "role": "secondary",
        "git_sha": GIT_SHA,
        "dependency_lock_sha256": SHA,
        "container_image_sha256": SHA,
        "runner_name_sha256": SHA,
        "runner_labels": ["self-hosted", "linux", "x64", "gpu", "lunavla-v3"],
        "runner_os": "Linux",
        "runner_os_version": "Ubuntu 22.04.5 LTS",
        "runner_arch": "X64",
        "python_version": "3.12.10",
        "cpu_count": 16,
        "memory_bytes": 64 * 1024**3,
        "disk_free_bytes": 100 * 1024**3,
        "gpu_count": 1,
        "cuda_visible_device_count": 1,
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "gpu_uuid_sha256": SHA,
        "driver_version": "570.124.06",
        "cuda_runtime": "12.8",
        "torch_version": "2.11.0+cu128",
        "torchvision_version": "0.26.0+cu128",
        "network_hosts": list(preflight.NETWORK_HOSTS),
        "workspace_clean": True,
        "container_isolated": True,
        "private_mounts_detected": False,
        "ephemeral_declared": True,
        "weight_accessed": False,
        "release_eligible": True,
        "claim_allowed": False,
        "checked_at": "2026-07-12",
    }
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot open release"):
        RunnerQualificationManifestV1.from_mapping(json.loads(path.read_text()))


def test_mount_path_decoder_handles_kernel_escapes() -> None:
    assert preflight._decode_mount_path("/workspace/private\\040repo") == "/workspace/private repo"


def test_network_preflight_enforces_tls12_or_newer(monkeypatch: pytest.MonkeyPatch) -> None:
    class Connection:
        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Context:
        minimum_version: object | None = None

        def wrap_socket(self, _raw: object, *, server_hostname: str) -> Connection:
            assert server_hostname in preflight.NETWORK_HOSTS
            return Connection()

    context = Context()
    monkeypatch.setattr(preflight.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(preflight.socket, "create_connection", lambda *_args, **_kwargs: Connection())

    assert preflight._verify_network() == preflight.NETWORK_HOSTS
    assert context.minimum_version is preflight.ssl.TLSVersion.TLSv1_2
