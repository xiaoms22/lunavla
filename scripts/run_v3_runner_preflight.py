from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import ssl
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from lunavla.v3 import RunnerQualificationManifestV1
from lunavla.v3.artifacts import sha256_file


ROOT = Path(__file__).resolve().parents[1]
GPU_LOCK = ROOT / "requirements-v3-smolvla-gpu-cu128.lock"
NETWORK_HOSTS = (
    "api.github.com",
    "download.pytorch.org",
    "github.com",
    "huggingface.co",
    "pypi.org",
)
REQUIRED_LABELS = ("self-hosted", "linux", "x64", "gpu", "lunavla-v3")
PRIVATE_MOUNT_PREFIXES = (
    "/Users",
    "/data",
    "/home",
    "/mnt",
    "/root",
    "/srv",
    "/workspace",
)
ALLOWED_SYSTEM_MOUNTS = {"/etc/hostname", "/etc/hosts", "/etc/resolv.conf"}


def _json(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("runner qualification manifest must contain a JSON object")
    return value


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _require_clean_git(expected_sha: str) -> None:
    if _git("rev-parse", "--verify", "HEAD^{commit}") != expected_sha:
        raise RuntimeError("runner preflight checkout does not match expected Git SHA")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("runner preflight requires a clean checkout")


def _gpu_identity() -> tuple[str, str, str]:
    lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("runner preflight requires exactly one nvidia-smi GPU")
    values = [item.strip() for item in lines[0].split(",")]
    if len(values) != 3 or any(not item for item in values):
        raise RuntimeError("nvidia-smi returned malformed GPU identity")
    return values[0], hashlib.sha256(values[1].encode("utf-8")).hexdigest(), values[2]


def _memory_bytes() -> int:
    host = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    limits: list[int] = [host]
    for candidate in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value.isdigit():
            limit = int(value)
            if 0 < limit < 2**60:
                limits.append(limit)
    return min(limits)


def _os_version() -> str:
    values: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value.strip().strip('"')
    return values.get("PRETTY_NAME") or values.get("VERSION_ID") or "unknown"


def _container_isolated() -> bool:
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def _decode_mount_path(value: str) -> str:
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n")


def _private_mounts() -> tuple[str, ...]:
    findings: set[str] = set()
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            raise RuntimeError("malformed /proc/self/mountinfo row")
        mount_point = _decode_mount_path(fields[4])
        if mount_point in ALLOWED_SYSTEM_MOUNTS:
            continue
        if any(mount_point == prefix or mount_point.startswith(prefix + "/") for prefix in PRIVATE_MOUNT_PREFIXES):
            findings.add(mount_point)
    return tuple(sorted(findings))


def _verify_network() -> tuple[str, ...]:
    context = ssl.create_default_context()
    verified: list[str] = []
    for host in NETWORK_HOSTS:
        with socket.create_connection((host, 443), timeout=10) as raw:
            with context.wrap_socket(raw, server_hostname=host):
                verified.append(host)
    return tuple(verified)


def _weight_accessed() -> bool:
    candidates = {
        Path.home() / ".cache/huggingface",
        *(Path(value).expanduser() for key in ("HF_HOME", "HF_HUB_CACHE") if (value := os.environ.get(key))),
    }
    for root in candidates:
        if root.exists() and any(root.rglob("model.safetensors")):
            return True
    return False


def qualify(
    *,
    role: str,
    expected_git_sha: str,
    container_image_sha256: str,
    runner_name: str,
    output_path: str | Path,
) -> RunnerQualificationManifestV1:
    _require_clean_git(expected_git_sha)
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("runner preflight requires Linux x86_64")
    if not _container_isolated():
        raise RuntimeError("runner preflight must execute inside an isolated container")
    if os.environ.get("LUNAVLA_EPHEMERAL_RUNNER") != "true":
        raise RuntimeError("runner launch must declare LUNAVLA_EPHEMERAL_RUNNER=true")
    if os.environ.get("LUNAVLA_CONTAINER_IMAGE_SHA256") != container_image_sha256:
        raise RuntimeError("runner launch image digest does not match the reviewed request")
    labels = tuple(item.strip() for item in os.environ.get("RUNNER_LABELS", "").split(",") if item.strip())
    if not set(REQUIRED_LABELS).issubset(labels):
        raise RuntimeError("runner does not expose all required LunaVLA labels")
    mounts = _private_mounts()
    if mounts:
        raise RuntimeError("runner container exposes a forbidden private mount target")
    if _weight_accessed():
        raise RuntimeError("runner preflight found model weights in a Hugging Face cache")

    import torch
    import torchvision

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("runner preflight requires exactly one visible CUDA device")
    gpu_name, gpu_uuid_sha256, driver_version = _gpu_identity()
    runner_temp = Path(os.environ.get("RUNNER_TEMP", "/tmp")).resolve()
    if not runner_temp.is_dir():
        raise RuntimeError("RUNNER_TEMP must name an existing directory")
    manifest = RunnerQualificationManifestV1(
        role=role,
        git_sha=expected_git_sha,
        dependency_lock_sha256=sha256_file(GPU_LOCK),
        container_image_sha256=container_image_sha256,
        runner_name_sha256=hashlib.sha256(runner_name.encode("utf-8")).hexdigest(),
        runner_labels=labels,
        runner_os="Linux",
        runner_os_version=_os_version(),
        runner_arch="X64",
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 0,
        memory_bytes=_memory_bytes(),
        disk_free_bytes=shutil.disk_usage(runner_temp).free,
        gpu_count=1,
        cuda_visible_device_count=torch.cuda.device_count(),
        gpu_name=gpu_name,
        gpu_uuid_sha256=gpu_uuid_sha256,
        driver_version=driver_version,
        cuda_runtime=str(torch.version.cuda),
        torch_version=str(torch.__version__),
        torchvision_version=str(torchvision.__version__),
        network_hosts=_verify_network(),
        workspace_clean=True,
        container_isolated=True,
        private_mounts_detected=False,
        ephemeral_declared=True,
        weight_accessed=False,
        release_eligible=False,
        claim_allowed=False,
        checked_at=datetime.now(UTC).date().isoformat(),
    )
    manifest.save(output_path)
    return manifest


def verify(path: str | Path) -> RunnerQualificationManifestV1:
    return RunnerQualificationManifestV1.from_mapping(_json(path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_v3_runner_preflight.py")
    commands = parser.add_subparsers(dest="command", required=True)
    qualify_parser = commands.add_parser("qualify")
    qualify_parser.add_argument("--role", choices=("authoritative", "secondary"), required=True)
    qualify_parser.add_argument("--expected-git-sha", required=True)
    qualify_parser.add_argument("--container-image-sha256", required=True)
    qualify_parser.add_argument("--runner-name", required=True)
    qualify_parser.add_argument("--out", required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "qualify":
        manifest = qualify(
            role=arguments.role,
            expected_git_sha=arguments.expected_git_sha,
            container_image_sha256=arguments.container_image_sha256,
            runner_name=arguments.runner_name,
            output_path=arguments.out,
        )
        print(json.dumps({"qualified": True, "manifest_sha256": manifest.sha256()}))
        return 0
    manifest = verify(arguments.manifest)
    print(json.dumps({"qualified": True, "manifest_sha256": manifest.sha256()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
