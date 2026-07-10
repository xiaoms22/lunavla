#!/usr/bin/env python3
"""Generate or verify the hash-locked Linux CPU dependency profiles."""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UV_VERSION = "0.11.26"
PYTHON_VERSION = "3.12"
LINUX_PLATFORM = "x86_64-manylinux_2_28"
FORBIDDEN_PACKAGE = re.compile(r"^(?:nvidia(?:[-_].*)?|triton)==", re.MULTILINE | re.IGNORECASE)


@dataclass(frozen=True)
class LockSpec:
    path: Path
    extras: tuple[str, ...]


LOCKS = (
    LockSpec(ROOT / "requirements-v2-core-cpu.lock", ("dev", "v2-core")),
    LockSpec(ROOT / "requirements-v2-cpu.lock", ("dev", "v2")),
    LockSpec(ROOT / "requirements-v2-release-cpu.lock", ("dev", "v2", "release")),
)


def _uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        raise RuntimeError(f"uv {UV_VERSION} is required but was not found")
    actual = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parts = actual.split()
    if len(parts) < 2 or parts[:2] != ["uv", UV_VERSION]:
        raise RuntimeError(f"expected uv {UV_VERSION}, found {actual!r}")
    return executable


def _compile(spec: LockSpec, destination: Path, *, constrain: bool) -> None:
    command = [
        _uv(),
        "pip",
        "compile",
        "pyproject.toml",
        "--python-version",
        PYTHON_VERSION,
        "--python-platform",
        LINUX_PLATFORM,
        "--torch-backend",
        "cpu",
        "--generate-hashes",
        "--no-annotate",
        "--only-binary",
        ":all:",
        "--output-file",
        str(destination),
        "--custom-compile-command",
        "python scripts/lock_v2_cpu.py --write",
        "--quiet",
    ]
    for extra in spec.extras:
        command.extend(("--extra", extra))
    if constrain:
        command.extend(("--constraints", str(spec.path)))
    subprocess.run(command, cwd=ROOT, check=True)


def _validate_lock(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    if FORBIDDEN_PACKAGE.search(content):
        raise RuntimeError(f"{path.name} contains a CUDA-only dependency")
    expected = {
        "numpy": "2.2.6",
        "torch": "2.11.0+cpu",
        "torchvision": "0.26.0+cpu",
    }
    for package, version in expected.items():
        if not re.search(rf"^{re.escape(package)}=={re.escape(version)}(?: \\)?$", content, re.MULTILINE):
            raise RuntimeError(f"{path.name} must pin {package}=={version}")


def _write() -> int:
    for spec in LOCKS:
        with tempfile.TemporaryDirectory(prefix="lunavla-cpu-lock-") as temp_dir:
            generated = Path(temp_dir) / spec.path.name
            _compile(spec, generated, constrain=False)
            _validate_lock(generated)
            spec.path.write_bytes(generated.read_bytes())
            print(f"wrote {spec.path.relative_to(ROOT)}")
    return 0


def _check() -> int:
    failed = False
    for spec in LOCKS:
        if not spec.path.is_file():
            print(f"missing {spec.path.relative_to(ROOT)}", file=sys.stderr)
            failed = True
            continue
        with tempfile.TemporaryDirectory(prefix="lunavla-cpu-lock-") as temp_dir:
            generated = Path(temp_dir) / spec.path.name
            _compile(spec, generated, constrain=True)
            _validate_lock(generated)
            expected = spec.path.read_text(encoding="utf-8").splitlines(keepends=True)
            actual = generated.read_text(encoding="utf-8").splitlines(keepends=True)
            if expected != actual:
                failed = True
                diff = difflib.unified_diff(
                    expected,
                    actual,
                    fromfile=str(spec.path.relative_to(ROOT)),
                    tofile="regenerated",
                )
                sys.stderr.writelines(diff)
            else:
                print(f"fresh {spec.path.relative_to(ROOT)}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify committed locks")
    mode.add_argument("--write", action="store_true", help="regenerate committed locks")
    args = parser.parse_args()
    try:
        return _check() if args.check else _write()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"CPU lock error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
