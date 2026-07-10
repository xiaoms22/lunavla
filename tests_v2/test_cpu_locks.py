from __future__ import annotations

from pathlib import Path

from scripts.lock_v2_cpu import forbidden_cpu_packages


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_package_guard_covers_cuda_nccl_nvidia_and_triton() -> None:
    names = {
        "numpy",
        "cuda-bindings",
        "cuda_toolkit",
        "nvidia-cublas-cu12",
        "torch-nccl",
        "nccl",
        "triton",
    }
    assert forbidden_cpu_packages(names) == {
        "cuda-bindings",
        "cuda-toolkit",
        "nvidia-cublas-cu12",
        "torch-nccl",
        "nccl",
        "triton",
    }


def test_real_pusht_profile_pins_the_compatible_pymunk_major() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "requirements-v2-integration-cpu.lock").read_text(
        encoding="utf-8"
    )
    assert '"lerobot[dataset,pusht]==0.6.*"' in project
    assert "pymunk==6.11.1" in lock
    assert "pymunk==7." not in lock
