from __future__ import annotations

from scripts.lock_v2_cpu import forbidden_cpu_packages


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
