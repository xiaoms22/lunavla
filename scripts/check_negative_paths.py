from __future__ import annotations

import tempfile
from pathlib import Path

import build_submission_pack
import validate_configs


def fail(message: str) -> None:
    raise SystemExit(f"negative path check failed: {message}")


def require_error(errors: list[str], expected: str, label: str) -> None:
    if not any(expected in error for error in errors):
        fail(f"{label} did not report `{expected}`; got {errors}")


def check_malformed_config(tmp_dir: Path) -> None:
    config_path = tmp_dir / "malformed.yaml"
    config_path.write_text("model: [\n", encoding="utf-8")
    errors = validate_configs.validate_config(config_path)
    require_error(errors, "could not be parsed", "malformed config")


def check_mismatched_chunk_config(tmp_dir: Path) -> None:
    config_path = tmp_dir / "mismatched_chunk.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  observation_dim: 4",
                "  instruction_dim: 8",
                "  action_dim: 2",
                "  chunk_size: 4",
                "project_name: negative_path_check",
                "framework: minimind-vla",
                "policy:",
                "  name: act",
                "  chunk_size: 2",
                "task: pusht",
                "dataset:",
                "  source: mock_pusht",
                "  num_episodes: 1",
                "  steps_per_episode: 2",
                "  seed: 1",
                "training:",
                "  device: cpu",
                "  batch_size: 1",
                "  num_steps: 1",
                "  learning_rate: 0.001",
                "  seed: 0",
                "  log_interval: 1",
                "eval:",
                "  episodes: 1",
                "  rollout_steps: 2",
                "  success_distance: 0.1",
                "artifacts:",
                "  output_dir: outputs/negative_path_check",
                "  checkpoint_name: checkpoint.pt",
                "  report_path: outputs/negative_path_check/report.md",
                "",
            ]
        ),
        encoding="utf-8",
    )
    errors = validate_configs.validate_config(config_path)
    require_error(errors, "`model.chunk_size` and `policy.chunk_size` must match", "mismatched chunk config")


def check_submission_pack_missing_source(tmp_dir: Path) -> None:
    missing_source = tmp_dir / "missing_report.md"
    pack_target = tmp_dir / "pack" / "missing_report.md"
    try:
        build_submission_pack.copy_file(missing_source, pack_target)
    except FileNotFoundError as exc:
        if "Missing submission source" not in str(exc):
            fail(f"submission pack raised an unclear error: {exc}")
        return
    fail("submission pack accepted a missing source file")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="minimind_vla_negative_") as tmp:
        tmp_dir = Path(tmp)
        checks = [
            check_malformed_config,
            check_mismatched_chunk_config,
            check_submission_pack_missing_source,
        ]
        for check in checks:
            check(tmp_dir)
            print(f"[ok] {check.__name__}")
    print("negative path checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
