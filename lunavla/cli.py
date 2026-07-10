"""Command-line entry point for v2 configuration and engine tooling."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Sequence

from lunavla.config import ExperimentConfig
from lunavla.manifest import RunManifest
from lunavla.migration import migrate_v11_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lunavla-v2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate and resolve a v2 config")
    validate.add_argument("config", type=Path)

    migrate = subparsers.add_parser("migrate-config", help="migrate a v1.1 config to v2")
    migrate.add_argument("source", type=Path)
    migrate.add_argument("destination", type=Path)
    migrate.add_argument("--overwrite", action="store_true")

    train = subparsers.add_parser("train", help="run a versioned v2 train/evaluate loop")
    train.add_argument("config", type=Path)
    train.add_argument("--overwrite", action="store_true")
    train.add_argument("--require-device", choices=("cpu", "mps", "cuda"))
    train.add_argument("--require-clean", action="store_true")
    train.add_argument("--root", type=Path)

    verify = subparsers.add_parser("verify-run", help="verify a v2 run manifest and hashes")
    verify.add_argument("run_dir", type=Path)
    return parser


def _repository_root(config_path: Path, explicit: Path | None) -> Path:
    config_path = config_path.resolve()
    if explicit is None:
        result = subprocess.run(
            ["git", "-C", str(config_path.parent), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError("config is not inside a Git checkout; pass --root explicitly")
        root = Path(result.stdout.strip()).resolve()
    else:
        root = explicit.resolve()
    if not (root / ".git").exists():
        raise ValueError("--root must identify a Git working tree")
    project_file = root / "pyproject.toml"
    try:
        project = tomllib.loads(project_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("repository root must contain a valid LunaVLA pyproject.toml") from exc
    if project.get("project", {}).get("name") != "lunavla":
        raise ValueError("repository root is not a LunaVLA checkout")
    if config_path != root and root not in config_path.parents:
        raise ValueError("training config must be located inside the selected LunaVLA checkout")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    if args.command == "validate-config":
        config = ExperimentConfig.load(args.config)
        print(json.dumps({"schema_version": 2, "config_sha256": config.sha256()}))
        return 0
    if args.command == "migrate-config":
        config = migrate_v11_file(args.source, args.destination, overwrite=args.overwrite)
        print(json.dumps({"destination": str(args.destination), "config_sha256": config.sha256()}))
        return 0
    if args.command == "train":
        from lunavla.run import run_experiment

        config = ExperimentConfig.load(args.config)
        root = _repository_root(args.config, args.root)
        manifest = run_experiment(
            config,
            root=root,
            overwrite=args.overwrite,
            require_device=args.require_device,
            require_clean=args.require_clean,
            command=["lunavla-v2", *raw_argv],
        )
        print(
            json.dumps(
                {
                    "run_id": manifest.run_id,
                    "config_sha256": manifest.config_sha256,
                    "checkpoint_sha256": manifest.checkpoint_sha256,
                    "metrics": manifest.metrics,
                }
            )
        )
        return 0
    if args.command == "verify-run":
        manifest = RunManifest.verify_run_dir(args.run_dir)
        print(json.dumps({"run_id": manifest.run_id, "verified": True}))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
