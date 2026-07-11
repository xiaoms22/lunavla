#!/usr/bin/env python3
"""Run or verify the pinned LunaVLA v2 LeRobot/PushT integration smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lunavla.lerobot_integration import IntegrationManifest, run_official_integration


ROOT = Path(__file__).resolve().parents[1]


def _run(args: argparse.Namespace) -> int:
    manifest = run_official_integration(
        root=ROOT,
        expected_git_sha=args.expected_git_sha,
        cache_dir=args.cache_dir,
        output_path=args.output,
    )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    manifest = IntegrationManifest.load(args.manifest)
    if manifest.git_sha != args.expected_git_sha:
        raise ValueError(
            f"manifest git_sha {manifest.git_sha} does not match {args.expected_git_sha}"
        )
    print(f"verified {args.manifest} for {manifest.git_sha}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the networked integration smoke")
    run_parser.add_argument(
        "--expected-git-sha",
        required=True,
        help="immutable 40-character SHA checked out by the workflow",
    )
    run_parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="temporary LeRobot cache outside the source checkout",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new integration_manifest.json path outside the source checkout",
    )
    run_parser.set_defaults(handler=_run)

    verify_parser = subparsers.add_parser("verify", help="verify a generated manifest")
    verify_parser.add_argument(
        "--expected-git-sha",
        required=True,
        help="immutable 40-character source SHA",
    )
    verify_parser.add_argument("manifest", type=Path)
    verify_parser.set_defaults(handler=_verify)

    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"LeRobot integration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
