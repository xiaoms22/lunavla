from __future__ import annotations

import argparse
import importlib
import json
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from lunavla.v3.stable_contracts import (
    StableEvidenceDesignV1,
    StableEvidenceRowV1,
    StableEvidenceSummaryV1,
    StableRepeatSentinelV1,
    RC_PACKAGE_VERSION,
    STABLE_PACKAGE_VERSION,
    release_candidate_from_mapping,
    validate_stable_design_set,
    verify_release_candidate_assets,
    verify_stable_evidence_bundle,
)
from lunavla.v3.stable_executor import TeachingFixtureStableExecutor
from lunavla.v3.stable_workflow import run_stable_study, verify_stable_study


def _mapping(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON document must contain a mapping: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_v3_stable_release")
    commands = parser.add_subparsers(dest="command", required=True)
    designs = commands.add_parser("validate-designs")
    designs.add_argument("designs", nargs="+")
    summary = commands.add_parser("verify-evidence-summary")
    summary.add_argument("design")
    summary.add_argument("rows")
    summary.add_argument("sentinel")
    summary.add_argument("summary")
    run_study = commands.add_parser("run-study")
    run_study.add_argument("design")
    run_study.add_argument("--out", required=True)
    run_study.add_argument("--overwrite", action="store_true")
    verify_study = commands.add_parser("verify-study")
    verify_study.add_argument("output_dir")
    candidate = commands.add_parser("verify-candidate")
    candidate.add_argument("candidate")
    candidate.add_argument("--asset-root")
    version = commands.add_parser("verify-version")
    version.add_argument("stage", choices=("rc", "stable"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate-designs":
        designs = tuple(StableEvidenceDesignV1.load(path) for path in arguments.designs)
        rows = validate_stable_design_set(designs)
        print(json.dumps({"valid": True, "rows": rows, "total_rows": sum(rows.values())}, sort_keys=True))
        return 0
    if arguments.command == "verify-evidence-summary":
        design = StableEvidenceDesignV1.load(arguments.design)
        row_values = json.loads(Path(arguments.rows).read_text(encoding="utf-8"))
        if not isinstance(row_values, list):
            raise TypeError("stable evidence rows JSON must contain a list")
        row_records = tuple(StableEvidenceRowV1.from_mapping(item) for item in row_values)
        sentinel = StableRepeatSentinelV1.from_mapping(_mapping(arguments.sentinel))
        summary = StableEvidenceSummaryV1.from_mapping(_mapping(arguments.summary))
        verify_stable_evidence_bundle(design, row_records, sentinel, summary)
        print(
            json.dumps(
                {
                    "valid": True,
                    "study_id": summary.study_id,
                    "release_eligible": summary.release_eligible,
                    "gate_reasons": list(summary.gate_reasons),
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "run-study":
        output = run_stable_study(
            arguments.design,
            arguments.out,
            TeachingFixtureStableExecutor(),
            overwrite=arguments.overwrite,
        )
        summary = verify_stable_study(output)
        print(
            json.dumps(
                {
                    "output_dir": str(output),
                    "study_id": summary.study_id,
                    "rows": summary.observed_rows,
                    "release_eligible": summary.release_eligible,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "verify-study":
        summary = verify_stable_study(arguments.output_dir)
        print(
            json.dumps(
                {
                    "valid": True,
                    "study_id": summary.study_id,
                    "rows": summary.observed_rows,
                    "release_eligible": summary.release_eligible,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "verify-candidate":
        candidate = release_candidate_from_mapping(_mapping(arguments.candidate))
        if arguments.asset_root is not None:
            verify_release_candidate_assets(candidate, arguments.asset_root)
        print(
            json.dumps(
                {
                    "valid": True,
                    "tag": candidate.expected_tag,
                    "git_sha": candidate.git_sha,
                    "pypi_published": candidate.pypi_published,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "verify-version":
        expected = RC_PACKAGE_VERSION if arguments.stage == "rc" else STABLE_PACKAGE_VERSION
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        public = getattr(importlib.import_module("lunavla"), "__version__", None)
        if project != expected or public != expected:
            raise RuntimeError(
                f"{arguments.stage} package version sources must both equal {expected}"
            )
        print(json.dumps({"valid": True, "stage": arguments.stage, "version": expected}, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
