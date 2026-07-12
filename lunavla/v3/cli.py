from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import yaml

from .artifacts import verify_run_directory
from .config import ExperimentConfig
from .diagnostic_workflow import (
    run_diagnostic,
    verify_diagnostic_output,
    write_diagnostic_report,
)
from .engine import dataset_for_config, run_alpha
from .migration import migrate_v2_mapping
from .integration_workflow import (
    print_source_preflight,
    run_integration,
    verify_integration,
)
from .stable_contracts import (
    StableEvidenceDesignV1,
    StableEvidenceRowV1,
    StableEvidenceSummaryV1,
    StableRepeatSentinelV1,
    StableReleaseCandidateV1,
    validate_stable_design_set,
    verify_stable_evidence_bundle,
)


def _json_mapping(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document must contain a mapping: {path}")
    return value


def _json_rows(path: str | Path) -> tuple[StableEvidenceRowV1, ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"JSON document must contain a row list: {path}")
    return tuple(StableEvidenceRowV1.from_mapping(item) for item in value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lunavla-v3")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config")
    migrate = subparsers.add_parser("migrate-config")
    migrate.add_argument("source")
    migrate.add_argument("--out", required=True)
    audit = subparsers.add_parser("data-audit")
    audit.add_argument("config")
    audit.add_argument("--out", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("config")
    replay.add_argument("--episode", type=int, default=0)
    replay.add_argument("--out", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("config")
    run.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify-run")
    verify.add_argument("run_dir")
    diagnostic_run = subparsers.add_parser("diagnostic-run")
    diagnostic_run.add_argument("design")
    diagnostic_run.add_argument("--overwrite", action="store_true")
    diagnostic_verify = subparsers.add_parser("diagnostic-verify")
    diagnostic_verify.add_argument("output_root")
    diagnostic_report = subparsers.add_parser("diagnostic-report")
    diagnostic_report.add_argument("output_root")
    diagnostic_report.add_argument("--out", required=True)
    source_preflight = subparsers.add_parser("source-preflight")
    source_preflight.add_argument("config")
    integration_run = subparsers.add_parser("integration-run")
    integration_run.add_argument("config")
    integration_run.add_argument("--cache-dir", required=True)
    integration_run.add_argument("--out", required=True)
    integration_verify = subparsers.add_parser("integration-verify")
    integration_verify.add_argument("output_root")
    stable_designs = subparsers.add_parser("validate-stable-designs")
    stable_designs.add_argument("designs", nargs="+")
    stable_evidence = subparsers.add_parser("verify-stable-evidence")
    stable_evidence.add_argument("design")
    stable_evidence.add_argument("rows")
    stable_evidence.add_argument("sentinel")
    stable_evidence.add_argument("summary")
    stable_candidate = subparsers.add_parser("verify-stable-candidate")
    stable_candidate.add_argument("candidate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate-config":
        config = ExperimentConfig.load(arguments.config)
        print(json.dumps({"valid": True, "config_sha256": config.sha256()}))
        return 0
    if arguments.command == "migrate-config":
        source = yaml.safe_load(Path(arguments.source).read_text(encoding="utf-8-sig"))
        migrated = migrate_v2_mapping(source)
        ExperimentConfig.from_mapping(migrated)
        Path(arguments.out).write_text(yaml.safe_dump(migrated, sort_keys=False), encoding="utf-8")
        return 0
    if arguments.command == "data-audit":
        config = ExperimentConfig.load(arguments.config)
        dataset_for_config(config).audit.save(arguments.out)
        return 0
    if arguments.command == "replay":
        config = ExperimentConfig.load(arguments.config)
        episodes = dataset_for_config(config).episodes
        if arguments.episode < 0 or arguments.episode >= len(episodes):
            raise IndexError("episode index is out of range")
        target = Path(arguments.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(episodes[arguments.episode].to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    if arguments.command == "run":
        result = run_alpha(ExperimentConfig.load(arguments.config), overwrite=arguments.overwrite)
        print(json.dumps({"output_dir": str(result.output_dir), "metrics": result.metrics}, sort_keys=True))
        return 0
    if arguments.command == "verify-run":
        manifest = verify_run_directory(arguments.run_dir)
        print(json.dumps({"valid": True, "git_sha": manifest["git_sha"]}, sort_keys=True))
        return 0
    if arguments.command == "diagnostic-run":
        output = run_diagnostic(arguments.design, overwrite=arguments.overwrite)
        print(json.dumps({"output_dir": str(output), "claim_allowed": False}, sort_keys=True))
        return 0
    if arguments.command == "diagnostic-verify":
        evidence = verify_diagnostic_output(arguments.output_root)
        print(json.dumps({"valid": True, "claim_allowed": evidence["claim_allowed"]}, sort_keys=True))
        return 0
    if arguments.command == "diagnostic-report":
        output = write_diagnostic_report(arguments.output_root, arguments.out)
        print(json.dumps({"report_dir": str(output)}, sort_keys=True))
        return 0
    if arguments.command == "source-preflight":
        print(json.dumps(print_source_preflight(arguments.config), sort_keys=True))
        return 0
    if arguments.command == "integration-run":
        output = run_integration(
            arguments.config,
            cache_dir=arguments.cache_dir,
            output_root=arguments.out,
        )
        print(json.dumps({"output_dir": str(output), "claim_allowed": False}, sort_keys=True))
        return 0
    if arguments.command == "integration-verify":
        integration_manifest = verify_integration(arguments.output_root)
        print(
            json.dumps(
                {
                    "valid": True,
                    "git_sha": integration_manifest.git_sha,
                    "claim_allowed": integration_manifest.claim_allowed,
                    "benchmark_claim": integration_manifest.benchmark_claim,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "validate-stable-designs":
        designs = tuple(StableEvidenceDesignV1.load(path) for path in arguments.designs)
        rows = validate_stable_design_set(designs)
        print(json.dumps({"valid": True, "rows": rows, "total_rows": sum(rows.values())}, sort_keys=True))
        return 0
    if arguments.command == "verify-stable-evidence":
        design = StableEvidenceDesignV1.load(arguments.design)
        row_records = _json_rows(arguments.rows)
        sentinel = StableRepeatSentinelV1.from_mapping(_json_mapping(arguments.sentinel))
        summary = StableEvidenceSummaryV1.from_mapping(_json_mapping(arguments.summary))
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
    if arguments.command == "verify-stable-candidate":
        candidate = StableReleaseCandidateV1.from_mapping(_json_mapping(arguments.candidate))
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
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
