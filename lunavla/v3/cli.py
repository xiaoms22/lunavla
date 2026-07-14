from __future__ import annotations

import argparse
import hashlib
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
from .profiling import run_profile, verify_profile
from .portfolio import build_portfolio, verify_portfolio
from .stable_contracts import (
    StableEvidenceDesignV1,
    StableEvidenceRowV1,
    StableEvidenceSummaryV1,
    StableRepeatSentinelV1,
    StableReleaseCandidateV1,
    release_candidate_from_mapping,
    validate_stable_design_set,
    verify_release_candidate_assets,
    verify_stable_evidence_bundle,
)
from .stable_executor import TeachingFixtureStableExecutor
from .stable_workflow import run_stable_study, verify_stable_study
from .v31_contracts import VLMBackendSpecV1
from .v31_evidence import V31EvidenceDesignV1
from .v31_tasks import make_v31_task_dataset
from .v31_vlm import (
    DeterministicFixtureExtractor,
    TransformersFrozenExtractor,
    build_frozen_feature_cache,
    preflight_local_model,
    run_qwen_observational_smoke,
    verify_frozen_feature_cache,
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
    profile_run = subparsers.add_parser("profile-run")
    profile_run.add_argument("design")
    profile_run.add_argument("--overwrite", action="store_true")
    profile_verify = subparsers.add_parser("profile-verify")
    profile_verify.add_argument("output_root")
    portfolio_build = subparsers.add_parser("portfolio-build")
    portfolio_build.add_argument("evidence_root")
    portfolio_build.add_argument("--out", required=True)
    portfolio_build.add_argument("--overwrite", action="store_true")
    portfolio_verify = subparsers.add_parser("portfolio-verify")
    portfolio_verify.add_argument("output_root")
    stable_designs = subparsers.add_parser("validate-stable-designs")
    stable_designs.add_argument("designs", nargs="+")
    stable_evidence = subparsers.add_parser("verify-stable-evidence")
    stable_evidence.add_argument("design")
    stable_evidence.add_argument("rows")
    stable_evidence.add_argument("sentinel")
    stable_evidence.add_argument("summary")
    stable_run = subparsers.add_parser("stable-run")
    stable_run.add_argument("design")
    stable_run.add_argument("--out", required=True)
    stable_run.add_argument("--overwrite", action="store_true")
    stable_verify = subparsers.add_parser("stable-verify")
    stable_verify.add_argument("output_dir")
    stable_candidate = subparsers.add_parser("verify-stable-candidate")
    stable_candidate.add_argument("candidate")
    release_candidate = subparsers.add_parser("verify-release-candidate")
    release_candidate.add_argument("candidate")
    release_candidate.add_argument("--asset-root")
    vlm_preflight = subparsers.add_parser("vlm-preflight")
    vlm_preflight.add_argument("spec")
    vlm_preflight.add_argument("--model-root", required=True)
    vlm_cache = subparsers.add_parser("vlm-cache")
    vlm_cache.add_argument("spec")
    vlm_cache.add_argument("--model-root", required=True)
    vlm_cache.add_argument("--out", required=True)
    vlm_cache.add_argument("--processor-sha256", required=True)
    vlm_cache.add_argument("--device-environment-sha256", required=True)
    vlm_cache.add_argument("--overwrite", action="store_true")
    vlm_cache_verify = subparsers.add_parser("vlm-cache-verify")
    vlm_cache_verify.add_argument("cache_root")
    qwen_smoke = subparsers.add_parser("qwen-observational-smoke")
    qwen_smoke.add_argument("spec")
    qwen_smoke.add_argument("--model-root", required=True)
    qwen_smoke.add_argument("--out", required=True)
    fixture_run = subparsers.add_parser("v31-fixture-run")
    fixture_run.add_argument("config")
    fixture_run.add_argument("--overwrite", action="store_true")
    evidence_design = subparsers.add_parser("validate-v31-evidence-design")
    evidence_design.add_argument("design")
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
        target.write_text(
            json.dumps(episodes[arguments.episode].to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    if arguments.command == "run":
        result = run_alpha(ExperimentConfig.load(arguments.config), overwrite=arguments.overwrite)
        print(
            json.dumps(
                {"output_dir": str(result.output_dir), "metrics": result.metrics}, sort_keys=True
            )
        )
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
        print(
            json.dumps({"valid": True, "claim_allowed": evidence["claim_allowed"]}, sort_keys=True)
        )
        return 0
    if arguments.command == "diagnostic-report":
        output = write_diagnostic_report(arguments.output_root, arguments.out)
        print(json.dumps({"report_dir": str(output)}, sort_keys=True))
        return 0
    if arguments.command == "profile-run":
        output = run_profile(arguments.design, overwrite=arguments.overwrite)
        print(json.dumps({"output_dir": str(output), "comparative_claim_allowed": False}))
        return 0
    if arguments.command == "profile-verify":
        profile_manifest = verify_profile(arguments.output_root)
        print(
            json.dumps(
                {
                    "valid": True,
                    "policy_id": profile_manifest.policy_id,
                    "release_eligible": profile_manifest.release_eligible,
                    "comparative_claim_allowed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "portfolio-build":
        output = build_portfolio(
            arguments.evidence_root,
            arguments.out,
            overwrite=arguments.overwrite,
        )
        portfolio_manifest = verify_portfolio(output)
        print(
            json.dumps(
                {
                    "output_dir": str(output),
                    "release_eligible": portfolio_manifest.release_eligible,
                    "total_rows": portfolio_manifest.total_rows,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "portfolio-verify":
        portfolio_manifest = verify_portfolio(arguments.output_root)
        print(
            json.dumps(
                {
                    "valid": True,
                    "git_sha": portfolio_manifest.git_sha,
                    "release_eligible": portfolio_manifest.release_eligible,
                    "total_rows": portfolio_manifest.total_rows,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "validate-stable-designs":
        designs = tuple(StableEvidenceDesignV1.load(path) for path in arguments.designs)
        rows = validate_stable_design_set(designs)
        print(
            json.dumps(
                {"valid": True, "rows": rows, "total_rows": sum(rows.values())}, sort_keys=True
            )
        )
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
    if arguments.command == "stable-run":
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
    if arguments.command == "stable-verify":
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
    if arguments.command == "verify-release-candidate":
        candidate = release_candidate_from_mapping(_json_mapping(arguments.candidate))
        if arguments.asset_root is not None:
            verify_release_candidate_assets(candidate, arguments.asset_root)
        print(
            json.dumps(
                {
                    "valid": True,
                    "tag": candidate.expected_tag,
                    "git_sha": candidate.git_sha,
                    "assets_verified": arguments.asset_root is not None,
                    "pypi_published": candidate.pypi_published,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "vlm-preflight":
        spec = VLMBackendSpecV1.from_mapping(_json_mapping(arguments.spec))
        preflight = preflight_local_model(spec, arguments.model_root)
        print(json.dumps(preflight.to_dict(), sort_keys=True))
        return 0
    if arguments.command == "vlm-cache":
        spec = VLMBackendSpecV1.from_mapping(_json_mapping(arguments.spec))
        extractor = TransformersFrozenExtractor(spec, arguments.model_root)
        output = build_frozen_feature_cache(
            make_v31_task_dataset(),
            spec,
            extractor,
            Path(arguments.out).resolve(),
            processor_sha256=arguments.processor_sha256,
            device_environment_sha256=arguments.device_environment_sha256,
            overwrite=arguments.overwrite,
        )
        index = verify_frozen_feature_cache(output)
        print(
            json.dumps(
                {"output_dir": str(output), "cache_index_sha256": index.sha256()}, sort_keys=True
            )
        )
        return 0
    if arguments.command == "vlm-cache-verify":
        index = verify_frozen_feature_cache(arguments.cache_root)
        print(json.dumps({"valid": True, "cache_index_sha256": index.sha256()}, sort_keys=True))
        return 0
    if arguments.command == "qwen-observational-smoke":
        spec = VLMBackendSpecV1.from_mapping(_json_mapping(arguments.spec))
        extractor = TransformersFrozenExtractor(spec, arguments.model_root)
        smoke = run_qwen_observational_smoke(make_v31_task_dataset(), spec, extractor)
        output = Path(arguments.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(smoke.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(smoke.to_dict(), sort_keys=True))
        return 0
    if arguments.command == "v31-fixture-run":
        config = ExperimentConfig.load(arguments.config)
        if config.task["id"] != "synthetic_vlm_suite":
            raise ValueError("v31-fixture-run requires synthetic_vlm_suite")
        dataset = make_v31_task_dataset(
            data_seed=config.dataset["seed"],
            train_per_task=int(config.dataset["parameters"]["train_per_task"]),
            held_out_per_cell=int(config.dataset["parameters"]["held_out_per_cell"]),
        )
        spec = VLMBackendSpecV1.from_mapping(config.vlm)
        repository_root = Path(__file__).resolve().parents[2]
        cache_root = (repository_root / config.feature_cache["root"]).resolve()
        if not cache_root.is_relative_to(repository_root):
            raise ValueError("feature cache escapes the repository root")
        device_hash = hashlib.sha256(b"lunavla-deterministic-fixture/cpu-v1").hexdigest()
        build_frozen_feature_cache(
            dataset,
            spec,
            DeterministicFixtureExtractor(int(config.policy["parameters"]["condition_input_dim"])),
            cache_root,
            processor_sha256=spec.processor_config_sha256,
            device_environment_sha256=device_hash,
            overwrite=arguments.overwrite,
        )
        result = run_alpha(config, overwrite=arguments.overwrite)
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "rows": result.metrics["row_count"],
                    "evaluation_type": result.metrics["evaluation_type"],
                    "claim_allowed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "validate-v31-evidence-design":
        v31_design = V31EvidenceDesignV1.load(arguments.design)
        print(
            json.dumps(
                {
                    "valid": True,
                    "expected_rows": v31_design.expected_rows,
                    "design_sha256": v31_design.sha256(),
                    "claim_allowed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
