from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def validate_manifest(path: Path, *, require_evaluation: bool = True) -> list[str]:
    errors: list[str] = []
    try:
        manifest = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"cannot read manifest: {exc}"]
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {manifest.get('schema_version')!r}")

    required = (
        "git_sha",
        "config",
        "config_sha256",
        "data_sha256",
        "dataset_split",
        "train_seeds",
        "eval_seeds",
        "python_version",
        "dependencies",
        "policy_id",
        "task_id",
        "checkpoint_sha256",
        "command",
        "metrics",
    )
    for key in required:
        if key not in manifest:
            errors.append(f"missing field: {key}")

    config = manifest.get("config")
    if isinstance(config, dict):
        if not manifest.get("run_id") and not config.get("project_name"):
            errors.append("missing field: run_id")
        actual_config_hash = sha256_bytes(canonical_json(config))
        if actual_config_hash != manifest.get("config_sha256"):
            errors.append("config SHA-256 mismatch")
        resolved_path = path.parent / "config.resolved.json"
        if not resolved_path.is_file():
            errors.append("missing config.resolved.json")
        else:
            try:
                if read_json(resolved_path) != config:
                    errors.append("config.resolved.json differs from manifest config")
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid config.resolved.json: {exc}")
        split = str(config.get("dataset", {}).get("split", "train"))
        data_path = path.parent / f"{split}_records.jsonl"
        if not data_path.is_file():
            errors.append(f"missing selected split data: {data_path.name}")
        elif sha256_file(data_path) != manifest.get("data_sha256"):
            errors.append("data SHA-256 mismatch")
    else:
        errors.append("config must be an object")

    checkpoint_path = path.parent / "checkpoint.json"
    if not checkpoint_path.is_file():
        errors.append("missing checkpoint.json")
    elif sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256"):
        errors.append("checkpoint SHA-256 mismatch")

    metrics = manifest.get("metrics")
    if require_evaluation:
        evaluation = metrics.get("evaluation") if isinstance(metrics, dict) else None
        summary_path = path.parent / "eval_summary.json"
        if not isinstance(evaluation, dict):
            errors.append("manifest metrics.evaluation is missing")
        if not summary_path.is_file():
            errors.append("missing eval_summary.json")
        elif isinstance(evaluation, dict):
            try:
                summary = read_json(summary_path)
                if summary != evaluation:
                    errors.append("eval_summary.json differs from manifest metrics.evaluation")
                if [int(value) for value in summary.get("eval_seeds", [])] != [
                    int(value) for value in manifest.get("eval_seeds", [])
                ]:
                    errors.append("manifest eval_seeds differ from evaluation")
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid eval_summary.json: {exc}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the core RunManifest and optionally attach a controlled-evidence declaration."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", help="Accepted for command compatibility; validated via config.resolved.json.")
    parser.add_argument("--checkpoint", help="Accepted for command compatibility; checkpoint.json is authoritative.")
    parser.add_argument("--metrics", help="Accepted for command compatibility; eval_summary.json is authoritative.")
    parser.add_argument("--data", nargs="*", default=[])
    parser.add_argument("--train-seeds", nargs="*", type=int)
    parser.add_argument("--eval-seeds", nargs="*", type=int)
    parser.add_argument("--split")
    parser.add_argument("--command")
    parser.add_argument("--experiment-family", default="unspecified")
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-clean-source", action="store_true")
    return parser.parse_args()


def resolve(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    errors = validate_manifest(manifest_path)
    if not errors:
        manifest = read_json(manifest_path)
        if args.train_seeds:
            configured_train_seed = int(manifest.get("config", {}).get("training", {}).get("seed", -1))
            if configured_train_seed not in set(args.train_seeds):
                errors.append("declared train seeds do not include config.training.seed")
        if args.eval_seeds and [int(value) for value in manifest.get("eval_seeds", [])] != args.eval_seeds:
            errors.append("declared eval seeds differ from manifest eval_seeds")
        if args.require_clean_source:
            # RunManifest intentionally stores the commit, while release CI guarantees a clean checkout.
            # Refuse an unknown commit here; dirty-tree state is enforced by the release workflow.
            if manifest.get("git_sha") in {None, "", "unknown"}:
                errors.append("manifest git_sha is unknown")
    if errors:
        for error in errors:
            print(f"manifest error: {error}", file=sys.stderr)
        return 1

    if args.check:
        print(f"manifest valid: {repo_relative(manifest_path)}")
        return 0

    evidence_path = run_dir / "evidence.json"
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "manifest": "manifest.json",
        "manifest_sha256": sha256_file(manifest_path),
        "experiment_family": args.experiment_family,
        "controlled": bool(args.controlled),
        "declared_train_seeds": args.train_seeds or [],
        "declared_eval_seeds": args.eval_seeds or read_json(manifest_path).get("eval_seeds", []),
        "created_at": os.environ.get("SOURCE_DATE_EPOCH"),
    }
    text = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if evidence_path.exists() and evidence_path.read_text(encoding="utf-8") != text and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {evidence_path}; pass --overwrite")
    evidence_path.write_text(text, encoding="utf-8")
    print(f"manifest valid: {repo_relative(manifest_path)}")
    print(f"evidence declaration: {repo_relative(evidence_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
