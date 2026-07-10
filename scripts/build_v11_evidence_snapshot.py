from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_run_manifest import read_json, sha256_file, validate_manifest


def resolve(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def safe_run_id(raw: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-.")
    if not value or value in {".", ".."}:
        raise ValueError(f"Unsafe run_id: {raw!r}")
    return value


def manifest_run_id(run_dir: Path) -> str:
    manifest = read_json(run_dir / "manifest.json")
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    return safe_run_id(
        str(manifest.get("run_id") or config.get("project_name") or run_dir.name)
    )


def copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": destination.as_posix(),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def representative_rollouts(run_dir: Path) -> dict[str, Path]:
    rollout_dir = run_dir / "rollouts"
    if not rollout_dir.is_dir():
        return {}
    selected: dict[str, Path] = {}
    for path in sorted(rollout_dir.glob("*.json")):
        try:
            rollout = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        label = "success" if bool(rollout.get("success")) else "failure"
        selected.setdefault(label, path)
        if len(selected) == 2:
            break
    return selected


def snapshot_run(
    run_dir: Path,
    output_root: Path,
    *,
    overwrite: bool,
    allow_observational: bool,
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    errors = validate_manifest(manifest_path)
    if errors:
        raise ValueError(f"Invalid manifest {manifest_path}: " + "; ".join(errors))
    manifest = read_json(manifest_path)
    evidence_path = run_dir / "evidence.json"
    evidence = read_json(evidence_path) if evidence_path.is_file() else {}
    if evidence and evidence.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"Evidence declaration does not match manifest: {evidence_path}")
    controlled = bool(evidence.get("controlled"))
    if not controlled and not allow_observational:
        raise ValueError(
            f"{manifest_path} is not declared controlled; pass --allow-observational to archive it with that label"
        )

    run_id = manifest_run_id(run_dir)
    destination = output_root / "runs" / run_id
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Snapshot exists: {destination}; pass --overwrite")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    files: list[dict[str, Any]] = []
    config_path = run_dir / "config.resolved.json"
    metrics_path = run_dir / "eval_summary.json"
    files.append(copy_verified(manifest_path, destination / "manifest.json"))
    files.append(copy_verified(config_path, destination / "resolved_config.json"))
    files.append(copy_verified(metrics_path, destination / "metrics.json"))
    if evidence_path.is_file():
        files.append(copy_verified(evidence_path, destination / "evidence.json"))

    for label, rollout_path in representative_rollouts(run_dir).items():
        files.append(copy_verified(rollout_path, destination / f"rollout_{label}.json"))

    for item in files:
        item["path"] = Path(item["path"]).relative_to(output_root).as_posix()
    return {
        "run_id": run_id,
        "manifest": f"runs/{run_id}/manifest.json",
        "controlled": controlled,
        "files": files,
    }


def snapshot_analysis(
    analysis_root: Path,
    output_root: Path,
    *,
    allow_observational: bool,
) -> dict[str, Any]:
    design_path = analysis_root / "design.json"
    episode_path = analysis_root / "per_episode.csv"
    design = read_json(design_path)
    controlled = bool(design.get("controlled"))
    if not controlled and not allow_observational:
        raise ValueError(
            f"{design_path} is not controlled; pass --allow-observational only for development snapshots"
        )
    train_seeds = [int(value) for value in design.get("train_seeds", [])]
    eval_seeds = [int(value) for value in design.get("eval_seeds", [])]
    expected_trials = len(train_seeds) * len(eval_seeds)
    if not train_seeds or not eval_seeds:
        raise ValueError(f"analysis design has no train/eval seeds: {design_path}")
    if not episode_path.is_file():
        raise FileNotFoundError(f"missing controlled per-episode metrics: {episode_path}")

    analysis_dir = output_root / "analysis"
    files = [
        copy_verified(design_path, analysis_dir / "design.json"),
        copy_verified(episode_path, analysis_dir / "per_episode.csv"),
    ]
    summaries: dict[str, str] = {}
    families = design.get("families")
    if not isinstance(families, dict) or not families:
        raise ValueError(f"analysis design has no experiment families: {design_path}")
    for family in sorted(families):
        if not re.fullmatch(r"[a-z0-9-]+", str(family)):
            raise ValueError(f"unsafe experiment family: {family!r}")
        source = analysis_root / str(family) / "summary.json"
        summary = read_json(source)
        if summary.get("family") != family:
            raise ValueError(f"summary family mismatch: {source}")
        if bool(summary.get("controlled")) != controlled:
            raise ValueError(f"summary controlled flag mismatch: {source}")
        for aggregate in summary.get("aggregates", []):
            if int(aggregate.get("trials", -1)) != expected_trials:
                raise ValueError(
                    f"aggregate trial count in {source} is not {expected_trials}"
                )
        for contrast in summary.get("contrasts", []):
            if int(contrast.get("paired_n", -1)) != expected_trials:
                raise ValueError(
                    f"paired contrast count in {source} is not {expected_trials}"
                )
        destination = analysis_dir / f"{family}_summary.json"
        files.append(copy_verified(source, destination))
        summaries[str(family)] = destination.relative_to(output_root).as_posix()

    for item in files:
        item["path"] = Path(item["path"]).relative_to(output_root).as_posix()
    return {
        "controlled": controlled,
        "train_seed_count": len(train_seeds),
        "eval_episode_count": len(eval_seeds),
        "design": "analysis/design.json",
        "per_episode": "analysis/per_episode.csv",
        "summaries": summaries,
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact v1.1 evidence snapshot from validated manifests.")
    parser.add_argument("--runs", nargs="+", required=True, help="Completed run directories containing manifest.json.")
    parser.add_argument(
        "--analysis-root",
        help="Controlled-run root containing design.json, per_episode.csv, and family summaries.",
    )
    parser.add_argument("--out-dir", default="results/v1.1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-observational", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = [resolve(raw) for raw in args.runs]
    declared: dict[str, Path] = {}
    for run_dir in run_dirs:
        run_id = manifest_run_id(run_dir)
        if run_id in declared:
            raise ValueError(
                f"duplicate run_id {run_id!r}: {declared[run_id]} and {run_dir}"
            )
        declared[run_id] = run_dir
    output_root = resolve(args.out_dir)
    tracked_readme = ROOT / "results" / "v1.1" / "README.md"
    readme_text = (
        tracked_readme.read_text(encoding="utf-8")
        if tracked_readme.is_file()
        else "# v1.1 evidence snapshots\n"
    )
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(readme_text, encoding="utf-8")
    index_path = output_root / "index.json"
    if index_path.exists():
        index = read_json(index_path)
    else:
        index = {"schema_version": "1.0", "release": "v1.1", "generated_at": None, "runs": []}
    existing = {str(item.get("run_id")): item for item in index.get("runs", [])}

    for run_dir in run_dirs:
        item = snapshot_run(
            run_dir,
            output_root,
            overwrite=args.overwrite,
            allow_observational=args.allow_observational,
        )
        existing[item["run_id"]] = item

    if args.analysis_root:
        index["analysis"] = snapshot_analysis(
            resolve(args.analysis_root),
            output_root,
            allow_observational=args.allow_observational,
        )

    index["schema_version"] = "1.0"
    index["release"] = "v1.1"
    index["generated_at"] = os.environ.get("SOURCE_DATE_EPOCH")
    index["runs"] = [existing[key] for key in sorted(existing)]
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        display = index_path.relative_to(ROOT).as_posix()
    except ValueError:
        display = index_path.as_posix()
    print(f"evidence snapshot: {display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
