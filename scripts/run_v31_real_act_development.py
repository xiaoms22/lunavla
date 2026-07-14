#!/usr/bin/env python3
"""Train the ten v3.1 ACT development runs against a verified real VLM cache.

This is deliberately not the 2,400-row scientific executor: it establishes the
five-seed training/checkpoint path first and keeps every claim closed.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import yaml

from lunavla.v3 import ExperimentConfig, VLMBackendSpecV1, verify_frozen_feature_cache
from lunavla.v3.engine import EngineV3, dataset_for_config


SEEDS = (11, 22, 33, 44, 55)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return sha, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        sha = os.environ.get("LUNAVLA_GIT_SHA", "")
        dirty = os.environ.get("LUNAVLA_GIT_DIRTY")
        if len(sha) != 40 or dirty not in {"0", "1"}:
            raise RuntimeError("container execution requires bound Git identity") from None
        return sha, dirty == "1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--backend-spec", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    output = args.out.resolve()
    cache = args.cache.resolve()
    repository = Path(__file__).resolve().parents[1]
    if not cache.is_relative_to(repository):
        raise ValueError("development cache must be contained by the repository output root")
    index = verify_frozen_feature_cache(cache)
    spec = VLMBackendSpecV1.from_mapping(json.loads(args.backend_spec.read_text()))
    if index.backend_spec_sha256 != spec.sha256():
        raise ValueError("cache and backend spec hashes differ")
    base = yaml.safe_load(args.base_config.read_text())
    if not isinstance(base, dict):
        raise TypeError("base config must be a mapping")
    if output.exists() and not args.overwrite:
        raise FileExistsError("output exists; use --overwrite")
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    sha, dirty = _git()
    records: list[dict[str, object]] = []
    try:
        for seed in SEEDS:
            for arm, mode in (("baseline", "learned_null"), ("smol_control", "frozen_feature")):
                payload = copy.deepcopy(base)
                payload["project_name"] = f"lunavla-v31-real-{arm}-seed-{seed}"
                payload["training"]["seed"] = seed
                payload["training"]["steps"] = args.steps
                payload["training"]["batch_size"] = 32
                payload["policy"]["parameters"]["condition_mode"] = mode
                payload["policy"]["parameters"]["condition_input_dim"] = 960
                payload["policy"]["parameters"]["instruction_dim"] = 960
                payload["policy"]["parameters"]["feature_intervention"] = "control"
                payload["dataset"]["parameters"] = {
                    "train_per_task": 6,
                    "held_out_per_cell": 2,
                }
                payload["vlm"] = spec.to_dict()
                payload["feature_cache"] = {
                    "enabled": True,
                    "root": cache.relative_to(repository).as_posix(),
                    "backend_spec_sha256": spec.sha256(),
                    "read_only": True,
                }
                payload["artifacts"]["output_dir"] = (
                    output.relative_to(repository).as_posix()
                    if output.is_relative_to(repository)
                    else "outputs/v3/v31-real-act-development"
                )
                config = ExperimentConfig.from_mapping(payload)
                bundle = dataset_for_config(config)
                engine = EngineV3(config)
                policy, losses = engine.train(bundle.source("train"))
                metrics = engine.evaluate_dataset(policy, bundle.select("test"))
                cell = staging / f"seed-{seed}" / arm
                cell.mkdir(parents=True)
                checkpoint = policy.save_checkpoint(
                    cell / "act_v3.pt",
                    metadata={"development_only": True, "claim_allowed": False},
                )
                resolved = cell / "resolved-config.json"
                resolved.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
                metrics_path = cell / "metrics.json"
                metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
                records.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "steps": args.steps,
                        "first_loss": losses[0],
                        "final_loss": losses[-1],
                        "checkpoint_sha256": _sha256(checkpoint),
                        "config_sha256": config.sha256(),
                        "metrics_sha256": _sha256(metrics_path),
                    }
                )
                del policy, engine
                gc.collect()
        manifest = {
            "schema_version": 1,
            "execution_role": "development_training_connectivity",
            "git_sha": sha,
            "git_dirty": dirty,
            "feature_source": "real_frozen_vlm",
            "cache_index_sha256": index.sha256(),
            "train_seeds": list(SEEDS),
            "training_runs": len(records),
            "claim_allowed": False,
            "release_eligible": False,
            "gate_reasons": ["development_budget", "no_dynamic_rollout_evidence"],
            "runs": records,
        }
        (staging / "development-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
        print(json.dumps({"output": str(output), "training_runs": len(records)}))
        return 0
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
