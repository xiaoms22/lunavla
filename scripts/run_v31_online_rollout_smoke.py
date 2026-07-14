#!/usr/bin/env python3
"""Run three bounded real-feature ACT rollouts against the dynamic v3.1 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lunavla.v3 import (
    ContentAddressedFrozenFeatureProviderV1,
    ExperimentConfig,
    FrozenFeatureCacheReaderV1,
    PolicySampleV3,
    TransformersFrozenExtractor,
    V31RolloutEnvV1,
    V31_TASK_IDS,
    VLMBackendSpecV1,
)
from lunavla.v3.act_policy import ActPolicyV3
from lunavla.v3.engine import EngineV3, dataset_for_config
from lunavla.v3.normalization import fit_normalization_stats


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend-spec", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--online-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("rollout smoke output already exists")
    config = ExperimentConfig.from_mapping(json.loads(args.config.read_text()))
    bundle = dataset_for_config(config)
    engine = EngineV3(config)
    engine.normalization = fit_normalization_stats(
        bundle.select("train"), config.feature_schema
    )
    policy = engine.restore_policy(args.checkpoint)
    if not isinstance(policy, ActPolicyV3):
        raise TypeError("online rollout smoke requires ActPolicyV3")
    spec = VLMBackendSpecV1.from_mapping(json.loads(args.backend_spec.read_text()))
    extractor = TransformersFrozenExtractor(spec, args.model_root.resolve())
    base = FrozenFeatureCacheReaderV1(args.cache.resolve())
    provider = ContentAddressedFrozenFeatureProviderV1(
        base_cache=base,
        extractor=extractor,
        root=args.online_cache.resolve(),
        processor_sha256=spec.processor_config_sha256,
        device_environment_sha256=hashlib.sha256(
            b"macos-arm64-mps-torch-2.11-single-image-no-split-512"
        ).hexdigest(),
    )
    policy.configure_rollout_feature_provider(provider, intervention="control")
    rows: list[dict[str, object]] = []
    for task_index, task_id in enumerate(V31_TASK_IDS):
        env = V31RolloutEnvV1(
            task_id=task_id,
            stratum="composition",
            episode_index=task_index,
            max_steps=args.max_steps,
        )
        try:
            observation = env.reset(seed=1000 + task_index)
            policy.reset(1000 + task_index)
            transition = None
            for step in range(args.max_steps):
                keys = provider.ensure_observations((observation,), batch_size=1)
                sample = PolicySampleV3(
                    (observation,),
                    np.asarray([True], dtype=bool),
                    None,
                    None,
                    observation.episode_id,
                    observation.step_index,
                )
                action = policy.predict_chunk(sample).values[0]
                transition = env.step(action)
                rows.append(
                    {
                        "task_id": task_id,
                        "episode_index": task_index,
                        "step": step,
                        "feature_key": keys[0],
                        "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
                        "final_distance": transition.info["final_distance"],
                        "success": transition.info["success"],
                    }
                )
                observation = transition.next_observation
                if transition.terminated or transition.truncated:
                    break
            if transition is None:
                raise RuntimeError("rollout produced no transition")
        finally:
            env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "execution_role": "real_feature_dynamic_rollout_smoke",
        "tasks": list(V31_TASK_IDS),
        "rows": rows,
        "checkpoint_sha256": _hash(args.checkpoint),
        "base_cache_binding_sha256": base.binding_sha256,
        "claim_allowed": False,
        "release_eligible": False,
        "gate_reasons": ["bounded_smoke", "incomplete_matrix"],
    }
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.out), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
