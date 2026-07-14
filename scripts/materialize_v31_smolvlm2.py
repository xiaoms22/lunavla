#!/usr/bin/env python3
"""Materialize the pinned SmolVLM2 runtime inventory outside Git.

Network access belongs only to this step. All later extraction and training use
the emitted exact-file backend spec with local_files_only=True.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from lunavla.v3.v31_contracts import VLMBackendSpecV1
from lunavla.v3.v31_vlm import SMOLVLM2_REPO_ID, SMOLVLM2_REVISION


RUNTIME_FILES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--license-evidence", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite model root: {output}")
    if not args.license_evidence.is_file():
        raise FileNotFoundError("license evidence snapshot is required")
    snapshot = Path(
        snapshot_download(
            repo_id=SMOLVLM2_REPO_ID,
            revision=SMOLVLM2_REVISION,
            allow_patterns=list(RUNTIME_FILES),
        )
    )
    staging = output.with_name(f".{output.name}.staging")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for relative in RUNTIME_FILES:
            source = snapshot / relative
            if not source.is_file():
                raise FileNotFoundError(f"pinned snapshot lacks {relative}")
            shutil.copyfile(source, staging / relative)
        inventory = {name: _sha256(staging / name) for name in RUNTIME_FILES}
        total = sum((staging / name).stat().st_size for name in RUNTIME_FILES)
        spec = VLMBackendSpecV1(
            backend_id="smolvlm2_500m",
            repo_id=SMOLVLM2_REPO_ID,
            revision=SMOLVLM2_REVISION,
            spdx_license="Apache-2.0",
            license_scope="model_weights",
            license_evidence_sha256=_sha256(args.license_evidence),
            processor_class="AutoProcessor",
            processor_config_sha256=inventory["processor_config.json"],
            model_config_sha256=inventory["config.json"],
            hidden_layer=-1,
            pooling="attention_mask_mean",
            image_token_layout="single_image_no_split_512",
            camera_order=("camera.primary",),
            model_dtype="float32",
            device="cpu",
            offload_plan="none",
            deterministic=True,
            evidence_role="claim_bearing",
            weight_files=inventory,
            total_weight_bytes=total,
        )
        staging.replace(output)
        manifest = output.parent / "smolvlm2-backend-spec.json"
        manifest.write_text(
            json.dumps(spec.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"model_root": str(output), "backend_spec": str(manifest)}))
        return 0
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
