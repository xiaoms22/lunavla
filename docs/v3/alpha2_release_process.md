# LunaVLA v3 Alpha 2 release process

Alpha 2 is fail-closed. The repository currently implements only the release contracts and the
manual dispatcher; it does not assert that the SmolVLA model-weight license is verified and does
not download the weights in CI.

## Immutable identities

- Git tag: `v3.0.0-alpha.2`
- Python distribution version after the gate-opening PR: `3.0.0a2`
- GPU runtime: Linux x86_64, one NVIDIA GPU, CUDA 12.8, Torch 2.11.0
- Model: `lerobot/smolvla_base` at an immutable reviewed revision
- Existing reviewed weight bytes must retain SHA-256
  `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.

The code repository's Apache-2.0 license is not accepted as evidence for model weights. A future
gate-opening PR must add `docs/v3/release/smolvla-license-review.json`; its evidence URL must be an
official Hugging Face URL that explicitly applies to the named model weights.

The current normalized model-card, repository-metadata, and file-inventory observations are pinned
by `docs/v3/release/smolvla-license-status.json`. That record is deliberately
`NOASSERTION`/`unverified`; citing the model, using it non-commercially, or receiving project-owner
permission cannot turn it into a `LicenseReviewV1`.

## Runner qualification

The dispatcher's `preflight` phase qualifies an isolated A100 runner without a license Environment
or model-weight access. Volcengine is the authoritative role and the isolated development-host GPU
is secondary. Each run produces a `RunnerQualificationManifestV1` with hashed runner/GPU identity,
locked dependencies, single-device checks, resource limits, outbound connectivity, mount isolation,
and fail-closed release fields. See [the operator guide](smolvla_runner_qualification.md).

## Two-phase dispatcher

`.github/workflows/v3-alpha2-release-dispatch.yml` must exist with identical bytes on the default
branch and the reviewed source SHA.

The `gpu` phase validates the committed license review before any weight access, installs the
hash-locked CUDA profile, verifies every downloaded file, performs one optimizer step, saves and
restores a checkpoint, and runs an inference smoke. It publishes only a small manifest and GitHub
attestation; weights, checkpoints, caches and samples are deleted or excluded.

The hosted CPU job then checks all protected v3 gates and builds a pre-tag candidate. The
`finalize` phase is allowed only after an SSH-signed annotated tag points to the same merge SHA and
GitHub reports the tag signature as verified. It reproduces the distributions, verifies the GPU
attestation, and creates a draft prerelease. Publishing that draft is a separate asset-review
action.

## Current blockers

- `configs/v3/smolvla_pretrained_gpu.yaml` remains
  `license_status=unverified`, `pretrained_enabled=false`, and `conformance_only=true`.
- No committed license review exists.
- No authoritative and secondary qualification manifests have been produced yet.
- The package remains version 2.0.0 until the gate-opening PR.

These are intentional release blockers, not incomplete claims. There is no timeout or automatic
downgrade. Alpha 2 does not publish to PyPI and does not claim task performance, modality benefit,
instruction following, policy superiority, or robot deployment readiness.
