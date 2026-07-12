# LunaVLA v3 Alpha 3 code-only release process

Alpha 3 publishes the CPU policy contracts and training paths without accessing pretrained
SmolVLA weights. Its fixed identities are tag `v3.0.0-alpha.3` and Python package `3.0.0a3`.

## Candidate boundary

The hosted CPU dispatcher requires the protected checks for ACT, Diffusion, SmolVLA public-API
conformance, v1/v2 compatibility, secrets and CodeQL. It builds a wheel, sdist, SBOM, provenance,
environment inventory, deterministic evidence archive and exact checksums.
The build backend is fixed to `setuptools==80.10.2`; PEP 517 isolation is disabled, and the sdist
is normalized to the release commit epoch before byte-for-byte reproduction.

The signed `v3.0.0-alpha.2` tag remains an unpublished failed candidate. Its isolated backend
resolved an unrecorded setuptools version, so its assets are not eligible for release and the tag
is not moved or rewritten.

The candidate must record `license_status=unverified`, `spdx_license=NOASSERTION`,
`pretrained_enabled=false`, `conformance_only=true`, `weight_accessed=false`,
`claim_allowed=false` and `pypi_published=false`. Model weights, checkpoints and caches are rejected
from the asset tree.

Finalization accepts only an SSH-signed annotated tag whose GitHub verification is valid and whose
target equals the candidate SHA. It reproduces wheel/sdist bytes before creating a draft
prerelease. Publishing the draft remains a separate asset-review action.

## SmolVLA v3.1 track

The immutable weight observations and fail-closed `LicenseReviewV1`, runner qualification and GPU
validation contracts remain available, but now target the earliest possible tag
`v3.1.0-alpha.1`. The dispatcher cannot open that gate while the official weight license is
`NOASSERTION` or a qualified runner is absent. A code license, public download, citation or project
owner permission is not weight-license evidence.

Alpha 3 makes no policy-superiority, task-performance, modality, instruction-following or robot
deployment claim and is not published to PyPI.
