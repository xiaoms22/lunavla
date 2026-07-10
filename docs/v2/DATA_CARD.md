# LunaVLA v2 release-candidate data card

v2 keeps the v1.1 synthetic state demonstrations and adds protocol fixtures for language and rendered images. These fixtures exist to make causal controls testable; they are not representative robot datasets.

The language fixture uses at least three targets with identical initial state and disjoint training/held-out instruction templates. Mask, shuffle, and counterfactual variants preserve pair identifiers and action targets.

The visual fixture renders RGB direct-reach and waypoint-reach observations and produces state-only, occluded-image, and shuffled-image controls. Rendered pixels are synthetic. A separate lazy adapter is responsible for converting supported LeRobot episodes to LunaVLA transitions while validating required state/action/image fields.

Every controlled evaluation must keep split, start state, target, policy initialization, training budget, and evaluation seeds fixed within a pair. Manifests must identify the source and intervention and hash the exact data used. No language or visual effectiveness claim follows merely from successful loading or training.

The beta connectivity smoke separately pins `lerobot/pusht` episode 0 at revision
`b1c3ecbae7f244acc039a3dbc255a00dad1372b9`. It verifies the upstream parquet, video, and episode
metadata hashes, decodes all 161 frames through PyAV, and records a schema-1
`integration_manifest.json`. This integration artifact has `claim_allowed=false`: loading one
official episode and stepping the real Gym PushT environment do not constitute a PushT performance
result. See [the pinned integration contract](lerobot_integration.md).

## Registered evidence provenance

- Language study: clean source `a546695445f6fa6e717cd560d5acf718e037940a`, workflow run `29106885353`, EvidenceManifest SHA-256 `106ea2421d37c6c374e31d01a788101e358317f76b6abc315318634e6c6fa3b8`.
- Visual study: clean source `bf0e550a7aa3fb0bb07354cd7cb525752c56268d`, workflow run `29110701437`, EvidenceManifest SHA-256 `d8ff8c798a6810a09a2905dbafd6f5259ac2356623ee6060d335d660db6e9056`.
- Real LeRobot/Gym smoke: the same Beta source `bf0e550a7aa3fb0bb07354cd7cb525752c56268d`, workflow run `29110699054`, integration manifest SHA-256 `7d52e3fd39e1ccb94adac82692a82d653434a4f12dd90762ea18797ab7a4f5ca`.

The first two snapshots are tracked under `results/v2/` and reverified read-only by CI. The integration manifest remains a provenance-attested workflow/release artifact; no upstream video or cache is committed.
