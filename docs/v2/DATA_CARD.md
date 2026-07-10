# LunaVLA v2 experimental data card

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
