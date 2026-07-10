# LunaVLA roadmap

## v1.1 — trustworthy teaching core

The release gate is a CPU-only, reproducible NumPy loop with honest documentation, tested numerical behavior, strict configuration/data/checkpoint contracts, controlled experiments, and hash-verifiable evidence. The release does not depend on reaching a particular success rate.

## v1.x maintenance

The NumPy quickstart remains lightweight and compatible across Python 3.10–3.12. Deprecated v1.0 aliases are removed only in a subsequent minor release with migration notes.

## v2 integration branch

The v2 branch starts from a green `v1.1.0` tag and keeps heavy dependencies out of the v1.x profile.

1. **v2.0 alpha — unified engine:** common observation, policy, environment, and dataset protocols; NumPy and PyTorch policies through one registry and engine. The `act` name is used only after action-query Transformer, CVAE/KL, masks, and temporal ensembling exist.
2. **v2.1 alpha — measurable language conditioning:** at least three instruction-dependent tasks plus held-out paraphrase, mask, shuffle, and counterfactual evaluation.
3. **v2.2 beta — visual and LeRobot adapters:** rendered visual point reach, state-only controls, image ablations, and an isolated LeRobot PushT path.
4. **v2.0 stable:** frozen public APIs and schemas, v1.1 migration tooling, compatibility notes, cards, signed releases, SBOM, manifests, and checksums.

Claims about a modality are gated on a controlled paired interval excluding zero, not simply on adding an input field.
