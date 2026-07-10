# LunaVLA roadmap

## v1.1 — trustworthy teaching core

The release gate is a CPU-only, reproducible NumPy loop with honest documentation, tested numerical behavior, strict configuration/data/checkpoint contracts, controlled experiments, and hash-verifiable evidence. The release does not depend on reaching a particular success rate.

## v1.x maintenance

The NumPy quickstart remains lightweight and compatible across Python 3.10–3.12. Deprecated v1.0 aliases are removed only in a subsequent minor release with migration notes.

## v2 integration branch

The v2 branch starts from a green `v1.1.0` tag and keeps heavy dependencies out of the v1.x profile.

1. **v2.0.0-alpha.1 — unified engine:** common observation, policy, environment, and dataset protocols; NumPy and PyTorch policies through one registry and engine. Alpha evidence verifies implementation integrity without a modality-effect claim.
2. **v2.0.0-alpha.2 — measurable language conditioning:** three instruction-dependent goals, held-out paraphrases, and multi-seed mask/shuffle/counterfactual evidence.
3. **v2.0.0-beta.1 — visual and LeRobot adapters:** vision-required direct/waypoint point reach, state-only controls, image ablations, and pinned real LeRobot dataset/environment smoke.
4. **v2.0.0-rc.1 — contract freeze:** freeze public APIs, config, manifest, and checkpoint schemas after the controlled gates pass.
5. **v2.0.0 — stable:** publish migration notes, cards, signed release assets, SBOM, manifests, and checksums from the post-merge `main` commit.

Claims about a modality are gated on a controlled paired interval excluding zero, not simply on adding an input field.
