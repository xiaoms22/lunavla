# LunaVLA roadmap

## v1.1 — trustworthy teaching core

The release gate is a CPU-only, reproducible NumPy loop with honest documentation, tested numerical behavior, strict configuration/data/checkpoint contracts, controlled experiments, and hash-verifiable evidence. The release does not depend on reaching a particular success rate.

## v1.x maintenance

The NumPy quickstart remains lightweight and compatible across Python 3.10–3.12. Deprecated v1.0 aliases are removed only in a subsequent minor release with migration notes.

## v2 integration branch

The v2 branch starts from a green `v1.1.0` tag and keeps heavy dependencies out of the v1.x profile.

1. **Complete — v2.0.0-alpha.1, unified engine:** common observation, policy, environment, and dataset protocols; NumPy and PyTorch policies through one registry and engine.
2. **Complete — v2.0.0-alpha.2, language evidence:** three instruction-dependent goals, held-out paraphrases, and the full multi-seed mask/shuffle/counterfactual study. Its contribution claim remains closed.
3. **Complete — v2.0.0-beta.1, visual and LeRobot adapters:** vision-required direct/waypoint point reach, state-only controls, the full image-ablation study, and pinned real LeRobot dataset/environment smoke. The visual contribution claim remains closed.
4. **Complete — v2.0.0-rc.1, contract freeze:** frozen public APIs, config, manifest, and checkpoint schemas; migration/card/security documentation; package/SBOM/checksum profile.
5. **Current — v2.0.0, stable gate:** merge `v2` through a protected PR, then run the fail-closed complete evidence and real LeRobot gates on the resulting `main` SHA before publishing signed matching assets.

Claims about a modality are gated on a controlled paired interval excluding zero, not simply on adding an input field.
