# LunaVLA v3 roadmap

1. **Alpha 1 — merged:** schema 3, feature/embodiment contracts, v2 migration,
   data QA/replay, fake task fixtures, CPU EngineV3, checkpoint/manifest verification.
2. **Alpha 2 contracts — implemented in this branch:** policy specs, history-aware samples,
   train-only normalization, strict registry dispatch, training configuration, and revision-2
   artifact envelopes.
3. **Alpha 2 native ACT — implemented in this branch:** action-query CVAE Transformer, masked
   chunks, one RGB camera, state/instruction conditioning, temporal ensembling, and exact CPU
   checkpoint resume through the unified Engine.
4. **Alpha 2 LeRobot Diffusion — implemented in this branch:** public policy/processor APIs,
   DDIM scheduler, history and padding contracts, hash-locked CPU dependencies, and exact resume.
5. **Alpha 2 SmolVLA adapter — implemented in this branch:** dependency-injected public policy and
   processor conformance, pinned upstream identity, and fail-closed pretrained/optimizer gates.
   Actual weights, optimizer, resume, and inference remain blocked until the model-weight license
   and a single GPU are verified.
6. **Alpha 2 release gate — infrastructure implemented, external gates closed:** strict license,
   GPU validation and release-candidate contracts; CUDA 12.8 and release CPU locks; a default-
   branch-compatible two-phase dispatcher; signed-tag and draft-prerelease verification. The
   package version `3.0.0a2` is reserved for the future gate-opening PR and no weight is accessed.
7. **Beta 1 — implemented in an unmerged Draft PR:** canonical prompt parity, declared state
   routes, same-split donors, step-wise paired interventions, five-layer failure records,
   revision-3 run manifests, EvidenceManifest v2, and verified static reports. The CPU gate runs a
   40-pair NumPy route/prompt matrix and an independent four-pair ACT image-shuffle smoke with
   synthetic thumbnails. These reduced fixtures prove only framework integrity and cannot open a
   scientific claim. The work remains unmerged until the Alpha 2 license/GPU release gate is
   satisfied and the Alpha 2 tag is created from `v3`.
8. **Beta 2 — planned:** pinned public PushT plus LIBERO-Spatial task IDs 0–3, explicitly reported as
   a diagnostic subset rather than a full benchmark.
9. **RC/stable — planned:** frozen contracts, five-seed controlled evidence, post-merge verification,
   signed assets, SBOM, provenance, and checksums.

No release requires a target success rate. Negative or neutral evidence remains publishable; an
incomplete or mixed-provenance matrix cannot open a claim.
