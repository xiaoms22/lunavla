# LunaVLA v3 roadmap

1. **Alpha 1 — merged:** schema 3, feature/embodiment contracts, v2 migration,
   data QA/replay, fake task fixtures, CPU EngineV3, checkpoint/manifest verification.
2. **Alpha 2 contracts — implemented:** policy specs, history-aware samples,
   train-only normalization, strict registry dispatch, training configuration, and revision-2
   artifact envelopes.
3. **Alpha 2 native ACT — implemented:** action-query CVAE Transformer, masked
   chunks, one RGB camera, state/instruction conditioning, temporal ensembling, and exact CPU
   checkpoint resume through the unified Engine.
4. **Alpha 2 LeRobot Diffusion — implemented:** public policy/processor APIs,
   DDIM scheduler, history and padding contracts, hash-locked CPU dependencies, and exact resume.
5. **Alpha 2 SmolVLA adapter — implemented:** dependency-injected public policy and
   processor conformance, pinned upstream identity, and fail-closed pretrained/optimizer gates.
   Actual weights, optimizer, resume, and inference remain blocked until the model-weight license
   and a single GPU are verified.
6. **Alpha 3 code-only release — published prerelease:** package `3.0.0a3`, hosted CPU checks,
   reproducible signed assets and a conformance-only SmolVLA status. No weight or GPU is required. The original
   license/GPU validation contracts move intact to the future v3.1 track.
7. **Beta 1 — integration candidate:** canonical prompt parity, declared state
   routes, same-split donors, step-wise paired interventions, five-layer failure records,
   revision-3 run manifests, EvidenceManifest v2, and verified static reports. The CPU gate runs a
   40-pair NumPy route/prompt matrix and an independent four-pair ACT image-shuffle smoke with
   synthetic thumbnails. These reduced fixtures prove only framework integrity and cannot open a
   scientific claim. Integration targets the protected `v3-next` branch after Alpha 3.
8. **Beta 2 — optional v3.1 connectivity Draft:** pinned public PushT plus LIBERO-Spatial task IDs
   0–3, explicitly reported as a diagnostic subset rather than a full benchmark. It does not block
   the fixture-only v3.0 stable release.
9. **CPU core hardening — verified on the integration baseline:** versioned environment-specific
   policy profiles and all 1,550 rows of the deterministic teaching evidence matrix, without GPU
   or pretrained weights. The release run must still be repeated on the RC and default-branch
   merge SHAs.
10. **RC/stable — active release candidate:** package `3.0.0rc1`, a strict RC candidate contract,
    exact release-asset inventory, independent `SHA256SUMS` verification, migration/model/data/
    safety documents, and asset/SBOM/provenance requirements. No RC or stable tag exists yet.
11. **v3.1 frozen VLM features — planned:** fixed public VLM revisions feed offline frozen
    features to the ACT action expert on deterministic synthetic tasks. No VLM contribution is
    implemented or claimed in v3.0.

No release requires a target success rate. Negative or neutral evidence remains publishable; an
incomplete or mixed-provenance matrix cannot open a claim.
