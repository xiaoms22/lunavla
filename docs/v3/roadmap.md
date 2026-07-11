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
5. **Alpha 2 SmolVLA — next sequential PR:** public-API conformance adapter with its pretrained
   weight and optimizer gates closed until the model-weight license and a single GPU are verified.
6. **Beta 1 — planned:** prompt rendering parity, declared state routes, paired interventions, and
   language/vision/state/action/execution failure reports.
7. **Beta 2 — planned:** pinned public PushT plus LIBERO-Spatial task IDs 0–3, explicitly reported as
   a diagnostic subset rather than a full benchmark.
8. **RC/stable — planned:** frozen contracts, five-seed controlled evidence, post-merge verification,
   signed assets, SBOM, provenance, and checksums.

No release requires a target success rate. Negative or neutral evidence remains publishable; an
incomplete or mixed-provenance matrix cannot open a claim.
