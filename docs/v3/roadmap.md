# LunaVLA v3 roadmap

1. **Alpha 1 — merged:** schema 3, feature/embodiment contracts, v2 migration,
   data QA/replay, fake task fixtures, CPU EngineV3, checkpoint/manifest verification.
2. **Alpha 2 contracts — implemented in this branch:** policy specs, history-aware samples,
   train-only normalization, strict registry dispatch, training configuration, and revision-2
   artifact envelopes.
3. **Alpha 2 policy ladder — planned as sequential PRs:** native ACT, LeRobot Diffusion, and
   upstream SmolVLA adapters through one engine, with strict resume and bounded single-GPU smokes.
4. **Beta 1 — planned:** prompt rendering parity, declared state routes, paired interventions, and
   language/vision/state/action/execution failure reports.
5. **Beta 2 — planned:** pinned public PushT plus LIBERO-Spatial task IDs 0–3, explicitly reported as
   a diagnostic subset rather than a full benchmark.
6. **RC/stable — planned:** frozen contracts, five-seed controlled evidence, post-merge verification,
   signed assets, SBOM, provenance, and checksums.

No release requires a target success rate. Negative or neutral evidence remains publishable; an
incomplete or mixed-provenance matrix cannot open a claim.
