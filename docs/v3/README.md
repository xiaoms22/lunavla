# LunaVLA v3 Beta 1 diagnostic framework draft

This branch builds the policy and artifact contracts needed by the Alpha 2 strategy ladder on top
of the merged Alpha 1 foundation. It preserves the frozen v2 public API and lightweight v1.x path.

Implemented in the current draft:

- named multimodal feature and embodiment contracts;
- immutable `ObservationV3`, `TransitionV3`, and ordered episode records;
- strict ExperimentConfig schema 3 and explicit v2→v3 migration;
- deterministic fake PushT and fake LIBERO fixtures for CPU data/engine tests;
- dataset QA, replay, strict checkpoint envelope, RunManifest schema 4, and tamper verification;
- `lunavla-v3` config, audit, replay, run, verify, and migration commands.
- strict `PolicySpecV3`, sample/batch, train-step, model-source, and registry contracts;
- train-split-only normalization statistics with stable hashes;
- explicit optimizer, scheduler, precision, clipping, and resume configuration defaults;
- schema-4 revision-2 checkpoint directories and manifests; revision 1 remains read-only.
- native `act_v3` action-query CVAE Transformer through the same registry and Engine, with one
  ordered RGB camera, state, instruction, masked chunks, temporal ensembling, and exact CPU resume.
- `diffusion_v3`, an adapter over the public LeRobot 0.6.0 Diffusion Policy and processor APIs,
  with train-only normalization, DDIM noise control, padding masks, `save_pretrained()` artifacts,
  exact CPU optimizer/RNG resume, and a hash-locked CPU dependency profile.
- `lerobot_smolvla`, a dependency-injected public-API conformance adapter. Its fixed model revision
  and 907 MB weight hash are recorded, but `license_status=unverified`, pretrained loading is
  disabled, and no optimizer/resume claim is made.
- config contract revision 2 with explicit prompt rendering and state routing while revision 1
  retains byte-stable resolved serialization;
- canonical byte-hashed prompts, routes `none/expert_only/prompt_only/dual`, same-split donor banks,
  and step-wise `control/mask/shuffle/counterfactual/layout_drift` prompt interventions;
- schema-4 revision-3 diagnostic run manifests, EvidenceManifest v2, tamper verification, per-pair
  CSV, and a dependency-free static report;
- `diagnostic-run`, `diagnostic-verify`, and `diagnostic-report` commands. The tracked CI design is
  reduced and always records `claim_allowed=false`.

The fake LIBERO fixture tests data shape and lifecycle only. This branch does not download real
LeRobot or LIBERO data, download or train SmolVLA weights, or establish language, image, state-route,
PushT, LIBERO, ACT, or Diffusion performance.

This work remains in a Draft PR and must not merge into `v3` before the separately gated Alpha 2
tag is created from the current integration-branch baseline.

- [Direction](v3_overview.md)
- [Roadmap](roadmap.md)
- [Compatibility](compatibility.md)
- [Capability matrix](capability_matrix.md)
- [Machine-readable public contract](public_api_contract.json)
