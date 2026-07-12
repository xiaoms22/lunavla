# LunaVLA v3 RC/stable preparation stacked draft

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
- strict Alpha 2 license, GPU validation and release-candidate contracts; hash-locked CUDA/release
  profiles; and a default-branch-compatible signed-tag dispatcher. These remain inactive until a
  separate gate-opening PR supplies explicit weight-license evidence and a qualifying runner.
- config contract revision 2 with explicit prompt rendering and state routing while revision 1
  retains byte-stable resolved serialization;
- canonical byte-hashed train/eval/deploy preprocessing, routes
  `none/expert_only/prompt_only/dual`, typed same-split instruction/image donor banks, and step-wise
  `control/mask/shuffle/counterfactual/layout_drift` prompt interventions;
- schema-4 revision-3 diagnostic run manifests, EvidenceManifest v2, tamper verification, per-pair
  CSV, and a dependency-free static report;
- `diagnostic-run`, `diagnostic-verify`, and `diagnostic-report` commands. The tracked CI design is
  a reduced 40-pair NumPy route/prompt matrix plus a separate four-pair ACT image-shuffle smoke;
  both always record `claim_allowed=false`.
- four byte-reproducible 16×16 synthetic PNG fixtures with a hash manifest. Real-data thumbnails
  remain prohibited.
- config contract revision 3 plus strict `ExternalDatasetSpecV1`, `SimulationTaskSpecV1`, and
  `IntegrationManifestV1` contracts for pinned PushT and LIBERO-Spatial task IDs 0–3;
- LeRobot v3 frame mapping, two-camera LIBERO state/action mapping, source download caps, and
  exactly-once PushT/LIBERO environment adapters;
- `source-preflight`, `integration-run`, and `integration-verify` commands, with an offline PR
  fixture gate and a separate manual single-A100 dispatcher. The real dispatcher has not run.
- predeclared stable evidence contracts for the exact 300-row PushT policy, 800-row LIBERO route,
  and 1,000-row LIBERO prompt matrices, plus strict evidence-summary and stable-release candidate
  verification. These contracts do not assert that the studies or release have run.

The PR fixture tests contracts, mapping, lifecycle, atomic output and tamper detection without
network access. No authoritative real-data/GPU manifest exists yet, so the public connectivity
wording remains closed. This branch does not download or train SmolVLA weights or establish
language, image, state-route, PushT, LIBERO, ACT, or Diffusion performance.

This work remains in a Draft PR and must not merge into `v3` before the separately gated Alpha 2
tag is created from the current integration-branch baseline.

- [Direction](v3_overview.md)
- [Roadmap](roadmap.md)
- [Compatibility](compatibility.md)
- [Capability matrix](capability_matrix.md)
- [Alpha 2 release process](alpha2_release_process.md)
- [SmolVLA runner qualification](smolvla_runner_qualification.md)
- [Beta 2 bounded integration](beta2_integration.md)
- [RC and stable release boundary](rc_stable_release.md)
- [v2 to v3 migration guide](migration_v2_to_v3.md)
- [Pre-RC model card](model_card.md)
- [Pre-RC data card](data_card.md)
- [Security, privacy and safety boundary](security_and_safety.md)
- [Machine-readable public contract](public_api_contract.json)
