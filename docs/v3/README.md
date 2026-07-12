# LunaVLA v3 CPU-core RC preparation

This branch builds the policy and artifact contracts introduced by the Alpha 2 strategy ladder on top
of the merged Alpha 1 foundation. It preserves the frozen v2 public API and lightweight v1.x path.

Implemented in the current candidate:

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
- a published hosted CPU code-only Alpha 3 prerelease. It packages ACT, Diffusion
  and SmolVLA public-API conformance without accessing weights. The original license/GPU contracts
  remain fail-closed in the separate v3.1 validation track.
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
- versioned CPU profiling with fixed 5-run warmup, 20 measurements, latency, throughput and peak
  RSS manifests. Measurements are environment-specific and cannot support policy rankings.
- predeclared CPU teaching-evidence contracts for exact 200-row policy, 600-row route and 750-row
  prompt-intervention matrices. These contracts verify structure and provenance; they do not
  assert that the five-seed studies or a stable release have run.
- `stable-run` and `stable-verify` execute each frozen study through the unified Engine, rerun the
  complete seed-11 subset, atomically preserve generations, and independently reject row, tree,
  statistics or sentinel tampering.

The v3.0 stable gate is intentionally fixture-only: the exact 1,550-row CPU evidence matrix is
required, while independently verified real PushT/LIBERO connectivity remains a non-blocking
supplement and cannot be substituted for performance evidence.

The fake LIBERO fixture tests data shape and lifecycle only. This branch does not download real
LeRobot or LIBERO data, download or train SmolVLA weights, or establish language, image, state-route,
PushT, LIBERO, ACT, or Diffusion performance.

This work targets the protected `v3-next` integration branch after the signed Alpha 3 prerelease.
Its reduced diagnostic fixtures remain framework-only and cannot open a scientific claim.

- [Direction](v3_overview.md)
- [Roadmap](roadmap.md)
- [Compatibility](compatibility.md)
- [Capability matrix](capability_matrix.md)
- [Alpha 3 release process](alpha3_release_process.md)
- [SmolVLA runner qualification](smolvla_runner_qualification.md)
- [CPU profiling](cpu_profiling.md)
- [Verified learner portfolio export](portfolio_export.md)
- [RC and stable release boundary](rc_stable_release.md)
- [v2 to v3 migration guide](migration_v2_to_v3.md)
- [Pre-RC model card](model_card.md)
- [Pre-RC data card](data_card.md)
- [Security, privacy and safety boundary](security_and_safety.md)
- [Machine-readable public contract](public_api_contract.json)
