# LunaVLA v3 Alpha 2 contracts candidate

This branch builds the policy and artifact contracts needed by the Alpha 2 strategy ladder on top
of the merged Alpha 1 foundation. It preserves the frozen v2 public API and lightweight v1.x path.

Implemented in this Alpha:

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
- a hosted CPU code-only Alpha 2 candidate and signed-tag dispatcher. It packages ACT, Diffusion
  and SmolVLA public-API conformance without accessing weights. The original license/GPU contracts
  remain fail-closed in the separate v3.1 validation track.

The fake LIBERO fixture tests data shape and lifecycle only. This branch does not download real
LeRobot or LIBERO data, download or train SmolVLA weights, or establish language, image, state-route,
PushT, LIBERO, ACT, or Diffusion performance.

- [Direction](v3_overview.md)
- [Roadmap](roadmap.md)
- [Compatibility](compatibility.md)
- [Capability matrix](capability_matrix.md)
- [Alpha 2 release process](alpha2_release_process.md)
- [SmolVLA runner qualification](smolvla_runner_qualification.md)
- [Machine-readable public contract](public_api_contract.json)
