# LunaVLA v3 Alpha 1

LunaVLA v3 Alpha 1 is a contract and data-foundation release candidate. It adds an isolated v3
namespace while preserving the frozen v2 public API and the lightweight v1.x path.

Implemented in this Alpha:

- named multimodal feature and embodiment contracts;
- immutable `ObservationV3`, `TransitionV3`, and ordered episode records;
- strict ExperimentConfig schema 3 and explicit v2→v3 migration;
- deterministic fake PushT and fake LIBERO fixtures for CPU data/engine tests;
- dataset QA, replay, strict checkpoint envelope, RunManifest schema 4, and tamper verification;
- `lunavla-v3` config, audit, replay, run, verify, and migration commands.

The fake LIBERO fixture tests data shape and lifecycle only. Alpha 1 does not download real LeRobot
or LIBERO data, implement Diffusion/SmolVLA, or establish language, image, state-route, PushT, or
LIBERO performance.

- [Direction](v3_overview.md)
- [Roadmap](roadmap.md)
- [Compatibility](compatibility.md)
- [Capability matrix](capability_matrix.md)
- [Machine-readable public contract](public_api_contract.json)
