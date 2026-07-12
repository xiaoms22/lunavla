# Beta 2 bounded PushT and LIBERO integration

Beta 2 is a stacked Draft implementation. It adds real public-source and simulation paths without
opening a benchmark, modality, policy-ranking, or robot-deployment claim.

## Immutable sources

| Component | Pinned identity | Use |
| --- | --- | --- |
| LeRobot | `0.6.0@30da8e687a6dfc617fcd94afc367ac7071c376ce` | Public dataset, policy and environment APIs |
| PushT | `lerobot/pusht@b1c3ecbae7f244acc039a3dbc255a00dad1372b9` | Episode 0, 161 decoded frames, 12 MiB cap |
| LIBERO | `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` | Spatial task IDs 0–3, minimum episode per task, 384 MiB cap |
| LIBERO runtime | `hf-libero==0.1.4` | Linux x64, headless EGL, init-state ID 0 |

The runtime follows the [LeRobot LIBERO task-ID interface](https://huggingface.co/docs/lerobot/libero)
and retains provenance to the [official LIBERO project](https://github.com/Lifelong-Robot-Learning/LIBERO).
The LeRobot dataset license recorded by the source contract is Apache-2.0; original LIBERO code and
data provenance remain separately documented as MIT and CC BY 4.0.

## What PR CI proves

- revision-3 source and task configs reject drift and task/data mismatch;
- LeRobot-format frames map to ordered `ObservationV3` values without raw metadata leakage;
- two 256×256 RGB cameras, state `[8]`, action `[7]`, 10 Hz and terminal boundaries are checked;
- PushT/LIBERO wrappers close upstream resources exactly once after success or failure;
- integration output is atomic and config, source, lock, runner, metrics and manifest hashes are
  independently verified;
- fixture evidence always has `claim_allowed=false` and `benchmark_claim=false`.

These tests are deterministic and offline. They do not establish that the Hub files or MuJoCo
environment have run on the current Git SHA.

## What the manual dispatcher must prove

The workflow accepts only a same-SHA `RunnerQualificationManifestV1` on an ephemeral runner with
labels `self-hosted`, `linux`, `x64`, `gpu`, `lunavla-v3`. Exactly one A100 must be visible through
both `nvidia-smi` and PyTorch. The workflow then:

1. validates Hub revision, file identity and download caps;
2. decodes PushT episode 0 and one deterministic episode for each LIBERO task ID 0–3;
3. performs reset, three steps and close for both environment families;
4. runs one finite ACT and Diffusion optimizer step through `EngineV3` for each task family;
5. uploads only configs, inventories, metrics, locks and integration manifests.

It never uploads videos, datasets, caches, model weights or checkpoints. The authoritative cloud
and secondary development runner must produce the same Git, dependency-lock and source hashes.

## Current gate state

No qualified runner is registered and no real Beta 2 integration manifest has been produced. The
Alpha 2 SmolVLA weight license remains `NOASSERTION/unverified`; pretrained access, Alpha 2 tag,
Beta merge and all release operations remain closed.
