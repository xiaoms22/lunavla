# Beta 2 hosted CPU PushT and LIBERO integration

Beta 2 is a hosted CPU integration candidate. It adds real public-source and simulation paths without
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

## What the hosted manual dispatcher must prove

The workflow runs on GitHub-hosted Ubuntu from the default branch and accepts only an immutable
same-SHA source ref whose dispatcher bytes match `main`. Its hash-locked Linux x64 environment
must report Torch `2.11.0+cpu`, torchvision `0.26.0+cpu`, LeRobot `0.6.0`, hf-libero `0.1.4`, and
no CUDA, NCCL or Triton packages. The workflow then:

1. validates Hub revision, file identity and download caps;
2. decodes PushT episode 0 and one deterministic episode for each LIBERO task ID 0–3;
3. performs reset, three steps and close for both environment families;
4. runs one finite ACT and Diffusion optimizer step through `EngineV3` for each task family;
5. uploads only configs, inventories, metrics, locks and integration manifests.

It never uploads videos, datasets, caches, model weights or checkpoints. It uploads and attests
only the two connectivity manifests, metrics and file-hash inventories.

## Current gate state

No same-SHA hosted CPU integration manifest has been produced for this candidate yet. The SmolVLA
weight license remains `NOASSERTION/unverified`; pretrained access remains closed and is unrelated
to the Beta 2 public-data connectivity gate.
