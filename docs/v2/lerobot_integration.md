# Pinned LeRobot and PushT integration

The beta integration is a connectivity check, not a PushT benchmark. It runs only in the
nightly/manual `V2 LeRobot Integration` workflow and is intentionally absent from the ordinary
pull-request gate so that a Hub outage cannot block unrelated changes.

## Dataset contract

The workflow accepts an exact 40-character LunaVLA commit SHA and uses this immutable upstream
selection:

- repository: `lerobot/pusht`
- revision: `b1c3ecbae7f244acc039a3dbc255a00dad1372b9`
- episode: `0`
- LeRobot dataset format: v3.0 through `LeRobotDataset`
- decoder: `pyav`
- image output: `return_uint8=true`
- preflight download ceiling: 12 MiB

Before a payload download, the integration checks the resolved Hub commit, the advertised byte
size, and the LFS SHA-256 for the upstream parquet, video, and episode metadata. It then hashes the
materialized files again. Episode 0 must decode all 161 frames as `96x96x3 uint8` images with
`float32` state/action arrays. Frame indices, terminal flags, and the terminal
`next_observation` self-boundary are validated before training is attempted.

## Bounded execution checks

Two execution checks follow the dataset validation:

1. A teaching-scale `transformer_chunk_cvae` receives two real frames and performs exactly one
   finite CPU optimizer step. At least one parameter must change.
2. `gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")` is reset headlessly, stepped three
   times with `float32` actions, checked for the expected pixels/state/action mapping, and closed in
   a `finally` boundary.

The environment smoke and the dataset episode are independent checks. They establish that the
two adapters are usable on the same source commit; they do not establish policy quality or replay
the dataset actions in the environment.

## Manifest and artifact boundary

The only uploaded artifact is `integration_manifest.json` (schema 1). It records the exact
LunaVLA Git SHA, dependency versions, upstream hashes, CPU device, dataset validation, optimizer
step, environment smoke, and `claim_allowed=false`. The workflow never uploads the Hugging Face
cache, parquet, video, decoded frames, or generated renderings.

The workflow also creates a GitHub build-provenance attestation whose subject is the exact manifest
bytes. This external attestation is the integrity boundary for runtime values such as the measured
optimizer loss; the manifest's own source SHA is not treated as a signature.

For a local diagnostic checkout, install the hash-locked Linux CPU profile and keep cache/output
paths outside the Git checkout:

```bash
uv pip sync requirements-v2-integration-cpu.lock \
  --require-hashes --strict --only-binary :all: --torch-backend cpu
uv pip install --no-deps --no-build-isolation --editable .
python scripts/run_v2_lerobot_integration.py run \
  --expected-git-sha "$(git rev-parse HEAD)" \
  --cache-dir /tmp/lunavla-lerobot-cache \
  --output /tmp/integration_manifest.json
python scripts/run_v2_lerobot_integration.py verify \
  --expected-git-sha "$(git rev-parse HEAD)" \
  /tmp/integration_manifest.json
```

Release automation must match a successful integration manifest to the exact beta source SHA.
Rerun the workflow after any merge; do not reuse a manifest produced for another commit.
