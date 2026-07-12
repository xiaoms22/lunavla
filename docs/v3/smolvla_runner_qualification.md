# SmolVLA single-A100 runner qualification

Runner qualification is independent from the model-weight license gate. It proves that an isolated
single-A100 environment can execute the locked CUDA profile; it does not download weights, enable
pretrained loading, authorize a release, or establish a scientific claim.

## Fixed roles and requirements

- `authoritative`: an isolated Volcengine A100 runner used for future release evidence.
- `secondary`: an isolated one-GPU slice of the eight-A100 development host used only to repeat the
  qualification.
- Linux x86_64, Python 3.12, NVIDIA A100, driver `>=570.26`, Torch `2.11.0+cu128`, torchvision
  `0.26.0+cu128`, at least 16 GiB memory and 30 GiB free temporary storage.
- `nvidia-smi` and `torch.cuda.device_count()` must each see exactly one GPU.
- The runner must have `self-hosted`, `linux`, `x64`, `gpu`, and `lunavla-v3` labels.

The multi-GPU host must isolate one GPU at the container boundary. `CUDA_VISIBLE_DEVICES` alone is
insufficient because the gate also counts devices reported by `nvidia-smi`.

## Ephemeral runner launch

Use a fresh Linux container image pinned by digest. The image must contain the GitHub Actions runner
and the host must expose exactly one A100 with the NVIDIA container runtime. Do not mount host home
directories, repositories, SSH material, cloud credentials, datasets, checkpoints, or model caches.

Configure the runner inside the container with a short-lived repository registration token:

```bash
./config.sh \
  --url https://github.com/xiaoms22/lunavla \
  --token "$GITHUB_RUNNER_TOKEN" \
  --name "$ONE_TIME_RUNNER_NAME" \
  --labels gpu,lunavla-v3 \
  --unattended \
  --ephemeral
LUNAVLA_EPHEMERAL_RUNNER=true ./run.sh
```

The container launch must also export
`RUNNER_LABELS=self-hosted,linux,x64,gpu,lunavla-v3` and
`LUNAVLA_CONTAINER_IMAGE_SHA256=<digest-without-prefix>`. The latter must match the manual workflow
input. Destroy the container after its one job; do not install it as a persistent service on either
host.

## Manual workflow

Run **V3 Alpha 2 Release Dispatcher** from the default branch with:

- `phase=preflight`;
- `source_ref` set to the reviewed v3 branch or commit;
- `expected_sha` set to that exact commit;
- `runner_role=authoritative` for Volcengine or `secondary` for the development host;
- `expected_container_image_sha256` set to the 64-character digest without the `sha256:` prefix;
- `enable_pretrained_gate=false`;
- no license-review hash, GPU run ID, or tag.

The workflow installs the hash-locked CUDA profile, verifies outbound TLS connectivity, rejects
private mount targets and pre-existing weight caches, then uploads and attests only
`runner-qualification-manifest.json`. The runner name and GPU UUID are stored only as SHA-256.

Both manifests must bind the same Git SHA and dependency lock. The authoritative and secondary
roles remain separate; a successful secondary run cannot replace the authoritative release run.

## License boundary

The normalized public-source observations and the fail-closed status are recorded in
`release/smolvla-license-status.json`. Project-owner permission for local evaluation is not upstream
license evidence. Until an official model-repository source explicitly licenses the fixed weights,
`LicenseReviewV1` cannot be created, the pretrained GPU phase cannot run, and Alpha 2 cannot be
tagged.
