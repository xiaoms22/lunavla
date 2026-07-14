# v3.1 real frozen-feature development container

This container is for local Apple Silicon development and CPU connectivity
testing. macOS MPS is not exposed inside Linux containers, so its output is not
authoritative release evidence and cannot by itself open a scientific claim.

Model files, feature caches, checkpoints, and rollouts must be mounted under
`/artifacts`; none are copied into the image or committed to Git.

```bash
docker build -f docker/v31-real-evidence/Dockerfile -t lunavla-v31-real:dev .
docker run --rm --network=none \
  -v "$PWD/outputs/v3/v31-real:/artifacts" \
  lunavla-v31-real:dev --help
```

Network access is permitted only in a separate materialization step. Extraction,
training, replay, verification, and reporting run with `--network=none` against
the pinned local model inventory.
