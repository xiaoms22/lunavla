# Evidence and release contract

## Local run manifest

Every completed training/evaluation run produces `manifest.json`. The manifest is a versioned record of source revision, resolved configuration, inputs, seeds, runtime, artifact hashes, command, and final metrics. Paths are repository-relative when possible; hashes, not workstation paths, identify evidence.

`scripts/create_run_manifest.py` can rebuild or validate a manifest after the core train/eval commands finish. Validation fails when a declared artifact is absent or its SHA-256 differs.

## Tracked evidence snapshot

`scripts/build_v11_evidence_snapshot.py` accepts completed run directories. It validates every manifest and copies only:

- manifest and resolved configuration;
- compact JSON/CSV metrics;
- one deterministic success rollout and one deterministic failure rollout when available.

It never copies checkpoints or the complete generated dataset. The snapshot index records each copied file and hash. `scripts/render_readme_results.py` renders the README table strictly from that index and its manifests.

For a controlled matrix, pass `--analysis-root outputs/controlled_v11`. The snapshot then also verifies and copies the predeclared design, per-episode CSV, treatment aggregates, Wilson intervals, and paired-bootstrap contrasts. README rows are rendered from those 5-seed aggregates rather than from individual seed runs.

## Release evidence bundle

The nightly/manual evidence workflow may package full manifests, metrics, representative rollouts, SBOM, and `SHA256SUMS`. Checkpoints and large output remain release assets, not Git-tracked files. A tag is eligible only after both the fast CI suite and full evidence workflow pass for the same commit.

## Claim policy

- A local output without a valid manifest is diagnostic material, not publishable evidence.
- A historical v1.0 report is provenance, not v1.1 evidence.
- A comparison is called controlled only when its predeclared invariant checks pass.
- Success uses a 95% Wilson interval; paired continuous differences use a seeded paired-bootstrap 95% interval.
- “Improves” is permitted only when the relevant interval supports the directional claim.
