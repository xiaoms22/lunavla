# v2 controlled evidence contract

`EvidenceDesign` is the immutable declaration for a language or visual study. It fixes the base configuration, train/data/split/evaluation seeds, arms, metrics, budget, bootstrap seed, and output roots. Unknown fields fail during parsing. Claim-bearing designs additionally lock the CPU Transformer architecture and task contract.

The canonical matrix is:

| Suite | Statistical training runs | Rollout arms | Arm-episodes |
| --- | ---: | --- | ---: |
| Language | 5 | control, mask, shuffle, counterfactual | 480 |
| Visual image | 5 | control, occlusion, shuffle | 360 |
| Visual state-only | 5 | state-only | 120 |

Language and visual-image seed 11 are each repeated once as reproducibility sentinels. Their checkpoint and metrics hashes must match, but repeats do not enter the statistical matrix.

## Commands

```bash
lunavla-v2 evidence-run <design.yaml>
lunavla-v2 evidence-verify <output-root>
lunavla-v2 evidence-snapshot <output-root> --out results/v2/<design-id>
```

`evidence-run` refuses dirty source trees and existing destinations. `evidence-verify` reconstructs the matrix and statistics from hashed per-run manifests, rollouts, and paired rows. Missing runs or cells, duplicate cells, mixed Git SHAs or dependencies, changed data/splits, schema 2 manifests, failed repeat sentinels, or any hash/config/fixture mismatch make verification fail.

`--allow-reduced-design` permits smaller CI studies only. Such output is always observational and every claim stays closed. `evidence-snapshot` first performs full verification, then copies review-sized configs, manifests, metrics, aggregate rows, repeat metadata, and a few rollout samples. It never copies checkpoints.

## Published snapshot verification

Tracked result text is generated through a separate read-only gate:

```bash
python scripts/render_readme_results.py --check \
  --v2-snapshot results/v2/language-alpha2
```

The publication gate starts from a reviewed registry that pins the official workflow's source Git
SHA, raw `EvidenceManifest` SHA-256, and `snapshot_manifest.json` SHA-256. It then checks the exact
file inventory and every listed digest. Resolved configs, metrics, donor banks, repeat files,
evaluation fixtures, and rollout samples must match their source RunManifest and paired rows. The
canonical `EvidenceDesign`, plan, clean single-SHA provenance, dependency consistency, and complete
paired matrix are also revalidated. Statistics and the claim decision are recomputed from the
paired rows with the predeclared bootstrap settings and must exactly match `EvidenceManifest`;
`aggregate.json` must be an exact projection of the same statistics and claims.

README values are read from the validated `EvidenceManifest`, never from handwritten numbers. A
changed listed file, coordinated registry/provenance/statistics edit, stale generated block, or an
allowed claim that does not match recomputation returns a non-zero error. Passing this gate
establishes snapshot integrity only.

## Claim gates

Language requires a counterfactual final-distance degradation and a control-success advantage whose clustered paired 95% intervals both exclude zero. Visual evidence requires occlusion and state-only distance degradation, with direct-reach and waypoint-reach strata passing independently. Mask, shuffle, first-action MSE, and all non-passing intervals remain visible as auxiliary or negative evidence.

An interval crossing zero is reported as “not yet established.” A successful training run, adapter import, or 100% success rate alone never opens a claim.
