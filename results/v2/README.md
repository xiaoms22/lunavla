# LunaVLA v2 evidence snapshots

This directory is reserved for small snapshots produced by `lunavla-v2 evidence-snapshot` after `evidence-verify` succeeds. Full checkpoints, complete rollouts, caches, and training output stay under ignored `outputs/` paths or in GitHub Release evidence bundles.

The tracked [`language-alpha2`](language-alpha2/) directory is the review-sized snapshot of the full language study. Its claim gate is closed: **Instruction-following has not yet been established.** No visual evidence snapshot is tracked here yet.

Its authority is the clean source commit
[`a546695`](https://github.com/xiaoms22/lunavla/commit/a546695445f6fa6e717cd560d5acf718e037940a)
and [workflow run 29106885353](https://github.com/xiaoms22/lunavla/actions/runs/29106885353).
The registered EvidenceManifest SHA-256 is
`106ea2421d37c6c374e31d01a788101e358317f76b6abc315318634e6c6fa3b8`.

The root README table is not maintained by hand. It is rendered from `language-alpha2/evidence_manifest.json` only after `snapshot_manifest.json` has passed read-only verification:

```bash
python scripts/render_readme_results.py --check
```

The verifier rejects missing or unlisted files, any registry or listed-file hash mismatch, unsafe
paths or symlinks, a reduced or dirty design, mixed source provenance, artifact/config/fixture/repeat
inconsistency, and any statistics or claim result that differs from recomputation over the paired
rows. Snapshot verification establishes integrity and faithful aggregation; it does not establish
a modality contribution.
