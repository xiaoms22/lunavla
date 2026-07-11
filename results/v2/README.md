# LunaVLA v2 evidence snapshots

This directory is reserved for small snapshots produced by `lunavla-v2 evidence-snapshot` after `evidence-verify` succeeds. Full checkpoints, complete rollouts, caches, and training output stay under ignored `outputs/` paths or in GitHub Release evidence bundles.

The tracked [`language-alpha2`](language-alpha2/) and [`visual-beta1`](visual-beta1/) directories are review-sized snapshots of the full studies. Both claim gates are closed: **Instruction-following has not yet been established**, and **visual-control contribution has not yet been established.**

Its authority is the clean source commit
[`a546695`](https://github.com/xiaoms22/lunavla/commit/a546695445f6fa6e717cd560d5acf718e037940a)
and [workflow run 29106885353](https://github.com/xiaoms22/lunavla/actions/runs/29106885353).
The registered EvidenceManifest SHA-256 is
`106ea2421d37c6c374e31d01a788101e358317f76b6abc315318634e6c6fa3b8`.

The visual snapshot is bound to clean source commit
[`bf0e550`](https://github.com/xiaoms22/lunavla/commit/bf0e550a7aa3fb0bb07354cd7cb525752c56268d)
and [workflow run 29110701437](https://github.com/xiaoms22/lunavla/actions/runs/29110701437).
Its registered EvidenceManifest SHA-256 is
`d8ff8c798a6810a09a2905dbafd6f5259ac2356623ee6060d335d660db6e9056`.

The root README tables are not maintained by hand. They are rendered from both registered EvidenceManifests only after each `snapshot_manifest.json` has passed read-only verification:

```bash
python scripts/render_readme_results.py --check
```

The verifier rejects missing or unlisted files, any registry or listed-file hash mismatch, unsafe
paths or symlinks, a reduced or dirty design, mixed source provenance, artifact/config/fixture/repeat
inconsistency, and any statistics or claim result that differs from recomputation over the paired
rows. Snapshot verification establishes integrity and faithful aggregation; it does not establish
a modality contribution.
