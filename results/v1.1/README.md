# v1.1 evidence snapshots

This directory contains small, reviewable evidence derived from validated run manifests. It intentionally excludes checkpoints, full generated datasets, and complete rollout sets.

`index.json` is the only input used to render the README result section. An empty `runs` list means no controlled v1.1 result has been published; it must never be filled with remembered or manually copied metrics.

To add a snapshot after controlled runs complete:

```bash
python scripts/build_v11_evidence_snapshot.py \
  --runs outputs/controlled_v11/<family>/<treatment>/<seed-run> [...] \
  --analysis-root outputs/controlled_v11 \
  --overwrite
python scripts/render_readme_results.py
python scripts/render_readme_results.py --check
```
