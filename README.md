# LunaVLA

![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)
![Status](https://img.shields.io/badge/status-v1.1%20development-yellow)

LunaVLA is a small, CPU-runnable imitation-learning/visuomotor-agent teaching core for people preparing to study Vision-Language-Action systems. It provides a complete state-to-action exercise: generate demonstrations, train a NumPy policy, evaluate rollouts, and inspect reproducibility evidence.

The current task is `pusht_style_point_reach`: a synthetic 2D point-reach exercise inspired by the shape of a PushT learning loop. It has no images, no T-block physics, no Transformer, and no real-robot interface. LunaVLA is therefore not a PushT benchmark or a production VLA model.

## Quick start

Requirements: Python 3.10–3.12 and NumPy. PyTorch and CUDA are not required by the v1.x teaching core.

```bash
git clone https://github.com/xiaoms22/lunavla.git
cd lunavla
python -m pip install -r requirements.txt
python scripts/run_cpu_smoke.py
```

The quickstart writes local artifacts under `outputs/`, which is intentionally not a source of published claims. A completed run should include its resolved config, checkpoint, metrics, and `manifest.json`. Use `--overwrite` only when intentionally replacing a local run.

## What is implemented

- `numpy_linear_chunk`: a linear NumPy policy that predicts fixed-size action chunks.
- `numpy_bc_mlp`: a small NumPy behavior-cloning MLP.
- A synthetic, state-based `pusht_style_point_reach` generator plus JSONL loading.
- Episode-level train/validation/test splits and configuration-driven evaluation.
- Receding-horizon and open-loop-chunk execution modes.
- Versioned configuration, checkpoint, data-record, action-chunk, and run-manifest contracts.

The legacy names `act` and `pusht_mock` remain temporary compatibility aliases for v1.1. They do not mean that this repository implements the ACT Transformer or the real PushT environment.

## Verified v1.1 results

<!-- VERIFIED_RESULTS_START -->
No controlled v1.1 result snapshot has been published yet. Historical v1.0 numbers were produced by non-controlled runs and are not evidence that action chunking caused an improvement.
<!-- VERIFIED_RESULTS_END -->

This section is generated from `results/v1.1/index.json` and the referenced manifests:

```bash
python scripts/render_readme_results.py --check
```

Only results with matching artifact hashes and the planned controlled design—one changed factor, five training seeds, and twenty fixed evaluation episodes per seed—may be described as controlled evidence. A 100% success rate is not a release requirement.

## Reproducibility and evidence

Local runs belong in `outputs/` and may contain large or transient artifacts. Small, reviewable evidence snapshots belong in [`results/v1.1/`](results/v1.1/README.md). Each snapshot records:

- the Git commit, resolved config and SHA-256;
- dataset, split, training seeds, and evaluation seeds;
- Python and dependency versions;
- policy and task identifiers;
- checkpoint and metrics SHA-256 values;
- the command, final metrics, and representative success/failure rollouts.

Build a snapshot from completed manifests with:

```bash
python scripts/build_v11_evidence_snapshot.py \
  --runs outputs/controlled_v11/<family>/<treatment>/<seed-run> [...] \
  --analysis-root outputs/controlled_v11 \
  --overwrite
python scripts/render_readme_results.py
```

The full evidence bundle, SBOM, and `SHA256SUMS` are release assets rather than tracked training output. See [`docs/evidence.md`](docs/evidence.md).

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md): policy intent, interfaces, and limitations.
- [`DATA_CARD.md`](DATA_CARD.md): generated-data schema, splits, and limitations.
- [`ROADMAP.md`](ROADMAP.md): v1.x maintenance and the gated v2 bridge.
- [`CHANGELOG.md`](CHANGELOG.md): user-visible changes.
- [`docs/evaluation.md`](docs/evaluation.md): rollout metrics and interpretation.
- [`docs/controlled_experiments.md`](docs/controlled_experiments.md): predeclared experiment design.
- [`docs/archive/v1.0/`](docs/archive/v1.0/README.md): correction-before historical material, retained for provenance only.

## Project layout

```text
configs/       versioned experiment configuration
dataset/       synthetic and JSONL data paths
model/         NumPy policy implementations and interfaces
trainer/       training entry points
scripts/       checks, experiments, and evidence tooling
results/v1.1/  small hash-verifiable evidence snapshots
outputs/       local generated artifacts (not published evidence)
```

## Claim boundary

Safe description:

> LunaVLA is a CPU-runnable teaching repository for a synthetic state-to-action imitation-learning loop with action-chunk experiments and reproducibility manifests.

Do not describe v1.x as a visual VLA, an ACT implementation, a real PushT result, or a real-robot deployment. Language is represented only by small deterministic features in the current core; instruction-following has not been established by counterfactual evaluation.

## License

Licensed under the [Apache License 2.0](LICENSE).
