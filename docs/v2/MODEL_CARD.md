# LunaVLA v2 release-candidate model card

v2 bridges the v1.1 NumPy baselines and an optional teaching-scale PyTorch action-chunk Transformer through one engine. The PyTorch implementation includes learned action queries, a conditional variational latent path with KL loss, padded-target masks, and temporal ensembling. The short registry alias `act` refers only to that implementation; it does not imply reproduction of a paper, benchmark score, or robot result.

Inputs use `Observation(state, instruction, image)`. Individual policies must declare and validate supported shape, dtype, and device combinations. Outputs use the same masked `ActionChunk` contract as v1.1.

The experimental image path uses a small CoordConv encoder: normalized `x` and `y` channels are appended to the pixels before convolution and global pooling. This makes target position representable without changing the canonical Transformer width, layer count, latent size, or action chunk. It is an architectural inductive bias, not evidence that images improve control; visual claims remain gated by paired occlusion, shuffle, and state-only experiments.

PyTorch checkpoint schema 3 identifies this encoder as `coordconv_xy_v1` and restores its parameters exactly. Visual schema 2 checkpoints predate the coordinate channels and are rejected explicitly rather than being loaded into a different first-convolution shape. Retraining is required for those experimental visual alpha checkpoints; state-only schema 2 checkpoints remain readable.

## RC artifact contract

The machine-readable golden descriptor is [`artifact_contracts.json`](artifact_contracts.json). Run manifest schema 3 uses an exact root, validates every declared SHA-256 digest and the clean/dirty source invariant, and strictly checks split IDs, seeds, dependencies, commands, finite metrics, and evidence hashes. Loaded manifests are deeply read-only; `to_dict()` returns an independent mutable JSON representation. Run manifest schema 2 remains readable but cannot be written.

Transformer checkpoint schema 3 validates an exact root and complete config, exact tensor names/shapes/dtypes, finite tensor and optimizer state, and JSON-safe finite metadata on both write and read. A schema 2 state-only Transformer checkpoint can be read and re-saved as schema 3; a schema 2 visual checkpoint is rejected because its pre-CoordConv first convolution is not architecture-compatible. NumPy checkpoint schema 1 likewise requires exact root, policy, and parameter fields with positive integer dimensions and finite numeric arrays. Unversioned NumPy checkpoints are legacy read-only inputs and are re-saved only in schema 1 format.

Intended uses are interface research, tiny-batch overfit checks, controlled modality ablations, and adapter development. It is not intended for real robots, safety-critical control, production serving, or claims of general instruction following or visual reasoning.

The completed language and image studies did not open their predeclared contribution gates. Instruction-following and visual-control contribution therefore remain **not established**. Synthetic point tasks can hide perception, contact, embodiment, and distribution-shift failures. LeRobot compatibility does not transfer evidence from an upstream dataset or environment to LunaVLA.
