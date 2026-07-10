# LunaVLA v2 experimental model card

v2 bridges the v1.1 NumPy baselines and an optional teaching-scale PyTorch action-chunk Transformer through one engine. The PyTorch implementation includes learned action queries, a conditional variational latent path with KL loss, padded-target masks, and temporal ensembling. The short registry alias `act` refers only to that implementation; it does not imply reproduction of a paper, benchmark score, or robot result.

Inputs use `Observation(state, instruction, image)`. Individual policies must declare and validate supported shape, dtype, and device combinations. Outputs use the same masked `ActionChunk` contract as v1.1.

The experimental image path uses a small CoordConv encoder: normalized `x` and `y` channels are appended to the pixels before convolution and global pooling. This makes target position representable without changing the canonical Transformer width, layer count, latent size, or action chunk. It is an architectural inductive bias, not evidence that images improve control; visual claims remain gated by paired occlusion, shuffle, and state-only experiments.

PyTorch checkpoint schema 3 identifies this encoder as `coordconv_xy_v1` and restores its parameters exactly. Schema 2 checkpoints predate the coordinate channels and are rejected explicitly rather than being loaded into a different first-convolution shape. Retraining is required for those experimental alpha checkpoints.

Intended uses are interface research, tiny-batch overfit checks, controlled modality ablations, and adapter development. It is not intended for real robots, safety-critical control, production serving, or claims of general instruction following or visual reasoning.

Language and image paths remain experimental until their paired multi-seed confidence intervals exclude zero. Synthetic point tasks can hide perception, contact, embodiment, and distribution-shift failures. LeRobot compatibility does not transfer evidence from an upstream dataset or environment to LunaVLA.
