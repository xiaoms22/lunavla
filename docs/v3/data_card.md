# LunaVLA v3 data card (pre-RC)

v3 pins two public data paths: `lerobot/pusht` at the recorded immutable revision and
`lerobot/libero` at the recorded immutable revision, limited to LIBERO-Spatial task IDs 0–3.
PushT integration selects episode 0. LIBERO integration selects the minimum episode index for each
of the four named tasks. Source inventories, selected files and decoded shapes are hashed.

Feature schemas declare camera order, state/action shape and dtype, units, frame, control rate and
normalization. Statistics are fit only on the training split. The policy receives a newly mapped
`ObservationV3`, never an upstream object or hidden metadata.

The repository stores only small manifests, metrics and synthetic fixtures. It does not publish raw
videos, data caches, model weights or complete checkpoints. The four LIBERO tasks are a diagnostic
subset and must not be described as the complete LIBERO benchmark.

