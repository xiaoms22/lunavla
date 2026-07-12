# LunaVLA v3 data card (pre-RC)

Stable performance evidence uses generated, deterministic LunaVLA teaching fixtures with data seed
42. Public `lerobot/pusht` and `lerobot/libero` revisions are used only by the separate connectivity
adapter Draft. They do not contribute benchmark claims or stable performance rows.

Feature schemas declare camera order, state/action shape and dtype, units, frame, control rate and
normalization. Statistics are fit only on the training split. The policy receives a newly mapped
`ObservationV3`, never an upstream object or hidden metadata.

The repository stores only small manifests, metrics and synthetic fixtures. It does not publish raw
videos, data caches, model weights or complete checkpoints. Any future four-task LIBERO result is a
diagnostic subset and must not be described as the complete LIBERO benchmark.
