# LunaVLA v1.1 data card

## Dataset

LunaVLA uses synthetic demonstrations for `pusht_style_point_reach`, a 2D point moving toward a configured goal. The name describes an educational task shape; the data does not contain a T block, contact physics, images, teleoperation, or real-robot measurements.

The same record contract can be serialized as local JSONL for loader and data-quality exercises. The included clean/noisy samples remain generated teaching data and are not LeRobot datasets.

## Versioned record schema

Each `VLARecord` contains:

- `observation`: state before the action;
- `action`: expert action applied at the current timestep;
- `next_observation`: state after that action;
- `episode_id` and contiguous `timestep`;
- `terminated`: whether the transition ends the episode;
- `success`: whether the configured success condition is reached;
- `language_instruction`: optional text;
- task context and metadata aligned with the current observation.

JSONL loading rejects unknown dimensions, non-finite values, duplicate `(episode_id, timestep)` pairs, and discontinuous timesteps.

## Generation

Generation parameters—including goal, start range, action clip, instruction dimension, episode count, step count, and random seed—come from `ExperimentConfig`. Expert actions move the point toward the goal under the configured action limit.

## Splits

Train, validation, and test partitions are assigned by episode, never by individual transition. Episode identifiers must be disjoint across all splits. The manifest records the split definition and data SHA-256 so a published result can be traced to the exact records used.

## Action chunks and padding

Chunk targets are built from consecutive actions in the same episode. The final short chunk is padded and accompanied by `valid_mask`; padded values do not contribute to loss. Changing padding values must not change the masked loss.

## Data-quality experiment

The clean/noisy comparison is a separate experiment family. It holds the policy, split, training budget, and evaluation starts fixed while changing only the declared data-quality treatment. Results are aggregated over five training seeds and twenty fixed evaluation episodes per seed.

## Limitations

- Generated expert trajectories are much simpler than human demonstrations.
- No visual or language-generalization distribution is represented.
- The fixed low-dimensional dynamics do not measure real manipulation ability.
- Results are meaningful only for the declared configuration and hashes.

Recommended description:

> Used synthetic point-reach demonstrations to validate a state-to-action imitation-learning and rollout-evaluation workflow.
