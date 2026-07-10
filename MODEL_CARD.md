# LunaVLA v1.1 model card

## Model family

LunaVLA v1.1 provides two teaching-scale NumPy policies:

- `numpy_linear_chunk`: a linear action-chunk predictor;
- `numpy_bc_mlp`: a small behavior-cloning multilayer perceptron.

The deprecated `act` alias resolves to `numpy_linear_chunk` for one compatibility release. It does not indicate an Action Chunking with Transformers implementation: v1.1 has no Transformer, action queries, CVAE/KL objective, or temporal ensembling.

## Intended use

- Learn an imitation-learning data → train → rollout-evaluate loop.
- Inspect chunk construction, masking, execution modes, and failure cases.
- Run small CPU experiments and practice reproducibility discipline.

## Not intended for

- Real-robot or safety-critical control.
- Claims about ACT, real PushT, visual perception, or state-of-the-art VLA systems.
- General manipulation or instruction-following claims.

## Inputs

The v1.1 task input contains a state vector `[x, y, goal_x, goal_y]` and optional deterministic instruction features. It contains no image pixels. The instruction features are a compact teaching device; their contribution has not been demonstrated with mask, shuffle, paraphrase, and counterfactual tests.

## Outputs

`predict_chunk()` returns an `ActionChunk`:

- `values`: finite floating-point actions with shape `[chunk_size, action_dim]`;
- `valid_mask`: a Boolean vector with shape `[chunk_size]`.

Padding is excluded from the training loss. The deprecated `predict_action()` adapter returns only the first action.

## Training and checkpoints

Both policies optimize mean squared error using NumPy on CPU. A new checkpoint is `checkpoint.json` with an explicit `schema_version`, policy type, dimensions, parameters, and metadata. Legacy JSON payloads named `checkpoint.pt` are read-only compatibility inputs; no PyTorch serialization is involved.

## Evaluation

The evaluator supports:

- `receding_horizon`: predict at every environment step and execute the first action;
- `open_loop_chunk`: predict once and execute each valid action in the chunk before replanning.

Reported behavioral metrics include success rate, final distance, rollout length, action smoothness, and categorized failures. Controlled comparisons use fixed evaluation starts, Wilson intervals for success, and paired bootstrap intervals for continuous metrics.

## Limitations and risks

- Synthetic state observations can hide perception and embodiment problems.
- The point-reach dynamics are not PushT physics.
- Small generated datasets make conclusions sensitive to seeds and configuration.
- Performance on this task does not imply transfer to visual input, LeRobot, or hardware.

Every publishable metric must link to a v1.1 manifest and verifiable artifact hashes. Historical v1.0 results are retained only as uncorrected provenance.
