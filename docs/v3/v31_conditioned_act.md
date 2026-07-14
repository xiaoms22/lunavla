# v3.1 conditioned ACT

The v3.1 conditioned path adds exactly one 64-dimensional condition token to
the existing native `act_v3` model. It reuses the ACT instruction projection as
the only trainable `feature_dim → d_model` mapping; `d_model` is fixed to 64 for
this study.

## Matched arms

- `frozen_feature` reads the audited feature for the typed
  `(split, task, episode, step)` identity and projects it to one 64-dimensional
  token.
- `learned_null` supplies a fixed first-basis input through the same projection.
  The resulting first projection column is therefore a learned 64-dimensional
  null token.

Both arms use the same ACT state, image encoder, action queries, CVAE,
Transformer layers, optimizer, parameter count, training budget, and checkpoint
format. The null arm does not read cached feature values. This avoids changing
model capacity while isolating the contribution of frozen features.

## Fail-closed behavior

- Conditioned ACT requires config contract revision 4, a positive feature
  dimension, `instruction_dim == condition_input_dim`, and `d_model == 64`.
- Frozen features are loaded only after independent cache verification. Missing
  typed identities, changed feature bytes, wrong backend hashes, shape drift,
  NaN/Inf, or writable cache mutation fail before policy execution.
- Checkpoints bind the condition arm, 64-dimensional token contract,
  PolicySpec, and normalization hash. Frozen-feature checkpoints additionally
  bind the feature-cache index hash. A checkpoint cannot be restored under the
  other arm.
- Existing unconditioned ACT configs serialize without new fields and existing
  checkpoints remain readable.

This implementation establishes the matched training and checkpoint path. It
does not establish that frozen VLM features improve task performance; that
claim remains gated on the separately preregistered 2,400-row evidence matrix.
