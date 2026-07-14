# v3.1 three-task evidence foundation

LunaVLA v3.1 now exercises `direct_pick_place`, `waypoint_sequence`, and
`failure_recovery` through the same `EngineV3`, policy registry, checkpoint,
restore, and run verifier used by the existing v3 vertical path. The committed
quick fixture is deterministic and local; it does not download a VLM.

```bash
lunavla-v3 v31-fixture-run configs/v3/v31_fixture_frozen_cpu.yaml --overwrite
lunavla-v3 verify-run outputs/v3/v31-fixture-frozen-cpu
```

The held-out evaluation is an offline first-action connectivity check across
the composition and paraphrase strata. It is not an environment success rate
or a VLM contribution result, and its output always has
`claim_allowed=false`.

Conditioned ACT supports three cache interventions:

- `control`: use the sample's immutable cached feature;
- `feature_mask`: replace every consumed feature with an exact all-zero vector;
- `feature_shuffle`: select a deterministic, non-self, content-different donor
  from the same dataset split at every policy step.

The intervention and shuffle seed are bound into resolved config and checkpoint
metadata. Cache tampering, intervention mismatch, missing donors, and cross-split
configuration errors fail closed.

The preregistered matrix is encoded in
`configs/v3/v31_frozen_vlm_evidence.yaml`. Its dimensions are fixed to five
training seeds, four arms, three tasks, two held-out strata, and twenty paired
episodes: exactly 2,400 rows. The design contract alone cannot open a claim.

```bash
lunavla-v3 validate-v31-evidence-design \
  configs/v3/v31_frozen_vlm_evidence.yaml
```

Real SmolVLM2 cache generation, real ACT multi-seed training, and Trace Lab
publishing remain separate later gates.

## Full matrix contract workflow

The evidence workflow now executes and independently verifies all 2,400 matrix
keys, repeats the complete seed-11 slice, calculates Wilson intervals and a
train-seed-clustered paired bootstrap with 10,000 samples, and emits a hashed
`V31EvidenceManifestV1`.

```bash
lunavla-v3 v31-evidence-run \
  configs/v3/v31_frozen_vlm_evidence.yaml \
  --out outputs/v3/v31-evidence-fixture --fixture
lunavla-v3 v31-evidence-verify outputs/v3/v31-evidence-fixture
```

`--fixture` is mandatory for the built-in CPU executor. It produces the exact
matrix and reproducibility sentinel but records
`feature_source=deterministic_fixture`, so `claim_allowed` remains false even
when descriptive thresholds happen to pass, and the fixture is not release
eligible. A real evidence executor must bind
the pinned SmolVLM2 cache, ACT checkpoints, dependency lock, clean Git SHA, and
paired episode inventory before it may use `real_frozen_vlm`.

The scientific gate requires all of the following:

- control minus baseline success interval has a lower bound above zero;
- baseline minus control final-distance interval has a lower bound above zero;
- control minus feature-shuffle success interval has a lower bound above zero;
- all six task-by-stratum success non-inferiority lower bounds are at least
  `-0.05`;
- the matrix is complete and homogeneous, and the seed-11 rows, checkpoint
  inventory, and metrics inventory reproduce exactly.

Negative results remain release-eligible when their evidence is complete; the
fixed conclusion is then “冻结 VLM 特征贡献尚未建立”.
