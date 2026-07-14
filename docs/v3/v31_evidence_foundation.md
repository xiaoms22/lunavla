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

Real SmolVLM2 cache generation, the full multi-seed executor, clustered
bootstrap aggregation, seed-11 repeat sentinel, and Trace Lab publishing remain
separate later gates.
