# v2 modality evaluation protocol

Adding an input field is not evidence that a policy uses it. v2 therefore records paired interventions before allowing a language or visual claim.

## Language

The language suite contains at least three goals that share the same initial state, so the instruction is required to identify the requested target. Training templates and held-out paraphrases are disjoint. Every evaluation example can be paired with:

- an explicit `[MASK]` instruction encoded as a zero feature vector;
- an instruction shuffled from another task;
- a counterfactual instruction naming a different target.

Every intervention is applied at each rollout decision, not only to a static example. The environment target, evaluation start, training seed, evaluation seed, and shared pair ID remain fixed. Shuffled and counterfactual instructions come from a pre-generated, hashed donor bank. No instruction-following statement is allowed unless both the counterfactual final-distance degradation and the control success advantage have clustered paired 95% intervals excluding zero.

## Vision

The visual suite renders both direct-reach and waypoint-reach task families. In the claim-eligible `vision_required` mode, policy state is only `[x, y, phase]`; each episode has independently seeded hidden goal/waypoint geometry, and identical states can correspond to different targets. The goal and waypoint are available to the policy only through pixels. The historical seven-value privileged state remains diagnostic-only.

Image-bearing and state-only training paths share the same non-image transition hash. Full-image occlusion and same-family image shuffle are applied throughout each rollout; shuffle donors are pre-generated and hashed. A visual contribution requires degradation under both occlusion and the independently trained state-only baseline, with direct-reach and waypoint-reach strata passing separately.

At least one real visual dataset path through the isolated LeRobot adapter is required before v2.2 beta. A successful import or decoded frame alone is an integration result, not a claim that vision improves control.

## Reporting

Success remains a proportion with a Wilson 95% interval. Continuous contrasts and success differences use a hierarchical paired bootstrap: training seeds are resampled first, then paired evaluation episodes within seed. Run manifests use schema 3 and record structured interventions/pairs, task family, final distance, deterministic runtime state, and hashes for design, config, evaluation fixture, paired data, checkpoint, and dependencies. Schema 2 remains read-only and cannot support controlled evidence.
