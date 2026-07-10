# v2 modality evaluation protocol

Adding an input field is not evidence that a policy uses it. v2 therefore records paired interventions before allowing a language or visual claim.

## Language

The language suite contains at least three goals that share the same initial state, so the instruction is required to identify the requested target. Training templates and held-out paraphrases are disjoint. Every evaluation example can be paired with:

- an explicit `[MASK]` instruction encoded as a zero feature vector;
- an instruction shuffled from another task;
- a counterfactual instruction naming a different target.

The state, action target, evaluation start, and pairing key remain fixed. No instruction-following statement is allowed unless the predeclared paired 95% interval excludes zero in the beneficial direction across multiple seeds.

## Vision

The visual suite renders both direct-reach and waypoint-reach task families. Every image-bearing example has a state-only control. Image occlusion and image shuffle preserve state and action targets under a pairing key.

At least one real visual dataset path through the isolated LeRobot adapter is required before v2.2 beta. A successful import or decoded frame alone is an integration result, not a claim that vision improves control.

## Reporting

Success remains a proportion with a Wilson 95% interval. Paired continuous differences use a paired bootstrap 95% interval. Run manifests must record modality intervention, pair IDs, seeds, split, config/checkpoint/data hashes, and exact dependency versions.
