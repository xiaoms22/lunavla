# Verified learner portfolio export

`PortfolioBundleV1` converts only the three complete, independently verified CPU teaching-evidence
studies into a small learner-facing bundle. It does not accept reduced, dirty, mixed-provenance,
claim-closed, or incomplete evidence.

The frozen input is exactly 1,550 rows:

- 200 `fixture_policy_ladder` rows;
- 600 `fixture_state_routes` rows;
- 750 `fixture_prompt_interventions` rows.

Build and verify a bundle with:

```console
lunavla-v3 portfolio-build outputs/stable-evidence --out outputs/portfolio
lunavla-v3 portfolio-verify outputs/portfolio
```

The builder independently verifies every design, row inventory, repeat sentinel and summary before
export. It then scans both source evidence and generated Markdown, HTML and JSON for private absolute
paths, internal project identifiers, internal URLs, credentials and contact email addresses. Output
is written through a sibling staging directory and replaced atomically only after verification.

The capability sentence and limitations are fixed by the public contract. Results are limited to
deterministic LunaVLA teaching fixtures and cannot be described as PushT, LIBERO, real-robot, GPU,
production, or SmolVLA-weight evidence.
