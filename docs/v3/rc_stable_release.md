# v3 RC and stable release boundary

This document defines the release boundary. The signed `v3.0.0-rc.1` prerelease has passed its
hosted evidence and asset gates; the stable tag does not yet exist.

The RC branch uses Python package version `3.0.0rc1`. `RcReleaseCandidateV1` binds the actual RC
merge SHA, all three evidence manifests, the verified Portfolio, public API and migration
descriptors, required checks, wheel/sdist, SBOM, provenance, privacy scan and exact checksums. The
asset verifier rejects missing, extra, symlinked or byte-modified files and independently parses
`SHA256SUMS` rather than trusting a precomputed success flag.

## Frozen candidate sequence

1. Preserve the signed `v3.0.0-alpha.3` code-only release and merged Beta 1 diagnostics.
2. Complete the CPU profiler and three deterministic teaching-fixture evidence studies.
3. Treat real PushT/LIBERO connectivity as a separately verified, non-blocking supplement. It is
   never part of the v3.0 stable performance matrix and cannot open a task-performance claim.
4. Freeze the public API descriptor and publish `v3.0.0-rc.1` / package `3.0.0rc1` only after its
   package, migration, SBOM, provenance, checksum and signature gates pass.
5. Merge the protected v3 candidate to `main`, rerun all stable evidence on the actual merge SHA,
   and publish tag `v3.0.0` / package `3.0.0` only after independent verification.

The distribution is not published to PyPI. GitHub Release assets must not contain weights,
checkpoints, caches, videos, raw datasets, private paths or credentials.

## Predeclared stable matrices

The machine-readable designs in `configs/v3/` are immutable inputs to the future release run:

| Study | Exact matrix | Rows |
| --- | --- | ---: |
| Fixture policy ladder | ACT + Diffusion × 5 train seeds × 20 episodes | 200 |
| Fixture state routes | ACT × 5 seeds × 3 instruction-dependent tasks × 10 episodes × 4 routes | 600 |
| Fixture prompt interventions | ACT × 5 seeds × 3 tasks × 10 episodes × 5 arms | 750 |

All designs use data seed 42, analysis seed 202701, 10,000 bootstrap samples and seed 11 as a
repeat sentinel. The exact total is 1,550 rows. A missing row, mixed source, failed sentinel or altered matrix closes release
eligibility. Negative or neutral results remain publishable after complete verification; they do
not become positive claims.

The seed 11 sentinel binds the deterministic seed-11 row subset, checkpoint and behavior metrics.
Environment-specific latency and memory remain in the separate `PolicyProfileManifestV1`; they are
not mixed into the bit-exact stable-evidence inventory.
Checkpoint bytes are hashed during the source/repeat comparison and then deleted. The v3.0
evidence bundle keeps only the semantic inventory and small manifests; it does not publish model
or optimizer checkpoints.

Each study is executed from a clean checkout and verified independently:

```bash
lunavla-v3 stable-run configs/v3/stable_pusht_policy_design.yaml --out outputs/v3/stable
lunavla-v3 stable-run configs/v3/stable_libero_route_design.yaml --out outputs/v3/stable
lunavla-v3 stable-run configs/v3/stable_libero_prompt_design.yaml --out outputs/v3/stable
lunavla-v3 stable-verify outputs/v3/stable/fixture_policy_ladder
lunavla-v3 stable-verify outputs/v3/stable/fixture_state_routes
lunavla-v3 stable-verify outputs/v3/stable/fixture_prompt_interventions
```

## External gates kept separate

SmolVLA weight validation and GPU evidence move to v3.1. v3.0 keeps only the public-API conformance
adapter with `NOASSERTION / unverified`, pretrained disabled and no derivative checkpoint. The
LIBERO environment asset license is also independently unverified, so its runtime path remains
fail-closed. Neither external condition may be replaced by a fixture, user permission, code
license or download availability. These conditions can block their own supplemental reports, but
they do not block the fixture-only v3.0 CPU teaching release.
