# v3 RC and stable release boundary

This is a release contract, not a release announcement. Alpha 3 is the current code-only
prerelease; no RC or stable tag exists.

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

## External gates kept separate

SmolVLA weight validation and GPU evidence move to v3.1. v3.0 keeps only the public-API conformance
adapter with `NOASSERTION / unverified`, pretrained disabled and no derivative checkpoint. The
LIBERO environment asset license is also independently unverified, so its runtime path remains
fail-closed. Neither external condition may be replaced by a fixture, user permission, code
license or download availability. These conditions can block their own supplemental reports, but
they do not block the fixture-only v3.0 CPU teaching release.
