# v3 RC and stable release boundary

This document is a release contract, not a release announcement. The RC/stable code is staged in a
Draft stack and no v3 tag currently exists.

## Frozen candidate sequence

1. Complete and publish the separately gated `v3.0.0-alpha.2` prerelease.
2. Rebase, review and merge the Beta 1 diagnostic PR; rerun its 40-pair NumPy and four-pair ACT
   fixture studies.
3. Rebase Beta 2, obtain authoritative and secondary same-SHA integration manifests, then merge it.
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
| PushT policy ladder | 3 policies × 5 train seeds × 20 evaluation episodes | 300 |
| LIBERO state routes | 5 train seeds × 4 tasks × 10 initial states × 4 routes | 800 |
| LIBERO prompt interventions | 5 train seeds × 4 tasks × 10 initial states × 5 arms | 1,000 |

All designs use data seed 42, analysis seed 202701, 10,000 bootstrap samples and seed 11 as a
repeat sentinel. A missing row, mixed source, failed sentinel or altered matrix closes release
eligibility. Negative or neutral results remain publishable after complete verification; they do
not become positive claims.

## External blockers

The SmolVLA weight remains `NOASSERTION / unverified`; code conformance does not authorize weight
use or redistribution. The release also requires isolated authoritative and secondary single-A100
manifests. Neither condition may be substituted with a fixture, user permission or a code license.

