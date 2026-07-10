# v2 release process

The default branch owns a guarded manual dispatcher because GitHub discovers
`workflow_dispatch` entry points from the default branch. The dispatcher checks out an explicit
same-repository ref, verifies its immutable 40-character SHA, installs the hash-locked Linux CPU
release profile, and invokes `scripts/run_v2_release_profile.py` from that source commit.

## Alpha evidence

The `alpha` profile requires a clean checkout and executes the full v1/v2 quality suite. It then
trains and verifies the NumPy, language Transformer, rendered visual Transformer, and visual
state-only CPU examples. Every manifest must report the requested Git SHA, `git_dirty=false`, and
no source-diff hash.

The profile builds wheel and source distributions, validates their metadata, records the Python
environment, produces a reproducible CycloneDX SBOM, archives the complete run artifacts, and
writes `SHA256SUMS`. Alpha results are capability evidence only: every modality-effect claim stays
disabled.

## Controlled language and visual evidence

The `language` profile executes the exact canonical
`configs/v2/evidence/language_alpha2.yaml` matrix: five statistical training seeds, four paired
rollout arms, 480 arm-episodes, and the seed-11 reproducibility sentinel. The `vision` profile does
the same for `configs/v2/evidence/visual_beta1.yaml`: five image runs, five paired state-only runs,
480 arm-episodes, and the image seed-11 sentinel. Both profiles require the requested clean Git SHA
in every schema-3 run manifest and enforce CPU in the resolved training, policy, and runtime
records.

Each controlled profile invokes `evidence-run`, `evidence-verify`, and `evidence-snapshot`, then
verifies the full output a second time. Its separately named evidence archive contains the complete
`outputs/evidence/...` tree, the checkpoint-free `results/v2/...` review snapshot, source design,
aggregate manifest, verification record, claim summary, distributions, environment, and SBOM.
A profile-specific evidence checksum file covers every full-output, review-snapshot, metadata,
distribution, environment, and SBOM file inside the archive. Top-level `SHA256SUMS` then covers the
archive and every release-side file.

Statistical uncertainty is not a workflow failure. The release candidate copies the verified
`EvidenceManifest` decision and permitted statement verbatim: if an interval crosses zero, the
profile can still complete, while the claim remains disabled and is reported as “not yet
established.” Missing cells, non-canonical budgets, dirty or mixed source SHAs, hash mismatches,
non-CPU runs, or failed reproducibility remain hard release failures.

The language evidence package uses project version `2.0.0a2`. A later Beta release must perform a
separate version bump before publishing the visual package; running the `vision` profile alone does
not assign a Beta version.

## Publishing boundary

Evidence completion does not itself create a tag or release. A maintainer verifies the workflow
and provenance URLs, confirms the package version/tag mapping, creates a GitHub-verified signed
tag, uploads the exact checked assets to a draft prerelease, and only then publishes it. LunaVLA
does not upload v2 artifacts to PyPI in this release train.
