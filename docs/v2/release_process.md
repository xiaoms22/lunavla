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

The already-verified language and visual evidence remain bound to their original Alpha/Beta source
commits and package versions. The RC source is explicitly versioned `2.0.0rc1`; running a release
profile never assigns or changes a package version on its own.

## RC contract profile

The `rc` profile reruns the complete quality gate, read-only verifies both registered full evidence
snapshots, builds the `2.0.0rc1` wheel and sdist, and packages the three machine-readable golden
descriptors together with the contract freeze, compatibility guide, model/data cards, and security
policy. Its release candidate records every contract-file hash and the original evidence Git SHA,
EvidenceManifest hash, workflow URL, closed claim statement, distributions, environment, and SBOM.
The candidate hashes the environment and SBOM and names the RC archive without attempting to hash
an archive that contains the candidate itself. After the archive is closed,
`rc-release-integrity.json` hashes the candidate, archive, contracts, distributions, environment,
and SBOM; top-level `SHA256SUMS` then covers that post-archive integrity record and every other
release-side asset. The dispatcher provenance-attests the distributions, RC evidence archive,
`SHA256SUMS`, SBOM, candidate, and environment requirements as separate subjects in one supported
multi-path attestation action.
The RC profile freezes interfaces; it does not rerun or reinterpret modality performance.

## Stable post-merge profile

The `stable` profile accepts only `source_ref=main`, requires its immutable SHA to equal the fetched
`origin/main` tip, and requires source, installed metadata, and public API version `2.0.0`. The
dispatcher first runs the pinned real LeRobot integration in a separate CPU job at that same SHA.
It attests the strict `integration_manifest.json`, then transfers both the manifest and GitHub
attestation bundle to the evidence job. The release script cryptographically verifies the bundle
with the exact repository, signer workflow, `refs/heads/main` source ref, source digest, and
GitHub-hosted runner policy; the manifest/bundle/verification hashes and identities all enter the
candidate.

Language and visual studies are rerun from their exact canonical seeds, arms, budgets, and
bootstrap settings. Execution-only designs redirect output to ignored
`outputs/stable-release/...` and `results/v2/stable-release/...` paths. The registered
`results/v2/language-alpha2` and `results/v2/visual-beta1` trees are hashed before and after the
run and cannot be reused or overwritten. The stable matrix must contain exactly 15 training runs
and 960 arm-episodes. Each current `EvidenceManifest` claim decision and permitted statement is
copied verbatim into the stable candidate, whether its interval opens or closes the claim.

The final `lunavla-v2-stable-evidence.tar.gz` combines both complete evidence trees, review
snapshots, frozen contracts, integration provenance, distributions, environment, SBOM, and an
internal file checksum manifest. `release-candidate.json` binds the archive, package version
`2.0.0`, expected tag `v2.0.0`, integration identity, distributions, and claim decisions. A final
`SHA256SUMS` must exactly cover every release-side file. The dispatcher provenance-attests the
distributions, combined archive, candidate, environment lock snapshot, SBOM, and checksum manifest.

## Publishing boundary

Evidence completion does not itself create a tag or release. A maintainer verifies the workflow
and provenance URLs, confirms the package version/tag mapping, creates a GitHub-verified signed
tag, uploads the exact checked assets to a draft prerelease, and only then publishes it. LunaVLA
does not upload v2 artifacts to PyPI in this release train. Stable remains CPU Linux authoritative;
neither the release workflow nor candidate depends on a GPU.

## Beta integration gate

`v2.0.0-beta.1` additionally requires the nightly/manual LeRobot integration to pass on the exact
candidate SHA. Its schema-1 manifest pins the official dataset revision, records the three source
hashes, checks all 161 frames, completes one bounded Transformer optimizer step, and closes a
headless `gym_pusht/PushT-v0` smoke. Only that JSON manifest is uploaded; dataset/video caches are
never release assets. The manifest must also pass its GitHub provenance-attestation verification.
Network failure does not block pull requests, but a successful rerun on the same candidate SHA is
mandatory before the beta prerelease. The smoke carries no PushT performance claim.

Before publishing `v2.0.0-rc.1`, the same real integration smoke is rerun on the exact RC commit.
Stable release evidence is built again only after the protected `v2` merge reaches `main`; no
pre-merge RC artifact may substitute for that post-merge stable run.
