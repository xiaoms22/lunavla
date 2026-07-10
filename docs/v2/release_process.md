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

## Publishing boundary

Evidence completion does not itself create a tag or release. A maintainer verifies the workflow
and provenance URLs, confirms the package version/tag mapping, creates a GitHub-verified signed
tag, uploads the exact checked assets to a draft prerelease, and only then publishes it. LunaVLA
does not upload v2 artifacts to PyPI in this release train.

## Beta integration gate

`v2.0.0-beta.1` additionally requires the nightly/manual LeRobot integration to pass on the exact
candidate SHA. Its schema-1 manifest pins the official dataset revision, records the three source
hashes, checks all 161 frames, completes one bounded Transformer optimizer step, and closes a
headless `gym_pusht/PushT-v0` smoke. Only that JSON manifest is uploaded; dataset/video caches are
never release assets. Network failure does not block pull requests, but a successful rerun on the
same candidate SHA is mandatory before the beta prerelease. The smoke carries no PushT performance
claim.
