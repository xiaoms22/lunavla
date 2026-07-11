# v2 release dispatcher

GitHub exposes manual `workflow_dispatch` entry points only from the default branch. The
`V2 Release Dispatcher` therefore lives on `main`, while it checks out and executes an
explicit LunaVLA v2 source revision.

Provide all three inputs:

- `source_ref`: a branch, tag, or commit in this repository; pull-request refs are rejected, and
  the `stable` profile accepts only `main`;
- `expected_sha`: the immutable 40-character lowercase commit SHA that `source_ref` must
  resolve to;
- `profile`: `alpha`, `language`, `vision`, `rc`, or `stable`.

The dispatcher validates inputs before checkout, disables persisted Git credentials, verifies
the checked-out commit and clean worktree, installs the hashed Linux CPU release environment,
and calls exactly:

```text
python scripts/run_v2_release_profile.py --profile "$PROFILE" --expected-sha "$EXPECTED_SHA"
```

The selected revision supplies this script and all release logic. Treat `source_ref` as code
execution: review it first and copy its immutable SHA from GitHub. A moving branch that no
longer resolves to `expected_sha` fails before any source script runs.

Successful runs attest and upload `release-assets/` plus `outputs/`. The dispatcher does not
create a tag, GitHub Release, PyPI upload, or self-hosted job; those remain separate protected
release decisions.

For `stable`, the dispatcher additionally requires the requested SHA to be the fetched
`origin/main` tip. A same-workflow CPU job creates and attests the real LeRobot integration
manifest first. The evidence job verifies that cryptographic bundle and then reruns both complete
controlled studies into isolated generated paths. The stable artifact contains `release-assets/`
only because its combined evidence archive already includes the complete output and review trees.
