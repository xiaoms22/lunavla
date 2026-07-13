# v3.1 frozen VLM feature path

LunaVLA v3.1 adds a fail-closed path for generating frozen visual-language
features for the deterministic teaching task suite. It does not download model
weights automatically and does not place weights or generated caches in Git or
release assets.

## Fixed backends and evidence roles

- SmolVLM2 uses `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Its cache is the planned
  claim-bearing v3.1 path, subject to the later controlled evidence gate.
- Qwen3-VL uses `Qwen/Qwen3-VL-2B-Instruct` at revision
  `e2378df056d88153dc44616229fa371fcb87e236`. It is limited to a 12-row
  observational connectivity smoke and cannot open a performance claim.

Both paths use the final hidden layer and attention-mask mean pooling. The VLM
is frozen, unquantized, and not fine-tuned.

## Local-only workflow

The caller first creates a `VLMBackendSpecV1` JSON document containing the
complete relative file inventory, byte count, immutable revision, and verified
license-evidence hash. LunaVLA then verifies an absolute local model directory:

```console
lunavla-v3 vlm-preflight backend-spec.json --model-root /absolute/model/snapshot
```

Missing, extra, or modified files fail before Transformers is imported. Model
loading uses `local_files_only=True` and public Transformers APIs.

The SmolVLM2 cache command uses an atomic sibling staging directory and refuses
to overwrite an existing cache unless explicitly requested:

```console
lunavla-v3 vlm-cache backend-spec.json \
  --model-root /absolute/model/snapshot \
  --out /absolute/cache/root \
  --processor-sha256 <sha256> \
  --device-environment-sha256 <sha256>
lunavla-v3 vlm-cache-verify /absolute/cache/root
```

Every feature is stored with a manifest binding the backend, processor, prompt,
image, typed sample identity, task, split, stratum, device environment, pooling,
shape, and feature bytes. Verification independently recomputes the inventory,
hashes, finite-value checks, split counts, and identity order.

The Qwen path writes only an observational manifest:

```console
lunavla-v3 qwen-observational-smoke backend-spec.json \
  --model-root /absolute/model/snapshot \
  --out /absolute/output/qwen-smoke.json
```

Its fixed matrix is three tasks by two held-out strata by two episodes, exactly
12 rows. `observational=true` and `claim_allowed=false` are contract invariants.

## CI boundary

CPU CI exercises the same cache and verification code with a named deterministic
fixture extractor. Fixture outputs prove contract integrity only; they are never
represented as SmolVLM2 or Qwen model evidence. Real weight acquisition and
large-model execution remain manual, local, and outside release artifacts.
