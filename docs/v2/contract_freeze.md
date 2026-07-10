# v2.0.0 RC contract freeze

`v2.0.0-rc.1` freezes the public LunaVLA v2 boundary shown below. The RC may receive bug fixes before v2.0.0, but a change to a frozen signature, required field, serialized meaning, or compatibility rule requires an explicit schema or release-version change.

| Contract | Writable schema | Compatibility input | Golden descriptor |
| --- | ---: | --- | --- |
| `Observation`, `VLAPolicy`, `TaskEnv`, `DatasetSource`, `Transition`, `PolicyBatch` | API descriptor 1 | Exact signatures and ownership rules | [`public_api_contract.json`](public_api_contract.json) |
| `ExperimentConfig` | 2 | v1.1 mapping through the strict migration layer | [`contracts/config-design-schema.json`](contracts/config-design-schema.json) |
| `EvidenceDesign` | 1 | None | [`contracts/config-design-schema.json`](contracts/config-design-schema.json) |
| `RunManifest` | 3 | Schema 2, read-only | [`artifact_contracts.json`](artifact_contracts.json) |
| Transformer checkpoint | 3 | Schema 2 state-only, read-only; schema 2 visual rejected | [`artifact_contracts.json`](artifact_contracts.json) |
| NumPy checkpoint | 1 | Strict unversioned legacy JSON, read-only | [`artifact_contracts.json`](artifact_contracts.json) |
| `EvidenceManifest` | 1 | None | [`evidence_contract.md`](evidence_contract.md) |

## Runtime semantics

- Public arrays are copied on construction, normalized, exposed read-only, and compared by shape, dtype, and value.
- `Transition.info` and resolved configuration/manifest trees are detached and recursively read-only. Their `to_dict()` methods return ordinary mutable JSON-compatible copies.
- `Engine.evaluate()` owns the supplied `TaskEnv` lifecycle and closes it exactly once, including exceptional paths. Callers must not reuse that environment afterward.
- NumPy policies accept CPU only. Unsupported task, modality, policy, device, or nested parameter combinations fail while parsing the configuration.

## Serialization semantics

- Current writers emit only the writable schema in the table. Read-only inputs are upgraded when saved; source files are never modified in place.
- Unknown or missing fields, boolean schema versions, invalid or non-finite values, malformed hashes, unsafe metadata, and dirty-source inconsistencies fail closed.
- Transformer schema 2 is readable only when it has no visual encoder. Old visual tensors predate `coordconv_xy_v1` and are rejected with a retraining message.
- No checkpoint, manifest, or evidence loader silently falls back to another policy or schema.

## Evidence and support boundary

The registered language and visual studies are complete, hash-verified, and reproducible, but both modality-effect claim gates remain closed. The RC therefore freezes interfaces—not claims that language or images improve behavior. Official support remains Python 3.12 on CPU Linux for v2; CUDA is an experimental manual path, and real-robot or production deployment is out of scope.

The stable release must preserve these contracts, pass v1.1 migration/compatibility tests, run the full post-merge evidence and real LeRobot gates on the actual `main` merge SHA, and publish matching signed assets, SBOM, provenance, and checksums.
