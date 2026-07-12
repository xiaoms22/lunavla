# v3 Beta 2 hosted CPU candidate capability matrix

| Area | Current status | Boundary |
| --- | --- | --- |
| Feature/config contracts | Implemented and CPU-tested | No real embodiment mapping yet |
| v2 migration | Implemented and golden-tested | Unknown physical metadata remains explicit |
| Data QA/replay | Implemented on deterministic fixtures | No real dataset download |
| Policy/normalization contracts | Implemented and CPU-tested | SmolVLA weights remain gated |
| Engine/checkpoint/manifest | Revision 2 policy artifacts plus revision 3 diagnostic runs | No policy comparison |
| Native ACT | CPU E2E, masked loss, checkpoint/resume, temporal ensemble | No performance claim |
| LeRobot Diffusion adapter | Public APIs, DDIM, processors, masked loss, exact CPU resume | Fake-data chain only; no performance claim |
| Fake PushT/LIBERO | Implemented as fixtures | Connectivity and lifecycle only |
| SmolVLA adapter | Public-API conformance fixture and pinned model identity | Pretrained, optimizer, resume, and inference gates closed |
| Alpha 3 release supply chain | Published hosted CPU code-only prerelease, locked build backend, signed tag, SBOM, provenance and checksums | No weight access, performance claim or PyPI publication |
| SmolVLA v3.1 validation | Strict license/GPU contracts and CUDA lock retained fail-closed | No runner, license review, weight access or validation release |
| Real PushT/LIBERO subset | Pinned contracts, adapters, CLI and hosted CPU dispatcher implemented; offline fixtures pass | Same-SHA networked manifest pending; no benchmark statement |
| Bounded integration evidence | Atomic IntegrationManifest v1 and independent hash verification | `claim_allowed=false`, `benchmark_claim=false`; real workflow pending |
| Prompt/state diagnostics | Canonical train/eval/deploy renderer, four routes, five prompt arms, typed donor/pair hashes | 40 reduced NumPy pairs only; no modality or routing claim |
| Image diagnostic smoke | ACT control/image-shuffle, step-wise image donors, four synthetic thumbnails | Four reduced pairs; no visual-contribution claim |
| Diagnostic evidence | EvidenceManifest v2, normalized cell contracts, gate reasons, per-pair CSV, verified HTML | `claim_allowed=false`; dirty/mixed inputs are inspectable but not release-eligible |

An input field or working adapter does not establish that the corresponding modality improves
behavior. Such claims remain closed until a predeclared paired multi-seed design passes.
