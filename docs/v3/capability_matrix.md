# v3 Beta 1 draft capability matrix

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
| Real PushT/LIBERO subset | Planned Beta 2 | No benchmark statement |
| Prompt/state diagnostics | Canonical train/eval/deploy renderer, four routes, five prompt arms, typed donor/pair hashes | 40 reduced NumPy pairs only; no modality or routing claim |
| Image diagnostic smoke | ACT control/image-shuffle, step-wise image donors, four synthetic thumbnails | Four reduced pairs; no visual-contribution claim |
| Diagnostic evidence | EvidenceManifest v2, normalized cell contracts, gate reasons, per-pair CSV, verified HTML | `claim_allowed=false`; dirty/mixed inputs are inspectable but not release-eligible |

An input field or working adapter does not establish that the corresponding modality improves
behavior. Such claims remain closed until a predeclared paired multi-seed design passes.
