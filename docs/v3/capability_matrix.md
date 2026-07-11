# v3 Alpha 2 contracts capability matrix

| Area | Current status | Boundary |
| --- | --- | --- |
| Feature/config contracts | Implemented and CPU-tested | No real embodiment mapping yet |
| v2 migration | Implemented and golden-tested | Unknown physical metadata remains explicit |
| Data QA/replay | Implemented on deterministic fixtures | No real dataset download |
| Policy/normalization contracts | Implemented and CPU-tested | SmolVLA remains gated |
| Engine/checkpoint/manifest | Revision 2 implemented for NumPy, ACT, and Diffusion | No policy comparison |
| Native ACT | CPU E2E, masked loss, checkpoint/resume, temporal ensemble | No performance claim |
| LeRobot Diffusion adapter | Public APIs, DDIM, processors, masked loss, exact CPU resume | Fake-data chain only; no performance claim |
| Fake PushT/LIBERO | Implemented as fixtures | Connectivity and lifecycle only |
| SmolVLA | Public-API adapter is the next Alpha 2 PR | Pretrained and optimizer gates closed |
| Real PushT/LIBERO subset | Planned Beta 2 | No benchmark statement |
| Prompt/state diagnostics | Planned Beta 1 | No modality or routing claim |

An input field or working adapter does not establish that the corresponding modality improves
behavior. Such claims remain closed until a predeclared paired multi-seed design passes.
