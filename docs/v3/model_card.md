# LunaVLA v3 model card (pre-RC)

LunaVLA is an educational and diagnostic framework, not a production robot controller. The v3
policy ladder contains native `act_v3`, a pinned public-API LeRobot Diffusion adapter and a
conformance-only SmolVLA adapter.

ACT and Diffusion have deterministic fixture training, checkpoint and resume tests. Those tests
establish software behavior, not task superiority. SmolVLA pretrained execution remains disabled
because the fixed upstream weight has no verified license in the public model repository. No
SmolVLA weight or derivative checkpoint is distributed by LunaVLA.

The future stable studies report all planned cells, failures and confidence intervals. A success
threshold is not a release requirement. Claims are generated only from verified evidence and stay
closed when intervals, task strata or provenance gates fail.

Do not use these policies for safety-critical control, autonomous deployment or decisions affecting
people. The project does not validate real-robot safety, real-time behavior, collision avoidance or
recovery from distribution shift.

