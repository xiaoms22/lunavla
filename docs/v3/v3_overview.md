# LunaVLA v3 direction

LunaVLA v3 extends the evidence-backed teaching engine toward public robot-learning data,
simulation, multiple policy families, and explicit language/vision/state diagnostics. The project
remains low-cost and evidence-first.

The intended learner loop is:

```text
audit public data -> map features -> train -> evaluate paired rollouts
                  -> diagnose failures -> verify evidence -> write a report
```

Alpha 1 implements the contracts, migration, deterministic fake data, CPU engine, artifacts, and
verification needed for that path. Planned later stages add pinned public PushT, a named four-task
LIBERO-Spatial subset, ACT, Diffusion Policy, and an adapter to upstream LeRobot SmolVLA.

LunaVLA does not claim production robot deployment, a new foundation model, real-robot transfer,
or full LIBERO benchmark performance.
