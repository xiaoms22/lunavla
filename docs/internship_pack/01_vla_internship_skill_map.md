# Beginner VLA Skill Map

LunaVLA is designed for learners who know the basic VLA terms but have not yet built a runnable project. It turns common embodied AI project skills into concrete code, metrics, and artifacts.

The baseline path should let a beginner trace one sample through `Dataset -> Policy -> Train -> Eval -> Report` without needing a real robot or a large model.

| Skill | What It Means | Repo Evidence |
| --- | --- | --- |
| VLA systems | Convert observations into actions under a task intent | README schema and `dataset/pusht_dataset.py` |
| Imitation learning | Learn a policy from demonstrations | `trainer/train_act_pusht.py` |
| ACT-style action chunks | Predict short action sequences | `policy.chunk_size` configs and ablation |
| Rollout evaluation | Test behavior over time | `eval_vla.py` and generated rollouts |
| Failure analysis | Explain why a run fails | `docs/failure_taxonomy.md` |
| Project communication | Turn code into interview evidence | resume bullets, Q&A, and report template |

After completing the baseline path, a learner should be able to say:

> I implemented a small VLA-style imitation-learning loop. The dataset records observations and actions, the ACT-style policy predicts action chunks, and evaluation runs rollouts to measure success rate and inspect failures.

## What This Proves

- You can build an end-to-end observation-to-action training loop.
- You understand the difference between training loss and rollout behavior.
- You can explain data schema, policy outputs, evaluation metrics, and failure modes.

## What It Does Not Claim

- Real-robot deployment.
- State-of-the-art robotics performance.
- A large foundation model reproduction.
