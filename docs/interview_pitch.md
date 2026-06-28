# Two-Minute Interview Pitch

I built MiniMind-VLA as a small but complete VLA internship project starter. The goal was to make the core embodied-learning loop easy to run and explain: observations go into an ACT-style policy, the policy predicts a short action chunk, and evaluation checks whether rollout behavior reaches the goal.

The project includes a PushT-style demonstration generator, config-driven training, checkpoint export, rollout metrics, failure-case logging, result summaries, README assets, and a static demo page. I kept the implementation small enough that I can explain the dataset schema, model input/output, training objective, and evaluation metrics line by line.

The main lesson is that a VLA-style project should not stop at training loss. I evaluate behavior with success rate, final distance, rollout length, and action smoothness, then inspect failed episodes to understand whether the issue comes from data, policy capacity, action chunking, or rollout drift.
