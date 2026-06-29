# Dataset Notes

LunaVLA uses a tiny VLA record abstraction so the full imitation-learning loop can run without external simulators or large downloads.

```json
{
  "observation": [x, y, goal_x, goal_y],
  "action": [dx, dy],
  "episode_id": 0,
  "timestep": 0,
  "success": false,
  "language_instruction": "push the T block to the goal",
  "metadata": {"task": "pusht_mock"}
}
```

The PushT-style generator is deliberately simple. It makes the first repository checkout runnable while preserving the shape needed for imitation learning:

1. sample an initial object position;
2. define a fixed goal;
3. create demonstration actions that move toward the goal;
4. train a policy to predict short action chunks;
5. evaluate by rolling out predicted first actions.

The supported dataset sources in the public repo are `mock_pusht` and `jsonl`. Use `jsonl` when you want to save generated records, edit them, or plug in your own small demonstration file with the same schema.
