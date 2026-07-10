# Behavior Cloning From Scratch

Behavior cloning is the smallest imitation-learning baseline in LunaVLA.

```text
demonstration trajectory -> (observation_t, action_t) pairs -> supervised regression
```

For the PushT-style teaching task, each record contains the current state and the expert action that moves the object toward the goal. The BC policy learns:

```text
observation_t + instruction_features -> action_t
```

## What The Code Does

| Concept | File | Meaning |
| --- | --- | --- |
| dataset | `dataset/pusht_dataset.py` | generates demonstration records |
| policy | `model/policy_bc.py` | small NumPy MLP action predictor |
| training | `trainer/train_bc_pusht.py` | supervised MSE training loop |
| smoke run | `scripts/run_bc_smoke.py` | train, eval, summarize, report |
| evaluation | `eval_vla.py` | closed-loop rollout from predicted actions |

## Why Low Loss Is Not Enough

BC is trained on expert states, but rollout evaluation visits states created by the policy's own actions. Small errors can compound: an action that is slightly wrong changes the next observation, and the next prediction may be made from a state that was rare in the demonstrations.

That is why LunaVLA reports success rate, final distance, action smoothness, failure labels, and failure subtasks instead of only reporting training loss.

## Run It

```bash
python scripts/run_bc_smoke.py
```

Then open:

- `outputs/bc_pusht_cpu_smoke/summary_report.md`
- `outputs/bc_pusht_cpu_smoke/project_report.md`
- `outputs/bc_pusht_cpu_smoke/web_demo.html`

## Interview-Safe Explanation

LunaVLA's BC baseline is a from-scratch supervised imitation-learning policy. It converts demonstration records into input/action pairs, trains a tiny MLP with MSE loss, and then evaluates the learned policy through rollout because closed-loop behavior can fail even when supervised loss is low.
