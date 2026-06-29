# Visual Attributions

LunaVLA uses three kinds of README visuals:

- Real PushT visual context converted from saved local LeRobot evaluation videos.
- Local project evidence generated from this repo with `python scripts/export_readme_assets.py --run-dir outputs/act_pusht_baseline --out-dir images`.
- Ecosystem context media copied or resized from official public project repositories.

## PushT Context

| local file | source | use |
| --- | --- | --- |
| `images/pusht_act_eval.gif` | Saved local ACT PushT evaluation video, converted from `pusht_act/videos/pusht_0/eval_episode_4.mp4`. | ACT policy visual reference for the real PushT task. |
| `images/pusht_diffusion_policy_eval.gif` | Saved local Diffusion Policy PushT evaluation video, converted from `pusht_diffusion_pretrained_eval.mp4`. | Diffusion Policy visual reference for the real PushT task. |

These GIFs are visual context for comparing policy behavior. LunaVLA's runnable claims come from the local PushT-style teaching loop, generated metrics, and reports.

## Ecosystem Media

| local file | source project | source file | license | use |
| --- | --- | --- | --- | --- |
| `images/ecosystem/lerobot_control_demo.webp` | [LeRobot](https://github.com/huggingface/lerobot) | `media/readme/robots_control_video.webp` | Apache-2.0 | Visual context for robot control workflows. |
| `images/ecosystem/lerobot_so100_demo.webp` | [LeRobot](https://github.com/huggingface/lerobot) | `media/readme/so100_video.webp` | Apache-2.0 | Visual context for low-cost robot learning demos. |
| `images/ecosystem/lerobot_vla_architecture.jpg` | [LeRobot](https://github.com/huggingface/lerobot) | `media/readme/VLA_architecture.jpg` | Apache-2.0 | Visual context for VLA-style system components. |
| `images/ecosystem/libero_sim_overview.jpg` | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `images/fig1.png` | MIT | Visual context for simulation benchmark tasks. |

These images are used only as ecosystem references. The runnable evidence claimed by LunaVLA comes from the local PushT-style loop, generated reports, local rollout trace, and README assets in this repository.

## Refresh

```bash
python scripts/prepare_homepage_media.py
```
