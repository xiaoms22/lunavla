# Visual Attributions

MiniMind-VLA uses two kinds of README visuals:

- Local project evidence generated from this repo with `python scripts/export_readme_assets.py --run-dir outputs/act_pusht_baseline --out-dir images`.
- Ecosystem context media copied or resized from official public project repositories.

## Ecosystem Media

| local file | source project | source file | license | use |
| --- | --- | --- | --- | --- |
| `images/ecosystem/lerobot_control_demo.webp` | [LeRobot](https://github.com/huggingface/lerobot) | `media/readme/robots_control_video.webp` | Apache-2.0 | Visual context for robot control workflows. |
| `images/ecosystem/lerobot_so100_demo.webp` | [LeRobot](https://github.com/huggingface/lerobot) | `media/readme/so100_video.webp` | Apache-2.0 | Visual context for low-cost robot learning demos. |
| `images/ecosystem/lerobot_vla_architecture.jpg` | [LeRobot](https://github.com/huggingface/lerobot) | `media/readme/VLA_architecture.jpg` | Apache-2.0 | Visual context for VLA-style system components. |
| `images/ecosystem/libero_sim_overview.jpg` | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `images/fig1.png` | MIT | Visual context for simulation benchmark tasks. |

These images are used only as ecosystem references. The runnable evidence claimed by MiniMind-VLA comes from the local PushT-style loop, generated reports, and README assets in this repository.

## Refresh

```bash
python scripts/prepare_homepage_media.py
```
