from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static rollout web demo.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/ or an absolute path.")
    return parser.parse_args()


def load_first_rollout(run_dir: Path) -> dict:
    rollout_dir = run_dir / "rollouts"
    files = sorted(rollout_dir.glob("episode_*.json"))
    if not files:
        return {"frames": [], "success": False, "final_distance": None}
    return json.loads(files[0].read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    summary_path = run_dir / "eval_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    rollout = load_first_rollout(run_dir)
    frames_json = json.dumps(rollout.get("frames", []))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MiniMind-VLA Rollout Demo</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #172026; background: #f7f3ec; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 32px 20px; }}
    h1 {{ font-size: 32px; margin: 0 0 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 24px 0; }}
    .metric {{ background: #ffffff; border: 1px solid #d9d3c8; border-radius: 8px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 6px; }}
    canvas {{ width: 100%; max-width: 720px; aspect-ratio: 1 / 1; background: #ffffff; border: 1px solid #d9d3c8; border-radius: 8px; }}
    .note {{ color: #53616b; line-height: 1.5; }}
  </style>
</head>
<body>
  <main>
    <h1>MiniMind-VLA Rollout Demo</h1>
    <p class="note">A static rollout trace generated from the first saved evaluation episode.</p>
    <section class="grid">
      <div class="metric">success rate<strong>{summary.get("success_rate", "n/a")}</strong></div>
      <div class="metric">mean distance<strong>{summary.get("mean_final_distance", "n/a")}</strong></div>
      <div class="metric">episodes<strong>{summary.get("episodes", "n/a")}</strong></div>
    </section>
    <canvas id="rollout" width="720" height="720"></canvas>
  </main>
  <script>
    const frames = {frames_json};
    const canvas = document.getElementById('rollout');
    const ctx = canvas.getContext('2d');
    function xy(p) {{ return [p[0] * canvas.width, p[1] * canvas.height]; }}
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#d9d3c8';
    for (let i = 0; i <= 10; i++) {{
      const v = i * canvas.width / 10;
      ctx.beginPath(); ctx.moveTo(v, 0); ctx.lineTo(v, canvas.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, v); ctx.lineTo(canvas.width, v); ctx.stroke();
    }}
    const goal = xy([0.8, 0.2]);
    ctx.fillStyle = '#0f766e';
    ctx.beginPath(); ctx.arc(goal[0], goal[1], 14, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#e0523f';
    ctx.lineWidth = 5;
    ctx.beginPath();
    frames.forEach((frame, index) => {{
      const point = xy(frame.position);
      if (index === 0) ctx.moveTo(point[0], point[1]);
      else ctx.lineTo(point[0], point[1]);
    }});
    ctx.stroke();
    if (frames.length) {{
      const start = xy(frames[0].position);
      const end = xy(frames[frames.length - 1].position);
      ctx.fillStyle = '#2f5bea';
      ctx.beginPath(); ctx.arc(start[0], start[1], 10, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#e0523f';
      ctx.beginPath(); ctx.arc(end[0], end[1], 10, 0, Math.PI * 2); ctx.fill();
    }}
  </script>
</body>
</html>
"""
    target = run_dir / "web_demo.html"
    target.write_text(html, encoding="utf-8")
    print(f"web demo: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
