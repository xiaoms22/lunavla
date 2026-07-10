from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static rollout result browser.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/ or an absolute path.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_rollouts(run_dir: Path) -> list[dict[str, Any]]:
    rollout_dir = run_dir / "rollouts"
    files = sorted(rollout_dir.glob("episode_*.json"))
    if not files:
        return []
    rollouts: list[dict[str, Any]] = []
    for path in files:
        rollout = read_json(path)
        if "episode_id" not in rollout:
            rollout["episode_id"] = int(path.stem.rsplit("_", 1)[-1])
        rollout["artifact"] = path.name
        rollouts.append(rollout)
    return rollouts


def failure_summary(failures: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        category = str(failure.get("category", "unknown"))
        counts[category] = counts.get(category, 0) + 1
    return counts


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    summary = read_json(run_dir / "eval_summary.json")
    rollouts = load_rollouts(run_dir)
    failures = read_jsonl(run_dir / "failure_cases.jsonl")
    failures_by_episode = {str(row.get("episode_id")): row for row in failures}
    payload_json = json.dumps(
        {
            "summary": summary,
            "rollouts": rollouts,
            "failuresByEpisode": failures_by_episode,
            "failureSummary": failure_summary(failures),
        }
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LunaVLA Rollout Browser</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5a6670;
      --line: #d7dee4;
      --paper: #ffffff;
      --wash: #f5f7f8;
      --blue: #2f5bea;
      --red: #d64a3a;
      --green: #0f766e;
      --amber: #b7791f;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: var(--ink); background: var(--wash); }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px 18px 40px; }}
    h1 {{ font-size: 30px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 24px 0 10px; }}
    p {{ margin: 0; }}
    .note {{ color: var(--muted); line-height: 1.5; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 22px 0; }}
    .metric {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 12px; min-height: 72px; }}
    .metric span {{ color: var(--muted); display: block; font-size: 13px; }}
    .metric strong {{ display: block; font-size: 21px; margin-top: 8px; overflow-wrap: anywhere; }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 18px; align-items: start; }}
    canvas {{ width: 100%; aspect-ratio: 1 / 1; background: var(--paper); border: 1px solid var(--line); border-radius: 8px; display: block; }}
    .panel {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .episode-list {{ display: grid; gap: 8px; max-height: 360px; overflow: auto; }}
    button {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--ink);
      cursor: pointer;
      padding: 10px 12px;
      text-align: left;
      font: inherit;
    }}
    button[aria-pressed="true"] {{ border-color: var(--blue); outline: 2px solid rgba(47, 91, 234, 0.18); }}
    .tag {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; margin-left: 6px; }}
    .ok {{ background: #e7f7ef; color: #11613e; }}
    .fail {{ background: #fdecea; color: #9f2d22; }}
    .detail {{ display: grid; gap: 8px; margin-top: 12px; color: var(--muted); font-size: 14px; }}
    .detail strong {{ color: var(--ink); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--paper); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eef2f4; color: var(--muted); font-weight: 600; }}
    tr:last-child td {{ border-bottom: 0; }}
    @media (max-width: 820px) {{
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>LunaVLA Rollout Browser</h1>
    <p class="note">Inspect saved rollout episodes, metrics, and first-pass failure labels from one local run.</p>
    <section class="metrics">
      <div class="metric"><span>success rate</span><strong id="success-rate">n/a</strong></div>
      <div class="metric"><span>mean distance</span><strong id="mean-distance">n/a</strong></div>
      <div class="metric"><span>episodes</span><strong id="episodes">n/a</strong></div>
      <div class="metric"><span>failure categories</span><strong id="failure-summary">none</strong></div>
    </section>
    <section class="layout">
      <canvas id="rollout" width="720" height="720"></canvas>
      <aside class="panel">
        <h2>Episodes</h2>
        <div id="episode-list" class="episode-list"></div>
        <div id="episode-detail" class="detail"></div>
      </aside>
    </section>
    <h2>Episode Table</h2>
    <table>
      <thead>
        <tr><th>episode</th><th>status</th><th>steps</th><th>final distance</th><th>failure label</th></tr>
      </thead>
      <tbody id="episode-table"></tbody>
    </table>
  </main>
  <script>
    const payload = {payload_json};
    const canvas = document.getElementById('rollout');
    const ctx = canvas.getContext('2d');
    const list = document.getElementById('episode-list');
    const detail = document.getElementById('episode-detail');
    const tableBody = document.getElementById('episode-table');
    let selected = 0;

    function fmt(value) {{
      if (value === undefined || value === null) return 'n/a';
      if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(6).replace(/0+$/, '').replace(/\\.$/, '');
      return String(value);
    }}
    function xy(p) {{ return [p[0] * canvas.width, p[1] * canvas.height]; }}
    function failureFor(rollout) {{ return payload.failuresByEpisode[String(rollout.episode_id)] || null; }}
    function statusTag(success) {{ return success ? '<span class="tag ok">success</span>' : '<span class="tag fail">failed</span>'; }}
    function categoryText(rollout) {{
      const failure = failureFor(rollout);
      return failure ? failure.category : 'none';
    }}
    function drawGrid() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#d7dee4';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 10; i++) {{
        const v = i * canvas.width / 10;
        ctx.beginPath(); ctx.moveTo(v, 0); ctx.lineTo(v, canvas.height); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, v); ctx.lineTo(canvas.width, v); ctx.stroke();
      }}
    }}
    function drawRollout(rollout) {{
      const frames = rollout.frames || [];
      drawGrid();
      const goal = xy(rollout.goal || [0.8, 0.2]);
      ctx.fillStyle = '#0f766e';
      ctx.beginPath(); ctx.arc(goal[0], goal[1], 14, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = rollout.success ? '#2f5bea' : '#d64a3a';
      ctx.lineWidth = 5;
      ctx.beginPath();
      const startPosition = rollout.initial_position || (frames[0] ? frames[0].position : null);
      if (startPosition) {{
        const start = xy(startPosition);
        ctx.moveTo(start[0], start[1]);
      }}
      frames.forEach((frame) => {{
        const point = xy(frame.position);
        ctx.lineTo(point[0], point[1]);
      }});
      ctx.stroke();
      if (startPosition) {{
        const start = xy(startPosition);
        ctx.fillStyle = '#2f5bea';
        ctx.beginPath(); ctx.arc(start[0], start[1], 10, 0, Math.PI * 2); ctx.fill();
      }}
      if (frames.length) {{
        const end = xy(frames[frames.length - 1].position);
        ctx.fillStyle = rollout.success ? '#0f766e' : '#d64a3a';
        ctx.beginPath(); ctx.arc(end[0], end[1], 10, 0, Math.PI * 2); ctx.fill();
      }}
    }}
    function renderSummary() {{
      const summary = payload.summary || {{}};
      document.getElementById('success-rate').textContent = fmt(summary.success_rate);
      document.getElementById('mean-distance').textContent = fmt(summary.mean_final_distance);
      document.getElementById('episodes').textContent = fmt(summary.episodes);
      const entries = Object.entries(payload.failureSummary || {{}});
      document.getElementById('failure-summary').textContent = entries.length ? entries.map(([k, v]) => `${{k}}:${{v}}`).join(', ') : 'none';
    }}
    function renderList() {{
      list.innerHTML = '';
      payload.rollouts.forEach((rollout, index) => {{
        const button = document.createElement('button');
        button.type = 'button';
        button.setAttribute('aria-pressed', index === selected ? 'true' : 'false');
        button.innerHTML = `Episode ${{fmt(rollout.episode_id)}} ${{statusTag(rollout.success)}}<br><small>${{fmt(rollout.steps)}} steps, distance ${{fmt(rollout.final_distance)}}</small>`;
        button.addEventListener('click', () => {{ selected = index; render(); }});
        list.appendChild(button);
      }});
    }}
    function renderTable() {{
      tableBody.innerHTML = '';
      payload.rollouts.forEach((rollout) => {{
        const row = document.createElement('tr');
        row.innerHTML = `<td>${{fmt(rollout.episode_id)}}</td><td>${{rollout.success ? 'success' : 'failed'}}</td><td>${{fmt(rollout.steps)}}</td><td>${{fmt(rollout.final_distance)}}</td><td>${{categoryText(rollout)}}</td>`;
        tableBody.appendChild(row);
      }});
    }}
    function renderDetail(rollout) {{
      const failure = failureFor(rollout);
      const finalContext = rollout.final_task_context || {{}};
      detail.innerHTML = `
        <div><strong>episode:</strong> ${{fmt(rollout.episode_id)}} ${{statusTag(rollout.success)}}</div>
        <div><strong>artifact:</strong> ${{fmt(rollout.artifact)}}</div>
        <div><strong>instruction:</strong> ${{fmt(rollout.instruction)}}</div>
        <div><strong>final subtask:</strong> ${{fmt(finalContext.subtask_id || 'unknown')}}</div>
        <div><strong>steps:</strong> ${{fmt(rollout.steps)}}</div>
        <div><strong>initial distance:</strong> ${{fmt(rollout.initial_distance)}}</div>
        <div><strong>min distance:</strong> ${{fmt(rollout.min_distance)}}</div>
        <div><strong>final distance:</strong> ${{fmt(rollout.final_distance)}}</div>
        <div><strong>failure label:</strong> ${{failure ? failure.category : 'none'}}</div>
        <div><strong>next check:</strong> ${{failure ? failure.next_minimal_fix : 'Inspect more episodes before making a stronger claim.'}}</div>
      `;
    }}
    function render() {{
      renderSummary();
      renderList();
      renderTable();
      if (!payload.rollouts.length) {{
        drawGrid();
        detail.innerHTML = '<div><strong>No rollout files found.</strong></div>';
        return;
      }}
      const rollout = payload.rollouts[selected] || payload.rollouts[0];
      drawRollout(rollout);
      renderDetail(rollout);
    }}
    render();
  </script>
</body>
</html>
"""
    target = run_dir / "web_demo.html"
    target.write_text(html, encoding="utf-8")
    print(f"rollout browser: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
