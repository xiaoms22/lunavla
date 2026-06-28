from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


Color = tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export README-visible MiniMind-VLA assets.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/ or an absolute path.")
    parser.add_argument("--out-dir", required=True, help="Output directory under repo root or an absolute path.")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def new_canvas(width: int, height: int, color: Color = (247, 243, 236)) -> list[list[Color]]:
    return [[color for _ in range(width)] for _ in range(height)]


def set_pixel(canvas: list[list[Color]], x: int, y: int, color: Color) -> None:
    if 0 <= y < len(canvas) and 0 <= x < len(canvas[0]):
        canvas[y][x] = color


def draw_line(canvas: list[list[Color]], x0: int, y0: int, x1: int, y1: int, color: Color, width: int = 1) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for ox in range(-(width // 2), width // 2 + 1):
            for oy in range(-(width // 2), width // 2 + 1):
                set_pixel(canvas, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_rect(canvas: list[list[Color]], x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
    for y in range(max(0, y0), min(len(canvas), y1)):
        for x in range(max(0, x0), min(len(canvas[0]), x1)):
            canvas[y][x] = color


def draw_circle(canvas: list[list[Color]], cx: int, cy: int, radius: int, color: Color) -> None:
    r2 = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                set_pixel(canvas, x, y, color)


def write_png(path: Path, canvas: list[list[Color]]) -> None:
    height = len(canvas)
    width = len(canvas[0])
    raw = bytearray()
    for row in canvas:
        raw.append(0)
        for r, g, b in row:
            raw.extend([r, g, b])

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def plot_to_canvas(points: Iterable[tuple[float, float]], width: int = 900, height: int = 520) -> list[list[Color]]:
    canvas = new_canvas(width, height)
    margin = 70
    draw_rect(canvas, margin, margin, width - margin, height - margin, (255, 255, 255))
    for i in range(6):
        x = margin + i * (width - 2 * margin) // 5
        y = margin + i * (height - 2 * margin) // 5
        draw_line(canvas, x, margin, x, height - margin, (221, 211, 196), 1)
        draw_line(canvas, margin, y, width - margin, y, (221, 211, 196), 1)

    pts = list(points)
    if not pts:
        return canvas
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0

    def tx(x: float) -> int:
        return int(margin + (x - min_x) / max(max_x - min_x, 1e-9) * (width - 2 * margin))

    def ty(y: float) -> int:
        return int(height - margin - (y - min_y) / max(max_y - min_y, 1e-9) * (height - 2 * margin))

    pixels = [(tx(x), ty(y)) for x, y in pts]
    for a, b in zip(pixels, pixels[1:]):
        draw_line(canvas, a[0], a[1], b[0], b[1], (224, 82, 63), 4)
    draw_circle(canvas, pixels[0][0], pixels[0][1], 8, (47, 91, 234))
    draw_circle(canvas, pixels[-1][0], pixels[-1][1], 8, (224, 82, 63))
    return canvas


def export_loss_curve(run_dir: Path, out_dir: Path) -> Path:
    rows: list[tuple[float, float]] = []
    with (run_dir / "loss_curve.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((float(row["step"]), float(row["loss"])))
    target = out_dir / "loss_curve_baseline.png"
    write_png(target, plot_to_canvas(rows))
    return target


def export_rollout(run_dir: Path, out_dir: Path) -> Path:
    files = sorted((run_dir / "rollouts").glob("episode_*.json"))
    if not files:
        raise FileNotFoundError(f"No rollout JSON files found under {run_dir / 'rollouts'}")
    rollout = json.loads(files[0].read_text(encoding="utf-8"))
    frames = rollout.get("frames", [])
    canvas = new_canvas(720, 720)
    margin = 60
    draw_rect(canvas, margin, margin, 660, 660, (255, 255, 255))
    for i in range(11):
        x = margin + i * 60
        y = margin + i * 60
        draw_line(canvas, x, margin, x, 660, (221, 211, 196), 1)
        draw_line(canvas, margin, y, 660, y, (221, 211, 196), 1)

    def point(pos: list[float]) -> tuple[int, int]:
        return int(margin + pos[0] * 600), int(margin + pos[1] * 600)

    goal = point([0.8, 0.2])
    draw_circle(canvas, goal[0], goal[1], 14, (15, 118, 110))
    pixels = [point(frame["position"]) for frame in frames]
    for a, b in zip(pixels, pixels[1:]):
        draw_line(canvas, a[0], a[1], b[0], b[1], (224, 82, 63), 5)
    if pixels:
        draw_circle(canvas, pixels[0][0], pixels[0][1], 11, (47, 91, 234))
        draw_circle(canvas, pixels[-1][0], pixels[-1][1], 11, (224, 82, 63))
    target = out_dir / "rollout_demo.png"
    write_png(target, canvas)
    return target


def export_result_table(run_dir: Path, out_dir: Path) -> Path:
    training = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_dir / "eval_summary.json").read_text(encoding="utf-8"))
    rows = [
        ("project", training.get("project_name", "unknown")),
        ("records", training.get("records", "n/a")),
        ("chunk size", training.get("chunk_size", "n/a")),
        ("final loss", training.get("final_loss", "n/a")),
        ("success rate", evaluation.get("success_rate", "n/a")),
        ("mean final distance", evaluation.get("mean_final_distance", "n/a")),
        ("mean rollout length", evaluation.get("mean_rollout_length", "n/a")),
        ("mean action smoothness", evaluation.get("mean_action_smoothness", "n/a")),
    ]
    row_height = 42
    width = 860
    height = 88 + row_height * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f3ec"/>',
        '<rect x="32" y="28" width="796" height="52" rx="8" fill="#172026"/>',
        '<text x="54" y="62" font-family="Arial, sans-serif" font-size="22" fill="#ffffff">MiniMind-VLA baseline result</text>',
    ]
    y = 92
    for idx, (key, value) in enumerate(rows):
        fill = "#ffffff" if idx % 2 == 0 else "#fbfaf7"
        parts.append(f'<rect x="32" y="{y}" width="796" height="{row_height}" fill="{fill}" stroke="#d9d3c8"/>')
        parts.append(f'<text x="54" y="{y + 27}" font-family="Arial, sans-serif" font-size="17" fill="#53616b">{key}</text>')
        parts.append(f'<text x="330" y="{y + 27}" font-family="Arial, sans-serif" font-size="17" fill="#172026">{value}</text>')
        y += row_height
    parts.append("</svg>")
    target = out_dir / "result_table.svg"
    target.write_text("\n".join(parts), encoding="utf-8")
    return target


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        export_rollout(run_dir, out_dir),
        export_loss_curve(run_dir, out_dir),
        export_result_table(run_dir, out_dir),
    ]
    try:
        run_dir_for_manifest = run_dir.relative_to(ROOT).as_posix()
    except ValueError:
        run_dir_for_manifest = str(run_dir)
    manifest = {
        "run_dir": run_dir_for_manifest,
        "assets": [str(path.relative_to(ROOT).as_posix()) for path in generated],
    }
    (out_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
