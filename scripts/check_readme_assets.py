from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageSequence


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check README-visible MiniMind-VLA image and animation assets.")
    parser.add_argument("--out", default="outputs/readme_asset_check.md", help="Markdown report path.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def add_row(rows: list[dict[str, Any]], asset: str, status: str, detail: str, next_action: str) -> None:
    rows.append(
        {
            "asset": asset,
            "status": status,
            "detail": detail,
            "next_action": next_action,
        }
    )


def status_rank(status: str) -> int:
    return {"pass": 0, "warn": 1, "fail": 2}[status]


def overall_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "fail"
    return max((str(row["status"]) for row in rows), key=status_rank)


def readme_image_targets() -> list[str]:
    if not README.exists():
        return []
    text = README.read_text(encoding="utf-8")
    targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    return [target.split("#", 1)[0].strip() for target in targets if target.strip()]


def parse_svg_size(path: Path) -> tuple[int | None, int | None]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    width_text = root.attrib.get("width")
    height_text = root.attrib.get("height")
    if width_text and height_text:
        return int(float(width_text)), int(float(height_text))
    view_box = root.attrib.get("viewBox")
    if view_box:
        parts = [float(part) for part in view_box.replace(",", " ").split()]
        if len(parts) == 4:
            return int(parts[2]), int(parts[3])
    return None, None


def raster_has_variation(image: Image.Image) -> bool:
    rgb = image.convert("RGB")
    extrema = rgb.getextrema()
    return any(high - low > 4 for low, high in extrema)


def frames_differ(path: Path) -> bool:
    with Image.open(path) as image:
        frames = [frame.convert("RGB").copy() for frame in ImageSequence.Iterator(image)]
    if len(frames) < 2:
        return False
    first = frames[0]
    for frame in frames[1:]:
        if ImageChops.difference(first, frame.resize(first.size)).getbbox():
            return True
    return False


def check_raster(path: Path, rows: list[dict[str, Any]]) -> None:
    asset = relative(path)
    try:
        with Image.open(path) as image:
            width, height = image.size
            frame_count = getattr(image, "n_frames", 1)
            varied = raster_has_variation(image)
    except OSError as exc:
        add_row(rows, asset, "fail", f"cannot open raster: {exc}", "Regenerate README assets.")
        return

    if width < 120 or height < 80:
        add_row(rows, asset, "fail", f"{width}x{height}", "Regenerate with a larger visual canvas.")
        return
    if not varied:
        add_row(rows, asset, "fail", f"{width}x{height}, single-color visual", "Regenerate from a completed run.")
        return
    if path.suffix.lower() == ".gif":
        if frame_count < 2:
            add_row(rows, asset, "fail", f"{width}x{height}, {frame_count} frame", "Regenerate the animated asset.")
            return
        if not frames_differ(path):
            add_row(rows, asset, "warn", f"{width}x{height}, {frame_count} frames with little motion", "Regenerate after a rollout with visible movement.")
            return
        add_row(rows, asset, "pass", f"{width}x{height}, {frame_count} animated frames", "Asset is README-ready.")
        return
    add_row(rows, asset, "pass", f"{width}x{height}", "Asset is README-ready.")


def check_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    asset = relative(path)
    try:
        width, height = parse_svg_size(path)
    except ET.ParseError as exc:
        add_row(rows, asset, "fail", f"invalid SVG: {exc}", "Regenerate or edit the SVG.")
        return
    text = path.read_text(encoding="utf-8")
    if width is None or height is None:
        add_row(rows, asset, "warn", "SVG has no explicit width or viewBox", "Add dimensions so GitHub renders it predictably.")
        return
    if width < 120 or height < 80:
        add_row(rows, asset, "fail", f"{width}x{height}", "Regenerate with a larger visual canvas.")
        return
    if "<text" not in text and "<path" not in text and "<rect" not in text:
        add_row(rows, asset, "warn", f"{width}x{height}, few visible elements", "Inspect the SVG before release.")
        return
    add_row(rows, asset, "pass", f"{width}x{height}", "Asset is README-ready.")


def check_asset_file(target: str, rows: list[dict[str, Any]]) -> None:
    if target.startswith(("http://", "https://", "mailto:")):
        add_row(rows, target, "pass", "external asset", "No local check needed.")
        return
    path = resolve(target)
    asset = relative(path)
    if not path.exists():
        add_row(rows, target, "fail", "missing file", "Regenerate README assets or fix the README link.")
        return
    if path.stat().st_size < 256:
        add_row(rows, asset, "fail", f"{path.stat().st_size} bytes", "Regenerate the asset; the file is too small.")
        return
    suffix = path.suffix.lower()
    if suffix in {".png", ".gif", ".jpg", ".jpeg"}:
        check_raster(path, rows)
    elif suffix == ".svg":
        check_svg(path, rows)
    else:
        add_row(rows, asset, "warn", f"unsupported suffix {suffix}", "Inspect this asset manually.")


def check_manifest(rows: list[dict[str, Any]], targets: list[str]) -> None:
    manifest_path = ROOT / "images" / "asset_manifest.json"
    if not manifest_path.exists():
        add_row(rows, "images/asset_manifest.json", "fail", "missing file", "Run export_readme_assets.py.")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        add_row(rows, "images/asset_manifest.json", "fail", f"invalid JSON: {exc}", "Regenerate README assets.")
        return
    assets = manifest.get("assets", [])
    missing = [asset for asset in assets if not resolve(asset).exists()]
    unreferenced = [asset for asset in assets if asset not in targets]
    if missing:
        add_row(rows, "images/asset_manifest.json", "fail", "missing " + ", ".join(missing), "Regenerate README assets.")
    elif unreferenced:
        add_row(
            rows,
            "images/asset_manifest.json",
            "warn",
            "not referenced in README: " + ", ".join(unreferenced),
            "Update README or the asset manifest.",
        )
    else:
        add_row(rows, "images/asset_manifest.json", "pass", f"{len(assets)} generated assets listed", "Manifest matches README assets.")


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines


def build_report(rows: list[dict[str, Any]]) -> str:
    overall = overall_status(rows)
    lines: list[str] = [
        "# MiniMind-VLA README Asset Check",
        "",
        f"Overall: `{overall}`",
        "",
        "This report checks whether README-visible images and animations are present and renderable.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(markdown_table(rows))
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/export_readme_assets.py --run-dir outputs/act_pusht_baseline --out-dir images",
            "python scripts/check_readme_assets.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    targets = readme_image_targets()
    if not targets:
        add_row(rows, "README.md", "fail", "no local image references found", "Add README-visible assets.")
    for target in targets:
        check_asset_file(target, rows)
    check_manifest(rows, targets)

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(rows), encoding="utf-8")

    status = overall_status(rows)
    print(f"README asset check: {status}")
    print(f"README asset report: {out_path}")
    return 1 if status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
