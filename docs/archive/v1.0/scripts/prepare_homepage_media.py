from __future__ import annotations

import argparse
import json
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "images" / "ecosystem"


SOURCES: list[dict[str, Any]] = [
    {
        "name": "LeRobot robot control demo",
        "source": "https://raw.githubusercontent.com/huggingface/lerobot/main/media/readme/robots_control_video.webp",
        "output": "lerobot_control_demo.webp",
        "mode": "copy",
        "license": "Apache-2.0",
        "project": "https://github.com/huggingface/lerobot",
    },
    {
        "name": "LeRobot SO-100 demo",
        "source": "https://raw.githubusercontent.com/huggingface/lerobot/main/media/readme/so100_video.webp",
        "output": "lerobot_so100_demo.webp",
        "mode": "copy",
        "license": "Apache-2.0",
        "project": "https://github.com/huggingface/lerobot",
    },
    {
        "name": "LeRobot VLA architecture overview",
        "source": "https://raw.githubusercontent.com/huggingface/lerobot/main/media/readme/VLA_architecture.jpg",
        "output": "lerobot_vla_architecture.jpg",
        "mode": "resize",
        "max_width": 960,
        "license": "Apache-2.0",
        "project": "https://github.com/huggingface/lerobot",
    },
    {
        "name": "LIBERO simulation benchmark overview",
        "source": "https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/images/fig1.png",
        "output": "libero_sim_overview.jpg",
        "mode": "resize",
        "max_width": 960,
        "license": "MIT",
        "project": "https://github.com/Lifelong-Robot-Learning/LIBERO",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare README ecosystem media from official public sources.")
    parser.add_argument("--out-dir", default="images/ecosystem", help="Output directory under repo root.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "LunaVLA media fetcher"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def write_copy(payload: bytes, path: Path) -> None:
    path.write_bytes(payload)


def write_resized_jpeg(payload: bytes, path: Path, max_width: int) -> None:
    with Image.open(BytesIO(payload)) as image:
        frame = image.convert("RGB")
        if frame.width > max_width:
            new_height = round(frame.height * max_width / frame.width)
            frame = frame.resize((max_width, new_height), Image.Resampling.LANCZOS)
        frame.save(path, format="JPEG", quality=84, optimize=True, progressive=True)


def prepare_assets(out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, Any]] = []
    for item in SOURCES:
        payload = fetch(str(item["source"]))
        target = out_dir / str(item["output"])
        if item["mode"] == "copy":
            write_copy(payload, target)
        elif item["mode"] == "resize":
            write_resized_jpeg(payload, target, int(item["max_width"]))
        else:
            raise ValueError(f"Unknown media mode: {item['mode']}")
        prepared.append(
            {
                "name": item["name"],
                "project": item["project"],
                "source": item["source"],
                "license": item["license"],
                "local_path": target.relative_to(ROOT).as_posix(),
                "bytes": target.stat().st_size,
            }
        )
    return prepared


def main() -> int:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    assets = prepare_assets(out_dir)
    manifest = {
        "note": "External ecosystem media copied or resized from official public project repositories.",
        "assets": assets,
    }
    manifest_path = out_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for asset in assets:
        print(f"{asset['local_path']} ({asset['bytes']} bytes)")
    print(f"{manifest_path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
