from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

README_MEDIA = [
    "images/pusht_act_eval.gif",
    "images/pusht_diffusion_policy_eval.gif",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the README media manifest for LunaVLA.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/ or an absolute path.")
    parser.add_argument("--out-dir", required=True, help="Output directory under repo root or an absolute path.")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {relative(path)}")


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    require_file(run_dir / "training_summary.json", "training summary")
    require_file(run_dir / "eval_summary.json", "evaluation summary")
    for asset in README_MEDIA:
        require_file(ROOT / asset, "README media asset")

    manifest = {
        "run_dir": relative(run_dir),
        "assets": README_MEDIA,
        "note": (
            "README PushT media are curated LeRobot evaluation animations. "
            "This command records the checked local run used beside those homepage assets; "
            "it does not regenerate the curated PushT comparison GIFs."
        ),
    }
    manifest_path = out_dir / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    for asset in README_MEDIA:
        print(ROOT / asset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
