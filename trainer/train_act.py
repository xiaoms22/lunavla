from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMind-VLA ACT training wrapper.")
    parser.add_argument("--config", default="configs/act_pusht_baseline.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subprocess.run(
        [sys.executable, "trainer/train_act_pusht.py", "--config", args.config],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
