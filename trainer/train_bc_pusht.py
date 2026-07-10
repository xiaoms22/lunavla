from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.train_core import train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NumPy BC MLP policy.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing experiment directory."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = train_from_config(
        args.config,
        overwrite=args.overwrite,
        expected_policy_type="numpy_bc_mlp",
    )
    print(f"trained: {summary['project_name']}")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"final_loss: {summary['final_loss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
