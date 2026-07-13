from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from lunavla.v3.config import ExperimentConfig
from lunavla.v3.diagnostic_workflow import (
    _image_donor_bank,
    _instruction_donor_bank,
    synthetic_thumbnail_payloads,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v3/diagnostic_act_image.yaml"
FIXTURES = ROOT / "tests_v3/fixtures/diagnostic_images"
SEEDS = (1000, 1001)


def generated_payloads() -> tuple[ExperimentConfig, dict[str, bytes]]:
    config = ExperimentConfig.load(CONFIG)
    _, instructions = _instruction_donor_bank(config, SEEDS, donor_seed=42)
    _, images = _image_donor_bank(
        config, SEEDS, instructions, donor_seed=42
    )
    return config, synthetic_thumbnail_payloads(config, SEEDS, images)


def descriptor(config: ExperimentConfig, payloads: dict[str, bytes]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "synthetic_only": True,
        "config_sha256": config.sha256(),
        "records": [
            {
                "path": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "shape": [16, 16, 3],
                "dtype": "uint8",
                "metadata_allowed": False,
            }
            for filename, payload in sorted(payloads.items())
        ],
    }


def write_fixtures() -> None:
    config, payloads = generated_payloads()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        (FIXTURES / filename).write_bytes(payload)
    (FIXTURES / "manifest.json").write_text(
        json.dumps(descriptor(config, payloads), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def check_fixtures() -> None:
    config, payloads = generated_payloads()
    expected = descriptor(config, payloads)
    actual = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit("diagnostic image fixture manifest drifted")
    for filename, payload in payloads.items():
        path = FIXTURES / filename
        if path.read_bytes() != payload:
            raise SystemExit(f"diagnostic image fixture bytes drifted: {filename}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (16, 16) or image.info:
                raise SystemExit(f"diagnostic image fixture metadata drifted: {filename}")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.write:
        write_fixtures()
    else:
        check_fixtures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
