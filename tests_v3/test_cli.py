from __future__ import annotations

import json
from pathlib import Path

from lunavla.v3.cli import main


def test_validate_audit_replay_run_and_verify_cli(tmp_path: Path, capsys: object) -> None:
    config = Path("configs/v3/fake_pusht_alpha.yaml")
    assert main(["validate-config", str(config)]) == 0
    audit = tmp_path / "audit.json"
    replay = tmp_path / "replay.json"
    assert main(["data-audit", str(config), "--out", str(audit)]) == 0
    assert main(["replay", str(config), "--episode", "0", "--out", str(replay)]) == 0
    assert json.loads(audit.read_text())["episode_count"] == 6
    assert json.loads(replay.read_text())["steps"] == 5

    payload = json.loads(json.dumps(__import__("yaml").safe_load(config.read_text())))
    payload["artifacts"]["output_dir"] = str(tmp_path / "run")
    temp_config = tmp_path / "config.yaml"
    temp_config.write_text(__import__("yaml").safe_dump(payload), encoding="utf-8")
    assert main(["run", str(temp_config)]) == 0
    assert main(["verify-run", str(tmp_path / "run")]) == 0


def test_migrate_config_cli(tmp_path: Path) -> None:
    source = Path("configs/v2/numpy_baseline.yaml")
    output = tmp_path / "v3.yaml"
    assert main(["migrate-config", str(source), "--out", str(output)]) == 0
    assert output.is_file()
    assert "schema_version: 3" in output.read_text()
