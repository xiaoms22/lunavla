from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from lunavla.v3 import (
    RunManifestV4R3,
    run_diagnostic,
    run_manifest_from_mapping,
    verify_diagnostic_output,
    write_diagnostic_report,
)


def _study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.chdir(tmp_path)
    shutil.copy2(
        Path(__file__).parents[1] / "configs/v3/diagnostic_fake_libero.yaml",
        tmp_path / "config.yaml",
    )
    source = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/v3/diagnostic_ci_design.yaml").read_text()
    )
    source["base_config"] = "config.yaml"
    source["output_dir"] = "study"
    design = tmp_path / "design.yaml"
    design.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return design, tmp_path / "study"


def test_reduced_diagnostic_vertical_path_is_complete_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design, output = _study(tmp_path, monkeypatch)
    assert run_diagnostic(design) == Path("study")
    evidence = verify_diagnostic_output(output)
    assert evidence["expected_pairs"] == 12
    assert evidence["observed_pairs"] == 12
    assert evidence["matrix_complete"] is True
    assert evidence["reduced_design"] is True
    assert evidence["claim_allowed"] is False
    manifests = sorted(output.glob("runs/*/*/manifest.json"))
    assert len(manifests) == 2
    parsed = run_manifest_from_mapping(json.loads(manifests[0].read_text()))
    assert isinstance(parsed, RunManifestV4R3)
    assert parsed.parity_verified and parsed.complete
    with pytest.raises(FileExistsError, match="already exists"):
        run_diagnostic(design)


def test_diagnostic_tampering_is_detected_and_report_requires_verified_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design, output = _study(tmp_path, monkeypatch)
    run_diagnostic(design)
    report = write_diagnostic_report(output, tmp_path / "report")
    assert (report / "index.html").is_file()
    trace = next(output.glob("runs/*/*/trace.jsonl"))
    trace.write_text(trace.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_diagnostic_output(output)
    with pytest.raises(ValueError, match="hash mismatch"):
        write_diagnostic_report(output, tmp_path / "bad-report")
