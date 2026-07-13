from __future__ import annotations

import csv
import hashlib
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
from lunavla.v3.artifacts import sha256_file
from scripts.generate_v3_diagnostic_image_fixtures import (
    FIXTURES,
    descriptor,
    generated_payloads,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _image_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    monkeypatch.chdir(tmp_path)
    shutil.copy2(
        Path(__file__).parents[1] / "configs/v3/diagnostic_act_image.yaml",
        tmp_path / "image-config.yaml",
    )
    source = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/v3/diagnostic_image_ci_design.yaml").read_text()
    )
    source["base_config"] = "image-config.yaml"
    source["output_dir"] = "image-study"
    design = tmp_path / "image-design.yaml"
    design.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return design, tmp_path / "image-study"


def test_reduced_diagnostic_vertical_path_is_complete_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design, output = _study(tmp_path, monkeypatch)
    assert run_diagnostic(design) == output
    evidence = verify_diagnostic_output(output)
    assert evidence["expected_pairs"] == 40
    assert evidence["observed_pairs"] == 40
    assert evidence["matrix_complete"] is True
    assert evidence["reduced_design"] is True
    assert evidence["claim_allowed"] is False
    assert "reduced_design" in evidence["gate_reasons"]
    assert "beta1_framework_only" in evidence["gate_reasons"]
    assert evidence["thumbnail_manifest_sha256"] is None
    manifests = sorted(output.glob("runs/*/*/manifest.json"))
    assert len(manifests) == 4
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


def test_failed_generation_is_marked_incomplete_without_replacing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design, _ = _study(tmp_path, monkeypatch)
    payload = yaml.safe_load(design.read_text())
    payload["evaluation_seeds"] = [1000]
    payload["output_dir"] = "bad"
    design.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="at least two"):
        run_diagnostic(design)
    marker = tmp_path / ".bad.incomplete.json"
    assert json.loads(marker.read_text())["complete"] is False
    assert not (tmp_path / "bad").exists()


def test_act_image_suite_executes_stepwise_shuffle_and_verified_thumbnails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design, output = _image_study(tmp_path, monkeypatch)
    assert run_diagnostic(design) == output
    evidence = verify_diagnostic_output(output)
    assert evidence["expected_pairs"] == evidence["observed_pairs"] == 4
    assert evidence["thumbnail_manifest_sha256"] is not None
    thumbnails = json.loads((output / "thumbnails.json").read_text())
    assert len(thumbnails["records"]) == 4
    shuffle_rows = [
        json.loads(line)
        for trace in output.glob("runs/*/*/trace.jsonl")
        for line in trace.read_text().splitlines()
        if json.loads(line)["arm"] == "image_shuffle"
    ]
    assert shuffle_rows
    assert all(row["donor_id"] and row["donor_content_sha256"] for row in shuffle_rows)
    report = write_diagnostic_report(output, tmp_path / "image-report")
    assert len(list((report / "thumbnails").glob("*.png"))) == 4
    assert "Synthetic thumbnails" in (report / "index.html").read_text()


def test_committed_synthetic_png_fixtures_are_byte_reproducible() -> None:
    config, payloads = generated_payloads()
    actual = json.loads((FIXTURES / "manifest.json").read_text())
    assert actual == descriptor(config, payloads)
    assert len(payloads) == 4
    for filename, payload in payloads.items():
        assert (FIXTURES / filename).read_bytes() == payload


def test_csv_semantics_are_recomputed_even_when_hash_is_updated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design, output = _study(tmp_path, monkeypatch)
    run_diagnostic(design)
    csv_path = output / "per_pair.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["total_reward"] = str(float(rows[0]["total_reward"]) + 1.0)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evidence_path = output / "evidence_manifest.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["per_pair_sha256"] = sha256_file(csv_path)
    _write_json(evidence_path, evidence)
    with pytest.raises(ValueError, match="CSV metrics"):
        verify_diagnostic_output(output)


def test_mixed_sha_remains_verifiable_but_closes_framework_and_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design, output = _study(tmp_path, monkeypatch)
    run_diagnostic(design)
    manifest_path = sorted(output.glob("runs/*/*/manifest.json"))[0]
    manifest = json.loads(manifest_path.read_text())
    manifest["git_sha"] = "0" * 40
    _write_json(manifest_path, manifest)
    evidence_path = output / "evidence_manifest.json"
    evidence = json.loads(evidence_path.read_text())
    relative = manifest_path.relative_to(output).as_posix()
    for record in evidence["source_manifests"]:
        if record["path"] == relative:
            record["sha256"] = sha256_file(manifest_path)
    evidence["homogeneous"] = False
    evidence["framework_statement_allowed"] = False
    evidence["release_eligible"] = False
    evidence["allowed_wording"] = None
    evidence["gate_reasons"] = sorted(set(evidence["gate_reasons"]) | {"mixed_sha"})
    _write_json(evidence_path, evidence)
    verified = verify_diagnostic_output(output)
    assert verified["claim_allowed"] is False
    assert verified["framework_statement_allowed"] is False
    assert verified["release_eligible"] is False
    assert "mixed_sha" in verified["gate_reasons"]


def test_extra_cell_config_difference_is_reported_without_opening_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design, output = _study(tmp_path, monkeypatch)
    run_diagnostic(design)
    run_root = sorted(output.glob("runs/*/*"))[0]
    config_path = run_root / "resolved_config.json"
    config = json.loads(config_path.read_text())
    config["training"]["learning_rate"] *= 2
    _write_json(config_path, config)
    cell_path = run_root / "cell_contract.json"
    normalized = json.loads(json.dumps(config))
    normalized["training"]["seed"] = "<design-train-seed>"
    normalized["routing"]["mode"] = "<design-route>"
    normalized["artifacts"]["output_dir"] = "<design-output>"
    _write_json(cell_path, normalized)

    envelope_path = run_root / "checkpoint/checkpoint.v3.json"
    envelope = json.loads(envelope_path.read_text())
    envelope["config_sha256"] = sha256_file(config_path)
    _write_json(envelope_path, envelope)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_sha256"] = sha256_file(config_path)
    manifest["cell_contract_sha256"] = sha256_file(cell_path)
    manifest["checkpoint_envelope_sha256"] = sha256_file(envelope_path)
    _write_json(manifest_path, manifest)

    evidence_path = output / "evidence_manifest.json"
    evidence = json.loads(evidence_path.read_text())
    relative = manifest_path.relative_to(output).as_posix()
    for record in evidence["source_manifests"]:
        if record["path"] == relative:
            record["sha256"] = sha256_file(manifest_path)
    evidence["homogeneous"] = False
    evidence["framework_statement_allowed"] = False
    evidence["release_eligible"] = False
    evidence["allowed_wording"] = None
    evidence["gate_reasons"] = sorted(
        set(evidence["gate_reasons"]) | {"config_drift"}
    )
    _write_json(evidence_path, evidence)
    verified = verify_diagnostic_output(output)
    assert verified["claim_allowed"] is False
    assert verified["release_eligible"] is False
    assert "config_drift" in verified["gate_reasons"]


def test_parity_semantics_are_recomputed_after_hash_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design, output = _study(tmp_path, monkeypatch)
    run_diagnostic(design)
    run_root = sorted(output.glob("runs/*/*"))[0]
    parity_path = run_root / "parity.json"
    parity = json.loads(parity_path.read_text())
    parity["records"][-1]["prompt_sha256"] = "f" * 64
    _write_json(parity_path, parity)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parity_manifest_sha256"] = sha256_file(parity_path)
    _write_json(manifest_path, manifest)
    evidence_path = output / "evidence_manifest.json"
    evidence = json.loads(evidence_path.read_text())
    relative = manifest_path.relative_to(output).as_posix()
    for record in evidence["source_manifests"]:
        if record["path"] == relative:
            record["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _write_json(evidence_path, evidence)
    with pytest.raises(ValueError, match="drift"):
        verify_diagnostic_output(output)
