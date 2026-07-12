from __future__ import annotations

import copy

import pytest

from lunavla.v3 import (
    CONNECTIVITY_STATEMENT,
    ExperimentConfig,
    ExternalDatasetSpecV1,
    IntegrationManifestV1,
    SimulationTaskSpecV1,
)


def test_revision3_real_configs_are_strict_and_round_trip() -> None:
    for path in (
        "configs/v3/beta2_pusht_integration.yaml",
        "configs/v3/beta2_libero_integration.yaml",
    ):
        config = ExperimentConfig.load(path)
        assert config.contract_revision == 3
        assert ExperimentConfig.from_mapping(config.to_dict()).sha256() == config.sha256()
        assert config.external_dataset_spec is not None
        assert config.simulation_task_spec is not None


@pytest.mark.parametrize("revision", [True, 3.0, "3"])
def test_revision3_rejects_boolean_and_coerced_versions(revision: object) -> None:
    payload = ExperimentConfig.load("configs/v3/beta2_pusht_integration.yaml").to_dict()
    payload["contract_revision"] = revision
    with pytest.raises(TypeError, match="contract_revision must be an integer"):
        ExperimentConfig.from_mapping(payload)


def test_external_source_rejects_drift_unknown_fields_and_path_escape() -> None:
    config = ExperimentConfig.load("configs/v3/beta2_pusht_integration.yaml")
    source = config.external_dataset_spec
    assert source is not None
    assert ExternalDatasetSpecV1.from_mapping(source.to_dict()).sha256() == source.sha256()
    payload = source.to_dict()
    payload["revision"] = "0" * 40
    with pytest.raises(ValueError, match="pinned"):
        drifted = ExternalDatasetSpecV1.from_mapping(payload)
        drifted.validate_supported_source()
    payload = source.to_dict()
    payload["typo"] = True
    with pytest.raises(ValueError, match="unknown ExternalDatasetSpecV1"):
        ExternalDatasetSpecV1.from_mapping(payload)
    payload = source.to_dict()
    payload["file_hashes"] = {"../escape": "0" * 64}
    with pytest.raises(ValueError, match="unsafe"):
        ExternalDatasetSpecV1.from_mapping(payload)


def test_libero_task_subset_and_feature_contract_are_pinned() -> None:
    config = ExperimentConfig.load("configs/v3/beta2_libero_integration.yaml")
    source = config.external_dataset_spec
    simulation = config.simulation_task_spec
    assert source is not None and source.task_ids == (0, 1, 2, 3)
    assert simulation is not None and simulation.init_state_ids == (0,)
    features = config.feature_schema
    assert tuple(item.name for item in features.by_role("image")) == (
        "camera.agentview",
        "camera.wrist",
    )
    assert features.by_role("state")[0].shape == (8,)
    assert features.by_role("action")[0].shape == (7,)
    assert all(item.rate_hz == 10.0 for item in features.features)
    payload = config.to_dict()
    payload["task"]["parameters"]["simulation"]["task_ids"] = [0, 1, 2]
    with pytest.raises(ValueError, match="task IDs 0-3"):
        ExperimentConfig.from_mapping(payload)


def test_real_task_dataset_pairing_and_implicit_camera_loss_fail() -> None:
    payload = ExperimentConfig.load("configs/v3/beta2_libero_integration.yaml").to_dict()
    payload["dataset"]["type"] = "lerobot_pusht"
    with pytest.raises(ValueError, match="source contract|must match"):
        ExperimentConfig.from_mapping(payload)
    payload = ExperimentConfig.load("configs/v3/beta2_libero_integration.yaml").to_dict()
    payload["policy"]["parameters"]["camera_features"] = ["camera.agentview"]
    with pytest.raises(ValueError, match="exactly match FeatureSchema order"):
        ExperimentConfig.from_mapping(payload)


def test_simulation_and_manifest_contracts_are_frozen_and_fail_closed() -> None:
    config = ExperimentConfig.load("configs/v3/beta2_pusht_integration.yaml")
    simulation = config.simulation_task_spec
    assert simulation is not None
    assert SimulationTaskSpecV1.from_mapping(simulation.to_dict()).sha256() == simulation.sha256()
    data = {"frames": 161}
    manifest = IntegrationManifestV1(
        git_sha="1" * 40,
        git_dirty=False,
        dependency_lock_sha256="2" * 64,
        source_spec_sha256="3" * 64,
        source_inventory_sha256="4" * 64,
        runner_qualification_sha256="5" * 64,
        runner_role="fixture",
        data_validation=data,
        environment_validation={"closed": True},
        policy_smokes=({"policy": "act_v3", "finite": True},),
        downloaded_bytes=0,
        claim_allowed=False,
        benchmark_claim=False,
        statement=CONNECTIVITY_STATEMENT,
    )
    data["frames"] = 0
    assert manifest.data_validation["frames"] == 161
    assert IntegrationManifestV1.from_mapping(manifest.to_dict()).sha256() == manifest.sha256()
    payload = copy.deepcopy(manifest.to_dict())
    payload["claim_allowed"] = True
    with pytest.raises(ValueError, match="cannot open"):
        IntegrationManifestV1.from_mapping(payload)
