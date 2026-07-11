from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from lunavla import lerobot_integration as integration
from lunavla.lerobot_integration import (
    CLAIM_SCOPE,
    ENV_ID,
    ENV_OBS_TYPE,
    EXPECTED_FRAME_COUNT,
    EXPECTED_IMAGE_SHAPE,
    EXPECTED_DEPENDENCY_VERSIONS,
    EXPECTED_LUNAVLA_VERSION,
    EXPECTED_NEXT_OBSERVATION_BOUNDARY,
    EXPECTED_PLANNED_DOWNLOAD_BYTES,
    EXPECTED_POLICY_CONFIG,
    EXPECTED_TERMINAL_FRAME_INDICES,
    INTEGRATION_ID,
    INTEGRATION_MANIFEST_SCHEMA_VERSION,
    MAX_DOWNLOAD_BYTES,
    OFFICIAL_EPISODE,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    OFFICIAL_RETURN_UINT8,
    OFFICIAL_SOURCE_FILES,
    OFFICIAL_VIDEO_BACKEND,
    IntegrationManifest,
    SourceFileContract,
    load_official_dataset,
    preflight_official_download,
    run_headless_pusht_smoke,
    run_official_integration,
    run_transformer_optimizer_step,
    validate_official_episode,
    verify_downloaded_source_files,
)


def _fake_samples() -> tuple[dict[str, Any], ...]:
    image = np.zeros((3, 96, 96), dtype=np.uint8)
    samples: list[dict[str, Any]] = []
    for index in range(EXPECTED_FRAME_COUNT):
        frame_image = image.copy()
        frame_image[:, 0, 0] = index % 256
        samples.append(
            {
                "observation.image": frame_image,
                "observation.state": np.asarray(
                    [float(index), float(index + 1)],
                    dtype=np.float32,
                ),
                "action": np.asarray([256.0, 256.0], dtype=np.float32),
                "episode_index": np.asarray(OFFICIAL_EPISODE, dtype=np.int64),
                "frame_index": np.asarray(index, dtype=np.int64),
                "next.reward": np.asarray(index / EXPECTED_FRAME_COUNT, dtype=np.float32),
                "next.done": np.asarray(
                    index in EXPECTED_TERMINAL_FRAME_INDICES,
                    dtype=bool,
                ),
                "task": "Push the T-shaped block onto the T-shaped target.",
            }
        )
    return tuple(samples)


def _hub_info(*, extra_size: int = 0) -> SimpleNamespace:
    siblings = [
        SimpleNamespace(
            rfilename=contract.path,
            size=contract.size,
            lfs=SimpleNamespace(size=contract.size, sha256=contract.sha256),
        )
        for contract in OFFICIAL_SOURCE_FILES
    ]
    siblings.append(
        SimpleNamespace(
            rfilename="meta/info.json",
            size=(
                EXPECTED_PLANNED_DOWNLOAD_BYTES
                - sum(contract.size for contract in OFFICIAL_SOURCE_FILES)
                + extra_size
            ),
            lfs=None,
        )
    )
    return SimpleNamespace(sha=OFFICIAL_REVISION, siblings=siblings)


class _FakeHubApi:
    def __init__(self, info: SimpleNamespace) -> None:
        self.info = info
        self.calls: list[tuple[str, str, bool]] = []

    def dataset_info(
        self,
        repo_id: str,
        *,
        revision: str,
        files_metadata: bool,
    ) -> SimpleNamespace:
        self.calls.append((repo_id, revision, files_metadata))
        return self.info


def test_preflight_pins_revision_hashes_and_download_limit() -> None:
    api = _FakeHubApi(_hub_info())
    result = preflight_official_download(api=api)

    assert api.calls == [(OFFICIAL_REPO_ID, OFFICIAL_REVISION, True)]
    assert result.resolved_revision == OFFICIAL_REVISION
    assert result.max_download_bytes == MAX_DOWNLOAD_BYTES
    assert result.planned_download_bytes == EXPECTED_PLANNED_DOWNLOAD_BYTES
    assert result.source_files == OFFICIAL_SOURCE_FILES


def test_preflight_rejects_revision_hash_and_size_drift() -> None:
    wrong_revision = _hub_info()
    wrong_revision.sha = "0" * 40
    with pytest.raises(ValueError, match="expected pinned revision"):
        preflight_official_download(api=_FakeHubApi(wrong_revision))

    wrong_hash = _hub_info()
    wrong_hash.siblings[0].lfs.sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preflight_official_download(api=_FakeHubApi(wrong_hash))

    too_large = _hub_info(extra_size=MAX_DOWNLOAD_BYTES)
    with pytest.raises(ValueError, match="exceeds the download limit"):
        preflight_official_download(api=_FakeHubApi(too_large))

    changed_tree = _hub_info(extra_size=1)
    with pytest.raises(ValueError, match="Hub tree size changed"):
        preflight_official_download(api=_FakeHubApi(changed_tree))


def test_downloaded_source_hash_verification_is_fail_closed(tmp_path: Path) -> None:
    payload = b"local fixture\n"
    import hashlib

    contract = SourceFileContract(
        path="data/file.bin",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    path = tmp_path / contract.path
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    assert verify_downloaded_source_files(
        tmp_path,
        source_files=(contract,),
    ) == {contract.path: contract.sha256}

    path.write_bytes(payload + b"tampered")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_downloaded_source_files(tmp_path, source_files=(contract,))


def test_real_dataset_factory_receives_only_the_pinned_contract(tmp_path: Path) -> None:
    calls: list[tuple[str, Mapping[str, object]]] = []

    def factory(repo_id: str, **kwargs: object) -> list[object]:
        calls.append((repo_id, kwargs))
        return []

    assert load_official_dataset(tmp_path, dataset_factory=factory) == []
    assert calls == [
        (
            OFFICIAL_REPO_ID,
            {
                "root": tmp_path,
                "episodes": [0],
                "revision": OFFICIAL_REVISION,
                "video_backend": OFFICIAL_VIDEO_BACKEND,
                "return_uint8": OFFICIAL_RETURN_UINT8,
            },
        )
    ]


def test_episode_validation_covers_dtype_shape_terminal_and_next_boundary() -> None:
    validation = validate_official_episode(_fake_samples())

    assert validation.frame_count == 161
    assert validation.image_shape == (96, 96, 3)
    assert validation.image_dtype == "uint8"
    assert validation.state_dtype == "float32"
    assert validation.action_dtype == "float32"
    assert validation.terminal_frame_indices == (159, 160)
    assert "terminal frames self-reference" in validation.next_observation_boundary


def test_episode_validation_rejects_upstream_contract_drift() -> None:
    samples = list(_fake_samples())
    samples[7] = dict(samples[7])
    samples[7]["observation.state"] = np.zeros(2, dtype=np.float64)
    with pytest.raises(TypeError, match="state must be float32"):
        validate_official_episode(samples)

    samples = list(_fake_samples())
    samples[8] = dict(samples[8])
    samples[8]["frame_index"] = np.asarray(9, dtype=np.int64)
    with pytest.raises(ValueError, match="discontinuous"):
        validate_official_episode(samples)


@pytest.mark.torch
def test_official_format_frames_complete_one_bounded_optimizer_step() -> None:
    result = run_transformer_optimizer_step(_fake_samples())

    assert result.policy_id == "transformer_chunk_cvae"
    assert result.device == "cpu"
    assert result.batch_size == 2
    assert result.steps == 1
    assert np.isfinite(result.loss)
    assert result.parameters_changed is True


class _FakePushTEnv:
    def __init__(self) -> None:
        self.closed = False
        self.actions: list[np.ndarray[Any, Any]] = []

    @staticmethod
    def _observation() -> dict[str, np.ndarray[Any, Any]]:
        return {
            "pixels": np.zeros(EXPECTED_IMAGE_SHAPE, dtype=np.uint8),
            "agent_pos": np.asarray([200.0, 220.0], dtype=np.float64),
        }

    def reset(self, *, seed: int) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, object]]:
        assert seed == 202611
        return self._observation(), {"is_success": False}

    def step(
        self,
        action: np.ndarray[Any, Any],
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, object]]:
        self.actions.append(action.copy())
        return self._observation(), 0.25, False, False, {"is_success": False}

    def close(self) -> None:
        self.closed = True


def test_headless_environment_smoke_maps_steps_and_closes_resources() -> None:
    env = _FakePushTEnv()
    calls: list[tuple[str, Mapping[str, object]]] = []

    def factory(env_id: str, **kwargs: object) -> _FakePushTEnv:
        calls.append((env_id, kwargs))
        return env

    result = run_headless_pusht_smoke(env_factory=factory)

    assert calls == [
        (ENV_ID, {"obs_type": ENV_OBS_TYPE, "render_mode": "rgb_array"})
    ]
    assert result.steps == 3
    assert result.close_completed is True
    assert env.closed is True
    assert len(env.actions) == 3
    assert all(action.dtype == np.float32 and action.shape == (2,) for action in env.actions)


def _manifest() -> IntegrationManifest:
    return IntegrationManifest(
        schema_version=INTEGRATION_MANIFEST_SCHEMA_VERSION,
        integration_id=INTEGRATION_ID,
        generated_at_utc="2026-07-11T00:00:00+00:00",
        git_sha="a" * 40,
        git_dirty=False,
        source={
            "repo_id": OFFICIAL_REPO_ID,
            "revision": OFFICIAL_REVISION,
            "episode": OFFICIAL_EPISODE,
            "video_backend": OFFICIAL_VIDEO_BACKEND,
            "return_uint8": OFFICIAL_RETURN_UINT8,
            "max_download_bytes": MAX_DOWNLOAD_BYTES,
            "planned_download_bytes": EXPECTED_PLANNED_DOWNLOAD_BYTES,
            "sha256": {item.path: item.sha256 for item in OFFICIAL_SOURCE_FILES},
        },
        dataset_validation={
            "frame_count": EXPECTED_FRAME_COUNT,
            "image_shape": EXPECTED_IMAGE_SHAPE,
            "image_dtype": "uint8",
            "state_shape": (2,),
            "state_dtype": "float32",
            "action_shape": (2,),
            "action_dtype": "float32",
            "episode_indices": (0,),
            "frame_index_start": 0,
            "frame_index_end": 160,
            "terminal_frame_indices": (159, 160),
            "next_observation_boundary": EXPECTED_NEXT_OBSERVATION_BOUNDARY,
        },
        optimizer_step={
            "policy_id": "transformer_chunk_cvae",
            "device": "cpu",
            "batch_size": 2,
            "steps": 1,
            "parameters_changed": True,
            "loss": 0.5,
            "policy_config": dict(EXPECTED_POLICY_CONFIG),
        },
        environment_smoke={
            "env_id": ENV_ID,
            "obs_type": ENV_OBS_TYPE,
            "seed": 202611,
            "steps": 3,
            "pixel_shape": EXPECTED_IMAGE_SHAPE,
            "pixel_dtype": "uint8",
            "agent_position_shape": (2,),
            "action_shape": (2,),
            "action_dtype": "float32",
            "close_completed": True,
        },
        dependencies={
            "lunavla": EXPECTED_LUNAVLA_VERSION,
            **EXPECTED_DEPENDENCY_VERSIONS,
        },
        python="3.12.13",
        device="cpu",
        deterministic=True,
        artifact_policy={
            "manifest_only": True,
            "cache_uploaded": False,
            "video_uploaded": False,
        },
        claim_allowed=False,
        claim_scope=CLAIM_SCOPE,
    )


def test_versioned_integration_manifest_round_trip_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    path = manifest.write(tmp_path / "integration_manifest.json")
    assert IntegrationManifest.load(path) == manifest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        manifest.write(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["claim_allowed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot authorize"):
        IntegrationManifest.load(path)


def test_manifest_rejects_unknown_fields_and_non_authoritative_source(tmp_path: Path) -> None:
    path = _manifest().write(tmp_path / "integration_manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown integration manifest fields"):
        IntegrationManifest.load(path)

    payload.pop("unexpected")
    payload["source"]["revision"] = "0" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source.revision"):
        IntegrationManifest.load(path)

    payload = _manifest().to_dict()
    payload["schema_version"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        IntegrationManifest.load(path)


@pytest.mark.parametrize(
    ("field_path", "replacement", "message"),
    [
        (("claim_scope",), "tampered claim", "claim_scope"),
        (
            ("source", "planned_download_bytes"),
            EXPECTED_PLANNED_DOWNLOAD_BYTES - 1,
            "planned_download_bytes",
        ),
        (
            ("optimizer_step", "policy_config", "d_model"),
            32,
            "policy_config",
        ),
        (("environment_smoke", "seed"), 0, "environment_smoke.seed"),
        (
            ("dataset_validation", "next_observation_boundary"),
            "cross episode",
            "next_observation_boundary",
        ),
        (("dependencies", "lunavla"), "2.0.0a2", "dependencies.lunavla"),
        (("generated_at_utc",), "not-a-timestamp", "generated_at_utc"),
        (("source", "episode"), False, "source.episode"),
        (("optimizer_step", "steps"), True, "optimizer_step.steps"),
    ],
)
def test_manifest_rejects_nested_contract_tampering(
    tmp_path: Path,
    field_path: tuple[str, ...],
    replacement: object,
    message: str,
) -> None:
    payload = _manifest().to_dict()
    current: dict[str, Any] = payload
    for field in field_path[:-1]:
        nested = current[field]
        assert isinstance(nested, dict)
        current = nested
    current[field_path[-1]] = replacement
    path = tmp_path / "integration_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        IntegrationManifest.load(path)


def test_manifest_rejects_unknown_nested_fields(tmp_path: Path) -> None:
    payload = _manifest().to_dict()
    source = payload["source"]
    assert isinstance(source, dict)
    source["unexpected"] = True
    path = tmp_path / "integration_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown source fields"):
        IntegrationManifest.load(path)


def _stub_official_integration(
    monkeypatch: pytest.MonkeyPatch,
    git_states: list[tuple[str, bool]],
) -> None:
    samples = _fake_samples()
    states = iter(git_states)
    monkeypatch.setattr(integration, "_git_state", lambda _root: next(states))
    monkeypatch.setattr(
        integration,
        "preflight_official_download",
        lambda **_kwargs: integration.DownloadPreflight(
            repo_id=OFFICIAL_REPO_ID,
            requested_revision=OFFICIAL_REVISION,
            resolved_revision=OFFICIAL_REVISION,
            planned_download_bytes=EXPECTED_PLANNED_DOWNLOAD_BYTES,
            max_download_bytes=MAX_DOWNLOAD_BYTES,
            source_files=OFFICIAL_SOURCE_FILES,
        ),
    )
    monkeypatch.setattr(integration, "load_official_dataset", lambda *_args, **_kwargs: samples)
    monkeypatch.setattr(integration, "materialize_episode", lambda _dataset: samples)
    monkeypatch.setattr(
        integration,
        "verify_downloaded_source_files",
        lambda _root: {item.path: item.sha256 for item in OFFICIAL_SOURCE_FILES},
    )
    monkeypatch.setattr(
        integration,
        "run_transformer_optimizer_step",
        lambda _samples: integration.OptimizerStepValidation(
            policy_id="transformer_chunk_cvae",
            device="cpu",
            batch_size=2,
            steps=1,
            loss=0.5,
            parameters_changed=True,
            policy_config=dict(EXPECTED_POLICY_CONFIG),
        ),
    )
    monkeypatch.setattr(
        integration,
        "run_headless_pusht_smoke",
        lambda **_kwargs: integration.EnvironmentValidation(
            env_id=ENV_ID,
            obs_type=ENV_OBS_TYPE,
            seed=202611,
            steps=3,
            pixel_shape=EXPECTED_IMAGE_SHAPE,
            pixel_dtype="uint8",
            agent_position_shape=(2,),
            action_shape=(2,),
            action_dtype="float32",
            close_completed=True,
        ),
    )
    monkeypatch.setattr(
        integration,
        "dependency_versions",
        lambda: {
            "lunavla": EXPECTED_LUNAVLA_VERSION,
            **EXPECTED_DEPENDENCY_VERSIONS,
        },
    )


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (("b" * 40, False), "does not match expected"),
        (("a" * 40, True), "requires a clean Git checkout"),
    ],
)
def test_official_integration_rejects_initial_source_mismatch_or_dirty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: tuple[str, bool],
    message: str,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(integration, "_git_state", lambda _root: state)
    with pytest.raises(ValueError, match=message):
        run_official_integration(
            root=checkout,
            expected_git_sha="a" * 40,
            cache_dir=tmp_path / "cache",
            output_path=tmp_path / "integration_manifest.json",
        )


def test_official_integration_rechecks_source_after_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _stub_official_integration(
        monkeypatch,
        [("a" * 40, False), ("a" * 40, True)],
    )
    output = tmp_path / "integration_manifest.json"
    with pytest.raises(ValueError, match="changed while.*running"):
        run_official_integration(
            root=checkout,
            expected_git_sha="a" * 40,
            cache_dir=tmp_path / "cache",
            output_path=output,
        )
    assert not output.exists()


def test_official_integration_writes_only_a_strict_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _stub_official_integration(
        monkeypatch,
        [("a" * 40, False), ("a" * 40, False)],
    )
    output = tmp_path / "integration_manifest.json"
    manifest = run_official_integration(
        root=checkout,
        expected_git_sha="a" * 40,
        cache_dir=tmp_path / "cache",
        output_path=output,
    )
    assert IntegrationManifest.load(output) == manifest
    assert tuple(path.name for path in output.parent.iterdir() if path.is_file()) == (
        "integration_manifest.json",
    )
