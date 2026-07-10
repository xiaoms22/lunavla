from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from lunavla.cli import main
from lunavla.config import ExperimentConfig
from lunavla.manifest import RunManifest
from lunavla.manifest import sha256_transitions
from lunavla.run import _dataset_source, _evaluate_with_cleanup, _task_env, run_experiment


def _config() -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "schema_version": 2,
            "project_name": "v2-run-smoke",
            "engine": "lunavla_v2",
            "policy": {
                "type": "numpy_linear_chunk",
                "state_dim": 4,
                "instruction_dim": 0,
                "action_dim": 2,
                "chunk_size": 2,
                "device": "cpu",
            },
            "task": {
                "id": "pusht_style_point_reach",
                "max_steps": 5,
                "goal": [0.8, 0.2],
            },
            "dataset": {
                "type": "memory",
                "split": "train",
                "seed": 3,
                "episode_count": 3,
                "parameters": {"steps_per_episode": 4},
            },
            "training": {
                "device": "cpu",
                "seed": 4,
                "batch_size": 4,
                "steps": 3,
                "learning_rate": 0.02,
            },
            "evaluation": {
                "execution_mode": "open_loop_chunk",
                "episodes": 2,
                "seed": 90,
            },
            "artifacts": {"output_dir": "outputs/v2-test"},
        }
    )


def test_evaluation_closes_external_environment_on_success_and_failure() -> None:
    class ClosableEnvironment:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    expected = object()
    environment = ClosableEnvironment()
    engine = type("SuccessfulEngine", (), {"evaluate": lambda *_args: expected})()
    assert _evaluate_with_cleanup(engine, object(), environment) is expected  # type: ignore[arg-type]
    assert environment.closed is True

    environment = ClosableEnvironment()

    def fail(*_args: object) -> object:
        raise RuntimeError("evaluation failed")

    engine = type("FailingEngine", (), {"evaluate": fail})()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        _evaluate_with_cleanup(engine, object(), environment)  # type: ignore[arg-type]
    assert environment.closed is True


@pytest.mark.integration
def test_run_writes_hash_verified_contract_and_refuses_overwrite(tmp_path: Path) -> None:
    config = _config()
    manifest = run_experiment(config, root=tmp_path, command=["lunavla-v2", "train"])
    output = tmp_path / "outputs/v2-test"
    assert RunManifest.load(output / "manifest.json") == manifest
    assert (output / "checkpoint.json").is_file()
    assert (output / "metrics.json").is_file()
    assert manifest.data_seeds == [3]
    assert manifest.train_seeds == [4]
    assert manifest.eval_seeds == [90, 91]
    split_sets = [set(manifest.dataset_split[name]) for name in ("train", "validation", "test")]
    assert split_sets[0].isdisjoint(split_sets[1])
    assert split_sets[0].isdisjoint(split_sets[2])
    assert split_sets[1].isdisjoint(split_sets[2])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_experiment(config, root=tmp_path)


@pytest.mark.integration
def test_failed_overwrite_preserves_previous_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    run_experiment(config, root=tmp_path)
    output = tmp_path / "outputs/v2-test"
    original_manifest = (output / "manifest.json").read_bytes()
    marker = output / "keep-on-failure.txt"
    marker.write_text("old evidence", encoding="utf-8")

    def fail_source(config: ExperimentConfig, root: Path) -> object:
        raise RuntimeError("injected training failure")

    monkeypatch.setattr("lunavla.run._dataset_source", fail_source)
    with pytest.raises(RuntimeError, match="injected"):
        run_experiment(config, root=tmp_path, overwrite=True)
    assert marker.read_text(encoding="utf-8") == "old evidence"
    assert (output / "manifest.json").read_bytes() == original_manifest
    assert not list((tmp_path / "outputs").glob(".v2-test.tmp-*"))


def test_train_cli_and_device_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "lunavla"\nversion = "0"\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(_config().to_dict()), encoding="utf-8")
    assert main(["train", str(config_path), "--require-device", "cpu"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["run_id"] == "v2-run-smoke"
    manifest = RunManifest.load(tmp_path / "outputs/v2-test/manifest.json")
    assert "--require-device" in manifest.command
    assert manifest.git_dirty is True
    with pytest.raises(ValueError, match="does not satisfy"):
        run_experiment(_config(), root=tmp_path, overwrite=True, require_device="cuda")


def test_run_rejects_output_outside_outputs(tmp_path: Path) -> None:
    payload = _config().to_dict()
    payload["artifacts"]["output_dir"] = "elsewhere/run"
    with pytest.raises(ValueError, match="under outputs"):
        ExperimentConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "unsafe",
    ["outputs/../lunavla", "outputs/../../outside", "outputs"],
)
def test_config_rejects_output_traversal(unsafe: str) -> None:
    payload = _config().to_dict()
    payload["artifacts"]["output_dir"] = unsafe
    with pytest.raises(ValueError, match="under outputs"):
        ExperimentConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "unsafe",
    ["../checkpoint.json", "nested/checkpoint.json", "/tmp/checkpoint.json"],
)
def test_config_rejects_checkpoint_path_escape(unsafe: str) -> None:
    payload = _config().to_dict()
    payload["artifacts"]["checkpoint_name"] = unsafe
    with pytest.raises(ValueError, match="plain file name"):
        ExperimentConfig.from_mapping(payload)


def test_run_rejects_symlinked_outputs_root(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "outputs").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        run_experiment(_config(), root=tmp_path)


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/v2/transformer_chunk_cpu.yaml",
        "configs/v2/transformer_visual_cpu.yaml",
    ],
)
def test_modality_dataset_seed_changes_actual_data(config_path: str) -> None:
    first = ExperimentConfig.load(config_path)
    payload = first.to_dict()
    payload["dataset"]["seed"] = int(first.dataset["seed"]) + 100
    second = ExperimentConfig.from_mapping(payload)
    first_hash = sha256_transitions(tuple(_dataset_source(first, Path.cwd()).load()))
    second_hash = sha256_transitions(tuple(_dataset_source(second, Path.cwd()).load()))
    assert first_hash != second_hash


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/v2/transformer_visual_cpu.yaml",
        "configs/v2/transformer_visual_state_only_cpu.yaml",
    ],
)
def test_visual_configs_forward_observation_mode_to_data_and_env(
    config_path: str,
) -> None:
    config = ExperimentConfig.load(config_path)
    source = _dataset_source(config, Path.cwd())
    env = _task_env(config)

    assert source.observation_mode == "vision_required"  # type: ignore[attr-defined]
    assert env.observation_mode == "vision_required"  # type: ignore[attr-defined]
    assert source.load()[0].observation.state.shape == (3,)
    assert env.reset(seed=19).state.shape == (3,)


def test_privileged_observation_mode_is_forwarded_explicitly() -> None:
    payload = ExperimentConfig.load(
        "configs/v2/transformer_visual_cpu.yaml"
    ).to_dict()
    payload["policy"]["state_dim"] = 7
    payload["dataset"]["parameters"]["observation_mode"] = "privileged"
    config = ExperimentConfig.from_mapping(payload)

    source = _dataset_source(config, Path.cwd())
    env = _task_env(config)
    assert source.observation_mode == "privileged"  # type: ignore[attr-defined]
    assert env.observation_mode == "privileged"  # type: ignore[attr-defined]
    assert source.load()[0].observation.state.shape == (7,)
    assert env.reset(seed=19).state.shape == (7,)
