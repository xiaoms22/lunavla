from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from lunavla.evidence import (
    ClaimDecision,
    EvidenceManifest,
    EvidenceSource,
    EvidenceStatistic,
    hierarchical_paired_bootstrap_interval,
    wilson_interval,
)


def _hierarchical_inputs() -> tuple[list[float], list[float], list[int], list[int]]:
    control: list[float] = []
    treatment: list[float] = []
    train_seeds: list[int] = []
    episode_ids: list[int] = []
    for train_seed, differences in (
        (11, (0.8, 1.0, 1.2, 1.4)),
        (22, (1.8, 2.0, 2.2, 2.4)),
        (33, (2.8, 3.0, 3.2, 3.4)),
    ):
        for episode_id, difference in enumerate(differences):
            control.append(0.0)
            treatment.append(difference)
            train_seeds.append(train_seed)
            episode_ids.append(episode_id)
    return control, treatment, train_seeds, episode_ids


def test_hierarchical_bootstrap_samples_seed_then_episode_and_is_deterministic() -> None:
    control, treatment, train_seeds, episode_ids = _hierarchical_inputs()
    result = hierarchical_paired_bootstrap_interval(
        control,
        treatment,
        train_seeds=train_seeds,
        episode_ids=episode_ids,
        metric="final_distance",
        samples=2_000,
        seed=19,
    )
    repeated = hierarchical_paired_bootstrap_interval(
        control,
        treatment,
        train_seeds=train_seeds,
        episode_ids=episode_ids,
        metric="final_distance",
        samples=2_000,
        seed=19,
    )

    matrix = np.asarray(treatment, dtype=np.float64).reshape(3, 4)
    rng = np.random.default_rng(19)
    cluster_indices = rng.integers(0, 3, size=(2_000, 3))
    episode_indices = rng.integers(0, 4, size=(2_000, 3, 4))
    sampled = np.take_along_axis(matrix[cluster_indices], episode_indices, axis=2)
    expected_interval = np.quantile(np.mean(sampled, axis=(1, 2)), [0.025, 0.975])

    assert result == repeated
    assert result.train_seeds == (11, 22, 33)
    assert result.train_seed_n == 3
    assert result.episodes_per_seed == 4
    assert result.paired_n == 12
    assert result.mean_difference == pytest.approx(float(np.mean(matrix)))
    assert (result.lower, result.upper) == pytest.approx(tuple(expected_interval))
    assert result.supports("positive")


def test_hierarchical_bootstrap_is_invariant_to_input_row_order() -> None:
    control, treatment, train_seeds, episode_ids = _hierarchical_inputs()
    order = [7, 0, 11, 3, 4, 8, 1, 10, 5, 2, 9, 6]
    shuffled = hierarchical_paired_bootstrap_interval(
        [control[index] for index in order],
        [treatment[index] for index in order],
        train_seeds=[train_seeds[index] for index in order],
        episode_ids=[episode_ids[index] for index in order],
        metric="distance",
        samples=1_000,
        seed=7,
    )
    ordered = hierarchical_paired_bootstrap_interval(
        control,
        treatment,
        train_seeds=train_seeds,
        episode_ids=episode_ids,
        metric="distance",
        samples=1_000,
        seed=7,
    )
    assert shuffled == ordered


@pytest.mark.parametrize(
    ("train_seeds", "episode_ids", "message"),
    [
        ([11, 11, 22, 22], [0, 0, 0, 1], "must be unique"),
        ([11, 11, 22, 22], [0, 1, 0, 2], "same episode_id set"),
        ([11, 11, 11, 11], [0, 1, 2, 3], "at least two training seeds"),
    ],
)
def test_hierarchical_bootstrap_rejects_invalid_cluster_matrices(
    train_seeds: list[int], episode_ids: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        hierarchical_paired_bootstrap_interval(
            [0.0] * 4,
            [1.0] * 4,
            train_seeds=train_seeds,
            episode_ids=episode_ids,
            metric="distance",
            samples=10,
        )


def test_wilson_rejects_boolean_and_fractional_counts() -> None:
    with pytest.raises(TypeError, match="successes must be an integer"):
        wilson_interval(True, 10)
    with pytest.raises(TypeError, match="trials must be an integer"):
        wilson_interval(1, 2.5)  # type: ignore[arg-type]


def _allowed_claim() -> ClaimDecision:
    return ClaimDecision.from_checks(
        claim_id="instruction_following",
        checks={
            "effect_interval": True,
            "full_design": True,
            "matrix_complete": True,
        },
        allowed_statement="Controlled evidence supports instruction following.",
        denied_statement="Instruction following has not been established.",
    )


def _statistic() -> EvidenceStatistic:
    return EvidenceStatistic(
        statistic_id="counterfactual_distance",
        metric="final_distance",
        scope="counterfactual-minus-control",
        method="hierarchical_paired_bootstrap",
        estimate=0.4,
        lower=0.1,
        upper=0.7,
        sample_n=120,
        train_seed_n=5,
    )


def _manifest(*, reduced: bool = False, matrix_complete: bool = True) -> EvidenceManifest:
    claim = _allowed_claim()
    if reduced or not matrix_complete:
        claim = ClaimDecision.from_checks(
            claim_id="instruction_following",
            checks={"full_design": not reduced, "matrix_complete": matrix_complete},
            allowed_statement="Controlled evidence supports instruction following.",
            denied_statement="Instruction following has not been established.",
        )
    return EvidenceManifest(
        schema_version=1,
        design_id="language-alpha2",
        design_sha256="a" * 64,
        reduced_design=reduced,
        matrix_complete=matrix_complete,
        integrity_checks=(("clean_source", True), ("single_git_sha", True)),
        sources=(EvidenceSource("seed-11", "b" * 64),) if matrix_complete else (),
        statistics=(_statistic(),),
        claims=(claim,),
    )


def test_evidence_manifest_round_trip_preserves_derived_claim_gate(tmp_path: Path) -> None:
    manifest = _manifest()
    assert manifest.claims[0].allowed
    assert manifest.claims[0].failed_checks == ()
    assert manifest.claims[0].statement.startswith("Controlled evidence")

    path = manifest.write(tmp_path / "evidence_manifest.json")
    loaded = EvidenceManifest.load(path)
    assert loaded == manifest

    tampered = copy.deepcopy(manifest.to_dict())
    tampered["claims"][0]["allowed"] = False
    with pytest.raises(ValueError, match="claim.allowed"):
        EvidenceManifest.from_mapping(tampered)


def test_reduced_and_incomplete_manifests_keep_claims_closed() -> None:
    reduced = _manifest(reduced=True)
    incomplete = _manifest(matrix_complete=False)
    assert not reduced.claims[0].allowed
    assert not incomplete.claims[0].allowed
    assert reduced.claims[0].statement.endswith("not been established.")


@pytest.mark.parametrize(
    ("reduced", "matrix_complete", "checks"),
    [
        (True, True, (("clean_source", True),)),
        (False, False, (("clean_source", True),)),
        (False, True, (("clean_source", False),)),
    ],
)
def test_evidence_manifest_rejects_open_claim_when_global_gate_fails(
    reduced: bool,
    matrix_complete: bool,
    checks: tuple[tuple[str, bool], ...],
) -> None:
    with pytest.raises(ValueError, match="fail closed"):
        EvidenceManifest(
            schema_version=1,
            design_id="language-alpha2",
            design_sha256="a" * 64,
            reduced_design=reduced,
            matrix_complete=matrix_complete,
            integrity_checks=checks,
            sources=(EvidenceSource("seed-11", "b" * 64),),
            statistics=(_statistic(),),
            claims=(_allowed_claim(),),
        )
