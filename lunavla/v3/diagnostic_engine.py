from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import numpy.typing as npt

from .config import ExperimentConfig
from .contracts import EpisodeRecordV3, ObservationV3, TransitionV3
from .diagnostics import InterventionSpecV1, PromptSpecV1, StateRouteSpecV1
from .normalization import NormalizationStatsV1


def typed_episode_key(value: str | int) -> str:
    payload = {
        "type": "integer" if isinstance(value, int) else "string",
        "value": value,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


class DiagnosticExecutionError(RuntimeError):
    def __init__(self, stage: str, origin: str, cause: BaseException) -> None:
        self.stage = stage
        self.origin = origin
        super().__init__(f"diagnostic {stage} failed in {origin}: {cause}")


@dataclass(frozen=True)
class RoutedObservationV1:
    observation: ObservationV3
    prompt_spec: PromptSpecV1
    expert_state_keys: tuple[str, ...]
    prompt_state_keys: tuple[str, ...]
    donor_id: str | None
    intervention_sha256: str


class DiagnosticRouterV1:
    def __init__(
        self,
        config: ExperimentConfig,
        normalization: NormalizationStatsV1,
        *,
        route: StateRouteSpecV1 | None = None,
        intervention: InterventionSpecV1 | None = None,
        donor_instructions: Mapping[str, tuple[str, str]] | None = None,
        donor_images: Mapping[
            tuple[str, int], tuple[str, Mapping[str, npt.NDArray[np.generic]]]
        ]
        | None = None,
        counterfactual_transform_id: str | None = None,
    ) -> None:
        if config.contract_revision != 2 or not config.diagnostics["enabled"]:
            raise ValueError("DiagnosticRouterV1 requires an enabled revision 2 config")
        self.config = config
        self.normalization = normalization
        self.route = route or StateRouteSpecV1(
            str(config.routing["mode"]), tuple(config.routing["state_features"])
        )
        self.intervention = intervention or InterventionSpecV1(
            "control", "prompt", "control", "rollout", {}
        )
        if self.intervention.kind not in {"prompt", "image"}:
            raise ValueError("Beta 1 router accepts prompt or image interventions only")
        self.donor_instructions = MappingProxyType(dict(donor_instructions or {}))
        self.donor_images = MappingProxyType(dict(donor_images or {}))
        self.counterfactual_transform_id = counterfactual_transform_id
        declared_states = tuple(item.name for item in config.feature_schema.by_role("state"))
        if self.route.state_features != tuple(config.routing["state_features"]):
            raise ValueError("route state features must match the resolved config")
        if any(item not in declared_states for item in self.route.state_features):
            raise ValueError("route references an undeclared state feature")

    def _instruction(self, observation: ObservationV3) -> tuple[str, str | None]:
        instruction = observation.instruction
        if instruction is None:
            raise ValueError("prompt diagnostics require a non-empty observation instruction")
        donor_id: str | None = None
        operator = self.intervention.operator
        if self.intervention.kind == "prompt" and operator == "mask":
            instruction = str(self.config.prompt["neutral_token"])
        elif self.intervention.kind == "prompt" and operator == "shuffle":
            key = typed_episode_key(observation.episode_id)
            if key not in self.donor_instructions:
                raise ValueError("shuffle intervention is missing a same-split donor")
            donor_id, instruction = self.donor_instructions[key]
            if donor_id == key:
                raise ValueError("shuffle donor cannot reference the recipient episode")
        elif self.intervention.kind == "prompt" and operator == "counterfactual":
            if self.counterfactual_transform_id != "fake_target_swap_v1":
                raise ValueError("counterfactual intervention requires fake_target_swap_v1")
            instruction = instruction.replace("green target", "blue counterfactual target")
            if instruction == observation.instruction:
                raise ValueError("counterfactual transform did not change the instruction")
        return instruction, donor_id

    def route_observation(
        self, observation: ObservationV3, *, phase: str = "eval"
    ) -> RoutedObservationV1:
        if phase not in {"train", "eval", "deploy"}:
            raise ValueError("diagnostic phase must be train, eval, or deploy")
        if tuple(observation.images) != tuple(
            item.name for item in self.config.feature_schema.by_role("image")
        ):
            raise ValueError("observation camera order drifted from FeatureSchema")
        instruction, donor_id = self._instruction(observation)
        prompt_state: dict[str, tuple[float, ...]] = {}
        expert_state: dict[str, npt.NDArray[np.float32]] = {}
        expert_keys: list[str] = []
        prompt_keys: list[str] = []
        for name in self.route.state_features:
            source = np.asarray(observation.state[name], dtype=np.float32)
            if self.route.prompt_enabled:
                prompt_state[name] = tuple(float(item) for item in source)
                prompt_keys.append(name)
            if self.route.expert_enabled:
                expert_state[name] = np.array(source, copy=True)
                expert_keys.append(name)
            else:
                stats = self.normalization.features.get(name)
                expert_state[name] = (
                    np.zeros_like(source) if stats is None else np.array(stats.offset, copy=True)
                )
        layout = (
            "assistant_before_cameras_v1"
            if self.intervention.kind == "prompt"
            and self.intervention.operator == "layout_drift"
            else "canonical"
        )
        prompt = PromptSpecV1(
            raw_instruction=instruction,
            public_slots=self.config.prompt["public_slots"],
            state_values=prompt_state,
            renderer_id=str(self.config.prompt["renderer_id"]),
            renderer_version=int(self.config.prompt["renderer_version"]),
            camera_order=tuple(self.config.prompt["camera_order"]),
            assistant_target=str(self.config.prompt["assistant_target"]),
            layout_variant=layout,
        )
        selected_images = {
            name: observation.images[name] for name in self.config.prompt["camera_order"]
        }
        if self.intervention.kind == "image" and self.intervention.operator == "shuffle":
            image_key = (typed_episode_key(observation.episode_id), observation.step_index)
            if image_key not in self.donor_images:
                raise ValueError("image shuffle is missing a same-split donor step")
            donor_id, donor_values = self.donor_images[image_key]
            if donor_id == image_key[0]:
                raise ValueError("image shuffle donor cannot reference the recipient episode")
            if tuple(donor_values) != tuple(selected_images):
                raise ValueError("image donor camera order does not match the policy view")
            selected_images = {
                name: np.array(donor_values[name], copy=True) for name in donor_values
            }
        metadata = {
            "diagnostic": {
                "route": self.route.mode,
                "arm_id": self.intervention.arm_id,
                "prompt_sha256": prompt.rendered_sha256,
                "expert_state_keys": expert_keys,
                "prompt_state_keys": prompt_keys,
                "donor_id": donor_id,
                "phase": phase,
            }
        }
        sanitized = ObservationV3(
            images=selected_images,
            state=expert_state,
            instruction=prompt.rendered_text,
            timestamp_s=observation.timestamp_s,
            episode_id=observation.episode_id,
            step_index=observation.step_index,
            metadata=metadata,
        )
        return RoutedObservationV1(
            sanitized,
            prompt,
            tuple(expert_keys),
            tuple(prompt_keys),
            donor_id,
            self.intervention.sha256(),
        )

    def route_episode(self, episode: EpisodeRecordV3) -> EpisodeRecordV3:
        transitions: list[TransitionV3] = []
        for transition in episode.transitions:
            current = self.route_observation(
                transition.observation, phase="train"
            ).observation
            next_observation = self.route_observation(
                transition.next_observation, phase="train"
            ).observation
            transitions.append(
                TransitionV3(
                    current,
                    transition.action,
                    transition.reward,
                    next_observation,
                    transition.terminated,
                    transition.truncated,
                    {"diagnostic_source_hash": hashlib.sha256(
                        typed_episode_key(episode.episode_id).encode("utf-8")
                    ).hexdigest()},
                )
            )
        return EpisodeRecordV3(
            episode.episode_id,
            tuple(transitions),
            {"diagnostic_route": self.route.mode},
        )
