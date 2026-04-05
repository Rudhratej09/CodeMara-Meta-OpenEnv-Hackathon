# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eco-LLM Routing Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EcoLLMAction, EcoLLMObservation


class EcoLLMEnv(EnvClient[EcoLLMAction, EcoLLMObservation, State]):
    """
    Client for the Eco-LLM Inference Routing Environment.

    Example:
        >>> with EcoLLMEnv(base_url="http://localhost:8000") as env:
        ...     obs = env.reset()
        ...     print(obs.observation.query)
        ...     result = env.step(EcoLLMAction(strategy="NONE", model_choice="SMALL"))
        ...     print(result.reward)
    """

    def _step_payload(self, action: EcoLLMAction) -> Dict:
        return {
            "strategy": action.strategy,
            "model_choice": action.model_choice,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EcoLLMObservation]:
        obs_data = payload.get("observation", {})
        observation = EcoLLMObservation(
            query=obs_data.get("query", ""),
            cache_contents=obs_data.get("cache_contents", []),
            carbon_intensity=obs_data.get("carbon_intensity", 0.5),
            step=obs_data.get("step", 0),
            total_steps=obs_data.get("total_steps", 1),
            task=obs_data.get("task", "easy"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            info=obs_data.get("info", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
