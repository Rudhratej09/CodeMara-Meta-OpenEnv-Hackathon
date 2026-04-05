# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deployment Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DeploymentAction, DeploymentObservation


class DeploymentEnv(
    EnvClient[DeploymentAction, DeploymentObservation, State]
):
    """
    Client for the Deployment Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with DeploymentEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(DeploymentAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = DeploymentEnv.from_docker_image("deployment-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(DeploymentAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DeploymentAction) -> Dict:
        """
        Convert DeploymentAction to JSON payload for step message.

        Args:
            action: DeploymentAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DeploymentObservation]:
        """
        Parse server response into StepResult[DeploymentObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with DeploymentObservation
        """
        obs_data = payload.get("observation", {})
        observation = DeploymentObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
