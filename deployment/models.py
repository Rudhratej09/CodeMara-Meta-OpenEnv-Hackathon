# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Eco-LLM Inference Routing Environment.

Hierarchical multi-objective LLM query routing environment:
- Strategy layer: NONE, USE_CACHE, DO_CASCADE, EARLY_EXIT, WAIT, CALL_KB
- Model layer: SMALL, MEDIUM, LARGE
- Reward: accuracy - λ·energy_cost - μ·latency + bonuses
"""

from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


VALID_STRATEGIES = ["NONE", "USE_CACHE", "DO_CASCADE", "EARLY_EXIT", "WAIT", "CALL_KB"]
VALID_MODELS = ["SMALL", "MEDIUM", "LARGE"]


class EcoLLMAction(Action):
    """Action for the Eco-LLM routing environment."""

    strategy: str = Field(
        default="NONE",
        description=f"Routing strategy. One of: {VALID_STRATEGIES}",
    )
    model_choice: str = Field(
        default="SMALL",
        description=f"LLM model to use. One of: {VALID_MODELS}. Ignored if strategy is USE_CACHE or WAIT.",
    )


class EcoLLMObservation(Observation):
    """Observation from the Eco-LLM routing environment."""

    query: str = Field(default="", description="Current query to route")
    cache_contents: List[str] = Field(
        default_factory=list,
        description="List of query strings already answered and cached",
    )
    carbon_intensity: float = Field(
        default=0.5,
        description="Current grid carbon intensity, 0.0 (clean) to 1.0 (dirty)",
    )
    step: int = Field(default=0, description="Current step number in the episode")
    total_steps: int = Field(default=1, description="Total steps in this task episode")
    task: str = Field(default="easy", description="Task difficulty: easy, medium, hard")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra debug info")
