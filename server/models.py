"""Pydantic models for the Eco-LLM Inference Routing environment."""

from __future__ import annotations

from enum import Enum
from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class Strategy(str, Enum):
    NONE = "NONE"
    USE_CACHE = "USE_CACHE"
    DO_CASCADE = "DO_CASCADE"
    EARLY_EXIT = "EARLY_EXIT"
    WAIT = "WAIT"
    CALL_KB = "CALL_KB"


class ModelChoice(str, Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"


class RLAction(Action):
    """Action taken by the routing agent at each step."""

    strategy: Strategy = Field(..., description="Routing strategy to apply")
    model_choice: ModelChoice = Field(..., description="Primary model for inference")
    exit_flag: bool = Field(..., description="Whether to terminate a cascade early")


class RLReward(BaseModel):
    """Structured reward breakdown for deterministic grading."""

    score: float = Field(..., description="1.0 if correct else 0.0")
    energy_cost: float = Field(..., description="Energy consumed this step")
    latency_cost: float = Field(..., description="Latency incurred this step")
    carbon_penalty: float = Field(..., description="Carbon-weighted energy penalty")
    latency_penalty: float = Field(..., description="Latency penalty with mu=0.01")
    bonuses: float = Field(..., description="Sum of bonuses or task-specific adjustments")
    total_reward: float = Field(..., description="Final reward for the step")
    correct: bool = Field(..., description="Whether the selected answer was correct")
    strategy_trace: list[str] = Field(
        default_factory=list,
        description="Ordered record of sub-actions executed during the step",
    )


class RLObservation(Observation):
    """Observation returned after reset or step."""

    query: str = Field(..., description="Current query to answer")
    cache_contents: list[str] = Field(
        default_factory=list,
        description="Queries answered correctly and available in cache",
    )
    carbon_intensity: float = Field(..., ge=0.0, le=1.0, description="Current carbon intensity")
    correct_answer: str | None = Field(
        default=None,
        description="Ground-truth answer, exposed for evaluation",
    )
    reward_details: RLReward = Field(..., description="Structured reward payload")


class RLState(State):
    """Internal environment state."""

    task_id: str = Field(..., description="Active task identifier")
    step_index: int = Field(..., ge=0, description="Index within the current task trajectory")
    query_index: int = Field(..., ge=0, description="Index of the active query")
    total_energy: float = Field(default=0.0, ge=0.0, description="Cumulative energy cost")
    total_latency: float = Field(default=0.0, ge=0.0, description="Cumulative latency cost")
    cache_contents: list[str] = Field(default_factory=list, description="Current cache contents")
    carbon_history_index: int = Field(default=0, ge=0, description="Current carbon history pointer")

    def public_payload(self) -> dict[str, Any]:
        return self.model_dump()
