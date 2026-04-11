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
    # We use explicit defaults to ensure the UI renders dropdowns
    strategy: Strategy = Field(default=Strategy.NONE, description="Routing strategy to apply")
    model_choice: ModelChoice = Field(default=ModelChoice.SMALL, description="Primary model size")
    exit_flag: bool = Field(default=False, description="Terminate cascade early")

class RLReward(BaseModel):
    score: float = 0.0
    energy_cost: float = 0.0
    latency_cost: float = 0.0
    carbon_penalty: float = 0.0
    latency_penalty: float = 0.0
    bonuses: float = 0.0
    total_reward: float = 0.0
    correct: bool = False
    strategy_trace: list[str] = Field(default_factory=list)

class RLObservation(Observation):
    query: str = ""
    cache_contents: list[str] = Field(default_factory=list)
    carbon_intensity: float = 0.0
    reward_details: RLReward = Field(default_factory=RLReward)
    kb_available: bool = False
    reward: float = 0.0
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
class RLState(State):
    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    step_index: int = 0
    query_index: int = 0
    total_energy: float = 0.0
    total_latency: float = 0.0
    cache_contents: list[str] = Field(default_factory=list)
    carbon_history_index: int = 0

    def public_payload(self) -> dict[str, Any]:
        return self.model_dump()
