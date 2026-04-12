from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from graders import GRADERS
from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import RLAction, RLObservation, RLState
from tasks import TASKS


environment = EcoLLMInferenceRoutingEnvironment()
app = FastAPI(title="Eco-LLM Inference Routing", version="1.0.0")

_TASK_META = [
    {
        "id": "easy",
        "grader": "graders:grade_task_1",
        "graders": ["graders:grade_task_1"],
        "name": "Single Query Routing",
        "title": "Single Query Routing",
        "description": "Route a single LLM query to the optimal model tier while minimising carbon footprint and latency.",
        "difficulty": "easy",
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "medium",
        "grader": "graders:grade_task_2",
        "graders": ["graders:grade_task_2"],
        "name": "Multi-Query Episode",
        "title": "Multi-Query Episode",
        "description": "Route three queries; LARGE model penalised -0.2 per use. Balance accuracy vs efficiency.",
        "difficulty": "medium",
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "hard",
        "grader": "graders:grade_task_3",
        "graders": ["graders:grade_task_3"],
        "name": "Stateful Carbon-Aware Routing",
        "title": "Stateful Carbon-Aware Routing",
        "description": "5-query episode with caching, KB lookups, cascade, and carbon-aware waiting.",
        "difficulty": "hard",
        "reward_range": [0.0, 1.0],
    },
]


class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None
    episode_id: str | None = None


class GradeRequest(BaseModel):
    task_id: str
    episode_id: str = ""
    rewards: list[float] = []
    steps: int = 0
    success: bool = False
    state: dict[str, Any] = {}
    reward: float = 0.0


@app.get("/")
def root() -> dict[str, Any]:
    return {"status": "ok", "service": "eco_llm_inference_routing"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "name": "eco_llm_inference_routing",
        "description": "Carbon-aware RL environment for multi-objective LLM query routing",
        "tasks": _TASK_META,
        "reward_range": [0.0, 1.0],
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action_schema": RLAction.model_json_schema(),
        "observation_schema": RLObservation.model_json_schema(),
        "state_schema": RLState.model_json_schema(),
    }


@app.get("/tasks")
def list_tasks() -> list[str]:
    return list(TASKS.keys())


@app.get("/tasks/meta")
def list_tasks_meta() -> dict[str, Any]:
    return {"tasks": _TASK_META, "count": len(_TASK_META)}


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    for task in _TASK_META:
        if task["id"] == task_id:
            return task
    raise HTTPException(status_code=404, detail=f"Unknown task '{task_id}'")


@app.post("/reset")
def reset(payload: ResetRequest | None = None) -> dict[str, Any]:
    task_id = "easy"
    seed = None
    episode_id = None
    if payload is not None:
        task_id = payload.task_id or "easy"
        seed = payload.seed
        episode_id = payload.episode_id
    observation = environment.reset(task_id=task_id, seed=seed, episode_id=episode_id)
    return {
        "observation": observation.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {
            "tasks": _TASK_META,
            "task_id": task_id,
        },
    }


@app.post("/step")
def step(payload: RLAction) -> dict[str, Any]:
    observation = environment.step(payload)
    return {
        "observation": observation.model_dump(),
        "reward": float(observation.reward),
        "done": bool(observation.done),
        "info": {
            "task_id": environment.current_task.task_id,
        },
    }


@app.get("/state")
def state() -> dict[str, Any]:
    return environment.state.model_dump()


def _grade_payload(request: GradeRequest) -> dict[str, Any]:
    valid = set(TASKS.keys())
    if request.task_id not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{request.task_id}'. Valid: {sorted(valid)}",
        )
    episode_data = {
        "task_id": request.task_id,
        "episode_id": request.episode_id,
        "rewards": request.rewards,
        "steps": request.steps,
        "reward": request.reward,
        "state": request.state,
    }
    score = float(GRADERS[request.task_id](episode_data))
    return {
        "task_id": request.task_id,
        "score": score,
        "reward": score,
        "normalized_reward": score,
        "success": score >= 0.5,
        "steps": request.steps or len(request.rewards),
    }


@app.post("/grade")
def grade(request: GradeRequest) -> dict[str, Any]:
    return _grade_payload(request)


@app.post("/grader")
def grader_endpoint(request: GradeRequest) -> dict[str, Any]:
    return _grade_payload(request)


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
