from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from graders import GRADERS, grade_task_1
from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import RLAction, RLObservation, RLState
from tasks import TASKS


environment = EcoLLMInferenceRoutingEnvironment()
app = FastAPI(title="Eco-LLM Inference Routing", version="1.0.0")

_TASK_META = [
    {
        "id": "task_1",
        "task_id": "task_1",
        "name": "Single Query Routing",
        "title": "Single Query Routing",
        "difficulty": "easy",
        "description": "Route a single LLM query to the optimal model tier while minimising carbon footprint and latency.",
        "max_steps": 10,
        "grader": "graders:grade_task_1",
        "graders": ["graders:grade_task_1"],
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "task_2",
        "task_id": "task_2",
        "name": "Multi-Query Episode",
        "title": "Multi-Query Episode",
        "difficulty": "medium",
        "description": "Route 3 queries; LARGE model penalised -0.2 per use.",
        "max_steps": 20,
        "grader": "graders:grade_task_2",
        "graders": ["graders:grade_task_2"],
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "task_3",
        "task_id": "task_3",
        "name": "Stateful Carbon-Aware Routing",
        "title": "Stateful Carbon-Aware Routing",
        "difficulty": "hard",
        "description": "5-query episode: caching, KB lookups, cascade, carbon-aware waiting.",
        "max_steps": 50,
        "grader": "graders:grade_task_3",
        "graders": ["graders:grade_task_3"],
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


class ReplayAction(BaseModel):
    strategy: str
    model_choice: str
    exit_flag: bool = False


class ReplayRequest(BaseModel):
    task_id: str
    actions: list[ReplayAction]


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
    task_id = "task_1"
    seed = None
    episode_id = None
    if payload is not None:
        task_id = payload.task_id or "task_1"
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
    fn = GRADERS.get(request.task_id)
    if fn is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{request.task_id}'. Valid: {sorted(GRADERS)}",
        )
    total_reward = sum(request.rewards)
    score = float(fn(request.state, total_reward))
    return {
        "score": round(score, 4),
        "task_id": request.task_id,
        "success": score >= 0.5,
        "total_reward": round(total_reward, 4),
        "steps": request.steps,
        "reward": round(score, 4),
        "normalized_reward": round(score, 4),
    }


@app.post("/grader")
def grader_endpoint(request: GradeRequest) -> dict[str, Any]:
    return _grade_payload(request)


@app.post("/grade")
def grade_endpoint(request: GradeRequest) -> dict[str, Any]:
    return _grade_payload(request)


@app.post("/grader/replay")
def grader_replay(request: ReplayRequest) -> dict[str, Any]:
    if request.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {request.task_id}")

    env = EcoLLMInferenceRoutingEnvironment()
    obs = env.reset(task_id=request.task_id)
    rewards: list[float] = []

    for _ in request.actions:
        if obs.done:
            break
        break

    total_reward = sum(rewards)
    fn = GRADERS.get(request.task_id, grade_task_1)
    score = float(fn({}, total_reward))
    return {
        "score": round(score, 4),
        "task_id": request.task_id,
        "success": score >= 0.5,
        "total_reward": round(total_reward, 4),
        "steps": len(rewards),
    }


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
