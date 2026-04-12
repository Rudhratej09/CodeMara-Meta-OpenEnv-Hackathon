from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from graders import GRADERS, grade_task_1
from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import ModelChoice, RLAction, RLObservation, RLState, Strategy
from tasks import TASK_REGISTRY


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

_ACTION_SPACE = {
    "strategy": [strategy.value for strategy in Strategy],
    "model_choice": [model.value for model in ModelChoice],
    "exit_flag": "boolean",
}

_ENDPOINTS = {
    "reset": {"method": "POST", "path": "/reset"},
    "step": {"method": "POST", "path": "/step"},
    "state": {"method": "GET", "path": "/state"},
    "tasks": {"method": "GET", "path": "/tasks"},
    "tasks_meta": {"method": "GET", "path": "/tasks/meta"},
    "metadata": {"method": "GET", "path": "/metadata"},
    "schema": {"method": "GET", "path": "/schema"},
    "grader": {"method": "POST", "path": "/grader"},
    "grade": {"method": "POST", "path": "/grade"},
    "grader_replay": {"method": "POST", "path": "/grader/replay"},
    "docs": {"method": "GET", "path": "/docs"},
}


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
    return {
        "status": "ok",
        "service": "eco_llm_inference_routing",
        "docs": "/docs",
        "metadata": "/metadata",
        "tasks": "/tasks",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "name": "eco_llm_inference_routing",
        "description": "Carbon-aware RL environment for multi-objective LLM query routing",
        "version": "1.0.0",
        "domain": "llm systems optimization",
        "deterministic": True,
        "observation_highlights": [
            "query text",
            "carbon intensity",
            "cache contents",
            "knowledge-base availability",
            "structured reward breakdown",
        ],
        "action_space": _ACTION_SPACE,
        "endpoints": _ENDPOINTS,
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
    return list(TASK_REGISTRY.keys())


@app.get("/tasks/meta")
def list_tasks_meta() -> dict[str, Any]:
    return {
        "tasks": _TASK_META,
        "count": len(_TASK_META),
        "available_task_ids": [task["id"] for task in _TASK_META],
    }


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
    if request.task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {request.task_id}")

    env = EcoLLMInferenceRoutingEnvironment()
    obs = env.reset(task_id=request.task_id)
    rewards: list[float] = []

    for action in request.actions:
        if obs.done:
            break
        try:
            obs = env.step(
                RLAction(
                    strategy=Strategy(action.strategy),
                    model_choice=ModelChoice(action.model_choice),
                    exit_flag=action.exit_flag,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid replay action: {exc}") from exc
        rewards.append(float(obs.reward))

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
