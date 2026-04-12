from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException
from openenv.core.env_server.http_server import create_app
from pydantic import BaseModel
import uvicorn

from graders import GRADERS, grade_easy
from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import ModelChoice, RLAction, RLObservation, RLState, Strategy
from tasks import TASKS as VALIDATOR_TASKS
from tasks import TASK_REGISTRY


app = create_app(
    EcoLLMInferenceRoutingEnvironment,
    RLAction,
    RLObservation,
    env_name="eco_llm_inference_routing",
    max_concurrent_envs=4,
)

_TASK_META = [
    {
        **task,
        "grader": task.get("grader_path", "unknown"),
        "graders": task.get("grader_paths", []),
    }
    for task in VALIDATOR_TASKS
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
def list_tasks() -> list[dict[str, Any]]:
    return _TASK_META


@app.get("/task_ids")
def list_task_ids() -> list[str]:
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


# ── Gradio UI (mounted at /ui — does NOT override any API routes) ─────────────
try:
    import gradio as gr
    from gradio.routes import mount_gradio_app
    from server.env import EcoLLMInferenceRoutingEnvironment
    from server.models import RLAction, Strategy, ModelChoice

    _env_store: dict = {}

    def ui_reset(task_id: str) -> str:
        env = EcoLLMInferenceRoutingEnvironment()
        obs = env.reset(task_id=task_id)
        _env_store["env"] = env
        _env_store["rewards"] = []
        _env_store["task_id"] = task_id
        return (
            f"✅ Reset — {task_id} ({env.current_task.difficulty})\n"
            f"Query     : {obs.query}\n"
            f"Carbon    : {obs.carbon_intensity:.2f}\n"
            f"Cache     : {obs.cache_contents or 'empty'}\n"
            f"KB avail  : {obs.kb_available}\n"
            f"Done      : {obs.done}"
        )

    def ui_step(strategy: str, model: str, exit_flag: bool) -> str:
        env = _env_store.get("env")
        if env is None:
            return "⚠️  Click Reset first."
        if env._episode_complete:
            rewards = _env_store.get("rewards", [])
            total = sum(rewards)
            return f"🏁 Episode complete.\nTotal reward: {total:.3f}\nRewards: {rewards}"
        try:
            obs = env.step(RLAction(
                strategy=Strategy(strategy),
                model_choice=ModelChoice(model),
                exit_flag=exit_flag,
            ))
        except Exception as exc:
            return f"❌ Error: {exc}"
        _env_store.setdefault("rewards", []).append(obs.reward_details.total_reward)
        rd = obs.reward_details
        out = (
            f"step reward : {rd.total_reward:+.3f}  ({'✓ correct' if rd.correct else '✗ wrong'})\n"
            f"  score     : {rd.score}\n"
            f"  carbon pen: {rd.carbon_penalty:.3f}\n"
            f"  lat pen   : {rd.latency_penalty:.3f}\n"
            f"  bonuses   : {rd.bonuses:.3f}\n"
            f"  trace     : {rd.strategy_trace}\n\n"
        )
        if obs.done:
            total = sum(_env_store["rewards"])
            out += f"🏁 Done! Total reward: {total:.3f}"
        else:
            out += (
                f"Next query : {obs.query}\n"
                f"Carbon     : {obs.carbon_intensity:.2f}\n"
                f"Cache      : {obs.cache_contents or 'empty'}\n"
                f"KB avail   : {obs.kb_available}"
            )
        return out

    with gr.Blocks(title="🌱 Eco-LLM Routing") as _demo:
        gr.Markdown(
            "# 🌱 Eco-LLM Inference Routing\n"
            "Carbon-aware LLM query routing — balance **accuracy**, **energy**, **latency** & **carbon**.\n\n"
            "**API endpoints:** `POST /reset` · `POST /step` · `GET /tasks` · `POST /grade`"
        )

        with gr.Row():
            task_dd = gr.Dropdown(
                choices=["task_1", "task_2", "task_3"],
                value="task_1",
                label="Task",
            )
            reset_btn = gr.Button("🔄 Reset", variant="primary")

        state_box = gr.Textbox(label="Environment State", lines=8, interactive=False)

        gr.Markdown("### Take a Step")
        with gr.Row():
            strat_dd = gr.Dropdown(
                choices=["NONE", "USE_CACHE", "DO_CASCADE", "EARLY_EXIT", "WAIT", "CALL_KB"],
                value="NONE",
                label="Strategy",
            )
            model_dd = gr.Dropdown(
                choices=["SMALL", "MEDIUM", "LARGE"],
                value="SMALL",
                label="Model",
            )
            exit_cb = gr.Checkbox(label="exit_flag", value=False)
        step_btn = gr.Button("▶️ Step", variant="secondary")
        result_box = gr.Textbox(label="Step Result", lines=10, interactive=False)

        gr.Markdown(
            "### Reward Formula\n"
            "`reward = score − (carbon × energy) − (0.1 × latency) + bonuses`\n\n"
            "| Model | Energy | Latency | | Strategy | Bonus |\n"
            "|-------|--------|---------|--|----------|-------|\n"
            "| SMALL | 0.1 | 1.0 | | USE_CACHE | +0.5 |\n"
            "| MEDIUM | 0.3 | 2.0 | | EARLY_EXIT | +0.1 |\n"
            "| LARGE | 0.6 | 5.0 | | CALL_KB | low cost |"
        )

        reset_btn.click(fn=ui_reset, inputs=[task_dd], outputs=[state_box])
        step_btn.click(fn=ui_step, inputs=[strat_dd, model_dd, exit_cb], outputs=[result_box])

    app = mount_gradio_app(app, _demo, path="/ui")

except ImportError:
    pass  # Gradio optional — API works without it
