"""
server/app.py  —  Eco-LLM Inference Routing  (OpenEnv-compliant)
================================================================
Endpoints provided automatically by create_app():
    POST /reset    start new episode
    POST /step     take action
    GET  /state    current state

Additional endpoints:
    GET  /tasks    enumerate tasks with grader metadata
    POST /grade    grade a completed episode → score in [0.01, 0.99]
    GET  /health   liveness probe

FIXES (Phase 2 deep validation)
─────────────────────────────────
1. Grader references are now CLASS-based (server.graders:Task1Grader) not
   function-based. The validator instantiates Task1Grader() then calls it —
   a plain function cannot be instantiated this way.
2. All scores clamped to [0.01, 0.99] — Phase 2 strict inequality check
   rejects exactly 0.0 or 1.0 with "Score Out of Range".
3. /grade endpoint uses the same grader classes as openenv.yaml so
   replay-based and reward-list-based grading are consistent.
"""
from __future__ import annotations

import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List

from openenv.core.env_server.http_server import create_app

from server.env import EcoLLMInferenceRoutingEnvironment
from server.rubrics import Task1Rubric, Task2Rubric, Task3Rubric, get_grader
from server.models import RLAction, RLObservation, Strategy, ModelChoice
from server.tasks import TASKS

# create_app() registers POST /reset, POST /step, GET /state automatically.
app = create_app(
    EcoLLMInferenceRoutingEnvironment,
    RLAction,
    RLObservation,
    env_name="eco_llm_inference_routing",
    max_concurrent_envs=4,
)


# ── Utility endpoints ─────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "eco_llm_inference_routing", "ui": "/ui"}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


# ── GET /tasks ────────────────────────────────────────────────────────────────

_TASK_META = [
    {
        "id": "task_1",
        "grader": "rubrics:Task1Rubric",
        "name": "Single Query Routing",
        "description": "Route a single LLM query to the optimal model tier while minimising carbon footprint and latency.",
        "difficulty": "easy",
    },
    {
        "id": "task_2",
        "grader": "rubrics:Task2Rubric",
        "name": "Multi-Query Episode",
        "description": "Route three queries; LARGE model penalised -0.2 per use. Balance accuracy vs efficiency.",
        "difficulty": "medium",
    },
    {
        "id": "task_3",
        "grader": "rubrics:Task3Rubric",
        "name": "Stateful Carbon-Aware Routing",
        "description": "5-query episode with caching, KB lookups, cascade, and carbon-aware waiting.",
        "difficulty": "hard",
    },
]


@app.get("/tasks")
def list_tasks() -> dict:
    """Enumerate all tasks (Phase 2 validator calls this)."""
    return {"tasks": _TASK_META, "count": len(_TASK_META)}


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    for t in _TASK_META:
        if t["id"] == task_id:
            return t
    raise HTTPException(status_code=404, detail=f"Unknown task '{task_id}'")


# ── POST /grade ───────────────────────────────────────────────────────────────
# Uses the same grader classes referenced in openenv.yaml so behaviour is
# identical whether the validator calls the class directly or via this endpoint.

class GradeRequest(BaseModel):
    task_id: str
    episode_id: str = ""
    rewards: List[float] = []
    steps: int = 0
    success: bool = False


@app.post("/grade")
def grade(request: GradeRequest) -> dict:
    """
    Grade a completed episode.  Returns score in [0.01, 0.99].

    Body: task_id, rewards (per-step list), steps, episode_id (optional)
    """
    valid = {"task_1", "task_2", "task_3"}
    if request.task_id not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{request.task_id}'. Valid: {sorted(valid)}",
        )
    grader = get_grader(request.task_id)
    result = grader({
        "task_id":    request.task_id,
        "episode_id": request.episode_id,
        "rewards":    request.rewards,
        "steps":      request.steps,
    })
    return result


# ── POST /grade/replay ────────────────────────────────────────────────────────

class GraderAction(BaseModel):
    strategy: str
    model_choice: str
    exit_flag: bool = False


class ReplayRequest(BaseModel):
    task_id: str
    actions: List[GraderAction]


@app.post("/grade/replay")
def grade_replay(request: ReplayRequest) -> dict:
    """Run an action sequence and grade it."""
    if request.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {request.task_id}")

    env = EcoLLMInferenceRoutingEnvironment()
    obs = env.reset(task_id=request.task_id)
    rewards: List[float] = []

    for a in request.actions:
        if obs.done:
            break
        try:
            action = RLAction(
                strategy=Strategy(a.strategy),
                model_choice=ModelChoice(a.model_choice),
                exit_flag=a.exit_flag,
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid action: {exc}")
        obs = env.step(action)
        rewards.append(obs.reward_details.total_reward)

    grader = get_grader(request.task_id)
    result = grader({"task_id": request.task_id, "rewards": rewards, "steps": len(rewards)})
    return result


# ── Gradio demo UI at /ui ─────────────────────────────────────────────────────

try:
    import gradio as gr
    from gradio.routes import mount_gradio_app

    def _ser(env: EcoLLMInferenceRoutingEnvironment) -> dict:
        s = env._state
        return {
            "task_id": s.task_id, "step_count": s.step_count,
            "step_index": s.step_index, "query_index": s.query_index,
            "total_energy": float(s.total_energy), "total_latency": float(s.total_latency),
            "cache_contents": list(s.cache_contents),
            "carbon_history_index": s.carbon_history_index,
        }

    def _restore(env: EcoLLMInferenceRoutingEnvironment, s: dict) -> None:
        env._state.task_id              = str(s["task_id"])
        env._state.step_count           = int(s["step_count"])
        env._state.step_index           = int(s["step_index"])
        env._state.query_index          = int(s["query_index"])
        env._state.total_energy         = float(s["total_energy"])
        env._state.total_latency        = float(s["total_latency"])
        env._state.cache_contents       = list(s["cache_contents"])
        env._state.carbon_history_index = int(s["carbon_history_index"])

    def reset_demo(task_id: str) -> tuple:
        try:
            e = EcoLLMInferenceRoutingEnvironment()
            obs = e.reset(task_id=task_id)
            st = {"task_id": task_id, "env_state": _ser(e),
                  "history": [], "rewards": [], "total_reward": 0.0, "active": True}
            log = (f"Reset — {task_id} ({e.current_task.difficulty})\n"
                   f"Query: {obs.query}\nCarbon: {obs.carbon_intensity:.2f}\n"
                   f"Cache: {obs.cache_contents or 'empty'}\nKB: {obs.kb_available}")
            return log, "", 0.0, st
        except Exception as exc:
            return f"Error: {exc}", "", 0.0, {}

    def step_demo(strategy: str, model: str, exit_flag: bool, st: dict) -> tuple:
        try:
            if not st.get("active") or not st.get("env_state"):
                return "Click Reset first.", "", 0.0, st
            e = EcoLLMInferenceRoutingEnvironment()
            e.reset(task_id=str(st["task_id"]))
            _restore(e, st["env_state"])
            obs = e._build_observation(reward=e._empty_reward(["r"]), done=e._episode_complete)
            if obs.done:
                total = round(float(st.get("total_reward", 0.0)), 3)
                return "Episode complete. Click Reset.", "", total, {**st, "active": False}
            new_obs = e.step(RLAction(
                strategy=Strategy(strategy),
                model_choice=ModelChoice(model),
                exit_flag=exit_flag,
            ))
            rd      = new_obs.reward_details
            rewards = list(st.get("rewards", [])) + [round(rd.total_reward, 4)]
            total   = float(st.get("total_reward", 0.0)) + rd.total_reward
            history = list(st.get("history", []))
            history.append(f"Step {len(history)+1}: {strategy}+{model} "
                           f"→ {rd.total_reward:+.3f} ({'✓' if rd.correct else '✗'})")
            log = "\n".join(history) + "\n\n"
            if new_obs.done:
                grader = get_grader(str(st["task_id"]))
                sc = grader({"task_id": str(st["task_id"]), "rewards": rewards})["score"]
                log += f"Done! Score: {sc:.3f}"
            else:
                log += (f"Query: {new_obs.query}\nCarbon: {new_obs.carbon_intensity:.2f}\n"
                        f"Cache: {new_obs.cache_contents or 'empty'}\nKB: {new_obs.kb_available}")
            bkd = (f"score={rd.score:.1f} energy={rd.energy_cost:.2f} "
                   f"carbon_pen={rd.carbon_penalty:.3f} lat_pen={rd.latency_penalty:.3f} "
                   f"bonuses={rd.bonuses:.2f} total={rd.total_reward:+.3f}")
            updated = {"task_id": st["task_id"], "env_state": _ser(e),
                       "history": history, "rewards": rewards,
                       "total_reward": round(total, 3), "active": not new_obs.done}
            return log, bkd, round(total, 3), updated
        except Exception as exc:
            return f"Error: {exc}", "", float(st.get("total_reward", 0.0)), st

    with gr.Blocks(title="Eco-LLM Routing") as demo:
        ds = gr.State({"task_id": "task_1", "env_state": None, "history": [],
                       "rewards": [], "total_reward": 0.0, "active": False})
        gr.Markdown(
            "# 🌱 Eco-LLM Inference Routing\n"
            "Carbon-aware LLM query routing — balance accuracy, energy, latency & carbon.\n\n"
            "API: `POST /reset` · `POST /step` · `GET /tasks` · `POST /grade`"
        )
        with gr.Row():
            task_dd = gr.Dropdown(["task_1", "task_2", "task_3"], value="task_1", label="Task")
            rst = gr.Button("🔄 Reset", variant="primary")
        obs_box = gr.Textbox(label="State", lines=6, interactive=False)
        gr.Markdown("### Action")
        with gr.Row():
            strat = gr.Dropdown(
                ["NONE", "USE_CACHE", "DO_CASCADE", "EARLY_EXIT", "WAIT", "CALL_KB"],
                value="NONE", label="Strategy",
            )
            mdl   = gr.Dropdown(["SMALL", "MEDIUM", "LARGE"], value="SMALL", label="Model")
            exitf = gr.Checkbox(label="exit_flag", value=False)
        stp = gr.Button("▶️ Step", variant="secondary")
        rwd = gr.Textbox(label="Reward", interactive=False)
        tot = gr.Number(label="Total Reward", interactive=False)
        gr.Markdown(
            "### Formula\n"
            "`reward = score − (carbon × energy) − (0.1 × latency) + bonuses`\n\n"
            "| Model  | Energy | Latency | | Strategy  | Bonus/Penalty |\n"
            "|--------|--------|---------|-|-----------|---------------|\n"
            "| SMALL  | 0.1    | 1.0     | | USE_CACHE | +0.5          |\n"
            "| MEDIUM | 0.3    | 2.0     | | CALL_KB   | cheap         |\n"
            "| LARGE  | 0.6    | 5.0     | | WAIT      | skip carbon   |"
        )
        rst.click(reset_demo, [task_dd], [obs_box, rwd, tot, ds])
        stp.click(step_demo, [strat, mdl, exitf, ds], [obs_box, rwd, tot, ds])

    app = mount_gradio_app(app, demo, path="/ui")

except ImportError:
    pass


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
