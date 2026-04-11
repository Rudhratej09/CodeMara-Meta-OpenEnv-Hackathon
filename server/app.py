"""
server/app.py  —  Eco-LLM Inference Routing  (OpenEnv-compliant)
================================================================
Endpoints provided automatically by create_app():
    POST /reset    start new episode
    POST /step     take action
    GET  /state    current state

Additional endpoints (Phase 2 validator requires):
    GET  /tasks    enumerate tasks with grader metadata
    POST /grade    grade a completed episode → score in [0.0, 1.0]
    GET  /health   liveness probe

FIXES applied vs previous version
───────────────────────────────────
1. Removed duplicate @app.post("/reset") — create_app() already registers it;
   duplicate caused route conflict / 500 on every POST /reset.
2. Renamed /grader → /grade (validator calls /grade, 404 otherwise).
3. Fixed grader bounds (task_1 max=1.5, task_2 max=4.5, task_3 max=7.5).
4. Added Gradio demo UI at /ui (HF Space requirement).
5. /reset no longer references undefined `env` variable.
"""
from __future__ import annotations

import uvicorn
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List

# ── OpenEnv bootstrap ────────────────────────────────────────────────────────
# create_app() registers POST /reset, POST /step, GET /state automatically.
# DO NOT add another @app.post("/reset") — it creates a duplicate route.
from openenv.core.env_server.http_server import create_app

from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import RLAction, RLObservation, Strategy, ModelChoice
from server.tasks import TASKS

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

# ── Phase 2: /tasks  ──────────────────────────────────────────────────────────
# The validator calls GET /tasks to enumerate tasks before running graders.

_TASK_META = [
    {
        "id": "task_1",
        "name": "Single Query Routing",
        "difficulty": "easy",
        "num_queries": 1,
        "max_reward": 1.5,
        "description": "Route a single query to the optimal model tier.",
        "grader": "server.app:grade_episode",
        "score_range": [0.0, 1.0],
    },
    {
        "id": "task_2",
        "name": "Multi-Query Episode",
        "difficulty": "medium",
        "num_queries": 3,
        "max_reward": 4.5,
        "description": "Route 3 queries; LARGE model penalised -0.2 per use.",
        "grader": "server.app:grade_episode",
        "score_range": [0.0, 1.0],
    },
    {
        "id": "task_3",
        "name": "Stateful Carbon-Aware Routing",
        "difficulty": "hard",
        "num_queries": 5,
        "max_reward": 7.5,
        "description": "5-query stateful episode with caching, KB, and carbon-aware waiting.",
        "grader": "server.app:grade_episode",
        "score_range": [0.0, 1.0],
    },
]

@app.get("/tasks")
def list_tasks() -> dict:
    """Enumerate all tasks with grader metadata (Phase 2 validator)."""
    return {"tasks": _TASK_META, "count": len(_TASK_META)}

@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    for t in _TASK_META:
        if t["id"] == task_id:
            return t
    raise HTTPException(status_code=404, detail=f"Unknown task '{task_id}'")

# ── Phase 2: /grade  ──────────────────────────────────────────────────────────
# The validator POSTs completed episode data here and expects score in [0, 1].
#
# GRADER BOUNDS (max achievable reward per task):
#   task_1: 1 query  × 1.5 max/query = 1.5
#   task_2: 3 queries × 1.5 max/query = 4.5
#   task_3: 5 queries × 1.5 max/query = 7.5
#
# Max reward per query = score(1.0) + cache_bonus(0.5) - 0 penalties = 1.5
# This is tight but achievable when the query is already in cache.

_MAX_REWARDS = {"task_1": 1.5, "task_2": 4.5, "task_3": 7.5}

def grade_episode(task_id: str, total_reward: float) -> float:
    """Return normalised score in [0.0, 1.0]."""
    max_r = _MAX_REWARDS.get(task_id, 1.5)
    return float(min(max(total_reward / max_r, 0.0), 1.0))

class GradeRequest(BaseModel):
    task_id: str
    episode_id: str = ""
    rewards: List[float] = []
    steps: int = 0
    success: bool = False

@app.post("/grade")
def grade(request: GradeRequest) -> dict:
    """
    Grade a completed episode.  Returns score in [0.0, 1.0].

    Body:
        task_id    : "task_1" | "task_2" | "task_3"
        rewards    : per-step reward list
        steps      : total steps taken
        episode_id : (optional) for logging
    """
    valid = set(_MAX_REWARDS)
    if request.task_id not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{request.task_id}'. Valid: {sorted(valid)}",
        )
    total = sum(request.rewards)
    score = grade_episode(request.task_id, total)
    return {
        "task_id":      request.task_id,
        "score":        round(score, 4),
        "success":      score >= 0.5,
        "total_reward": round(total, 4),
        "steps":        request.steps,
        "details": {
            "per_step_rewards": [round(r, 4) for r in request.rewards],
            "max_possible_reward": _MAX_REWARDS[request.task_id],
        },
    }

# ── Optional: replay-based grader (actions → score) ──────────────────────────
# Some validators submit action sequences rather than pre-computed rewards.

class GraderAction(BaseModel):
    strategy: str
    model_choice: str
    exit_flag: bool = False

class ReplayRequest(BaseModel):
    task_id: str
    actions: List[GraderAction]

@app.post("/grade/replay")
def grade_replay(request: ReplayRequest) -> dict:
    """Run action sequence and grade. Returns score in [0.0, 1.0]."""
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

    total = sum(rewards)
    score = grade_episode(request.task_id, total)
    return {
        "task_id": request.task_id,
        "score":   round(score, 4),
        "success": score >= 0.5,
        "reward":  round(total, 4),
        "steps":   len(rewards),
    }

# ── Gradio demo UI at /ui ─────────────────────────────────────────────────────
# Mounted at /ui so it DOES NOT override /reset, /step, /grade, /tasks.

try:
    import gradio as gr
    from gradio.routes import mount_gradio_app

    def _ser(env: EcoLLMInferenceRoutingEnvironment) -> dict:
        s = env._state
        return {
            "task_id": s.task_id, "step_count": s.step_count,
            "step_index": s.step_index, "query_index": s.query_index,
            "total_energy": float(s.total_energy), "total_latency": float(s.total_latency),
            "cache_contents": list(s.cache_contents), "carbon_history_index": s.carbon_history_index,
        }

    def _restore(env: EcoLLMInferenceRoutingEnvironment, s: dict) -> None:
        env._state.task_id = str(s["task_id"])
        env._state.step_count = int(s["step_count"])
        env._state.step_index = int(s["step_index"])
        env._state.query_index = int(s["query_index"])
        env._state.total_energy = float(s["total_energy"])
        env._state.total_latency = float(s["total_latency"])
        env._state.cache_contents = list(s["cache_contents"])
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
                strategy=Strategy(strategy), model_choice=ModelChoice(model), exit_flag=exit_flag))
            rd = new_obs.reward_details
            rewards = list(st.get("rewards", [])) + [round(rd.total_reward, 4)]
            total = float(st.get("total_reward", 0.0)) + rd.total_reward
            history = list(st.get("history", []))
            history.append(f"Step {len(history)+1}: {strategy}+{model} "
                           f"→ {rd.total_reward:+.3f} ({'✓' if rd.correct else '✗'})")
            log = "\n".join(history) + "\n\n"
            if new_obs.done:
                sc = grade_episode(str(st["task_id"]), total)
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
            strat = gr.Dropdown(["NONE","USE_CACHE","DO_CASCADE","EARLY_EXIT","WAIT","CALL_KB"],
                                value="NONE", label="Strategy")
            mdl   = gr.Dropdown(["SMALL","MEDIUM","LARGE"], value="SMALL", label="Model")
            exitf = gr.Checkbox(label="exit_flag", value=False)
        stp = gr.Button("▶️ Step", variant="secondary")
        rwd = gr.Textbox(label="Reward", interactive=False)
        tot = gr.Number(label="Total Reward", interactive=False)
        gr.Markdown(
            "### Formula\n"
            "`reward = score − (carbon × energy) − (0.1 × latency) + bonuses`\n\n"
            "| Model  | Energy | Latency | | Strategy  | Bonus/Penalty |\n"
            "|--------|--------|---------|--|-----------|---------------|\n"
            "| SMALL  | 0.1    | 1.0     | | USE_CACHE | +0.5          |\n"
            "| MEDIUM | 0.3    | 2.0     | | CALL_KB   | cheap         |\n"
            "| LARGE  | 0.6    | 5.0     | | WAIT      | skip carbon   |"
        )
        rst.click(reset_demo, [task_dd], [obs_box, rwd, tot, ds])
        stp.click(step_demo, [strat, mdl, exitf, ds], [obs_box, rwd, tot, ds])

    app = mount_gradio_app(app, demo, path="/ui")

except ImportError:
    pass   # Gradio optional — API works fine without it


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
