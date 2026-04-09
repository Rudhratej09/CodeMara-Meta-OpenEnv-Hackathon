"""Full FastAPI app for the Eco-LLM Inference Routing environment."""
from __future__ import annotations

import uvicorn
import gradio as gr
from fastapi import FastAPI, HTTPException
from gradio.routes import mount_gradio_app
from openenv.core.env_server.http_server import create_app
from pydantic import BaseModel
from typing import List

# Your local logic imports
from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import RLAction, RLObservation, Strategy, ModelChoice
from server.tasks import TASKS

# ─────────────────────────────────────────
# 🎯 GRADER CONSTANTS
# Each task defines [WORST, BEST] episode-reward bounds.
# WORST = lowest achievable cumulative reward (naïve NONE+SMALL agent).
# BEST  = highest achievable cumulative reward (optimal strategy).
# score = soft_clamp( (episode_reward - WORST) / (BEST - WORST), 0.01, 0.99 )
# This guarantees scores are strictly between 0 and 1.
# ─────────────────────────────────────────
_GRADER_BOUNDS = {
    # task_1: 1 query, min_model=MEDIUM
    #   WORST: NONE+SMALL → wrong → -(0.8×0.1 + 0.1×1.0) = -0.18
    #   BEST:  EARLY_EXIT+MEDIUM → 1 - 0.8×0.3 - 0.1×2.0 + 0.1 = 0.66
    "task_1": {"worst": -0.18, "best": 0.66},
    # task_2: 3 queries (min: SMALL, MEDIUM, LARGE); large_model_penalty=-0.2
    #   WORST: NONE+SMALL → q1 correct (SMALL ok), q2/q3 wrong
    #          = (1-0.08-0.1) + (0-0.06-0.1) + (0-0.04-0.1) = 0.52
    #   BEST:  EARLY_EXIT per query (SMALL/MEDIUM/LARGE) = 0.92+0.72+0.16 = 1.80
    "task_2": {"worst": 0.52, "best": 1.80},
    # task_3: 5 queries (min: SMALL/kb, MEDIUM, SMALL/kb, LARGE, MEDIUM)
    #   WORST: NONE+SMALL → q1/q3 correct, q2/q4/q5 wrong = 1.24
    #   BEST:  CALL_KB(q1)+EE+MEDIUM(q2)+USE_CACHE(q3)+EE+LARGE(q4)+EE+MEDIUM(q5)
    #          = 0.81+0.72+1.50+0.42+0.75 = 4.20
    "task_3": {"worst": 1.24, "best": 4.20},
}
_GRADER_EPS = 0.01


def _run_episode(task_id: str, actions: list[dict]) -> float:
    """Replay a fixed list of actions and return the normalised score."""
    env = EcoLLMInferenceRoutingEnvironment()
    obs = env.reset(task_id=task_id)
    episode_reward = 0.0

    for raw_action in actions:
        if obs.done:
            break
        action = RLAction(
            strategy=Strategy(raw_action["strategy"]),
            model_choice=ModelChoice(raw_action["model_choice"]),
            exit_flag=bool(raw_action.get("exit_flag", False)),
        )
        obs = env.step(action)
        episode_reward += obs.reward_details.total_reward

    bounds = _GRADER_BOUNDS[task_id]
    worst, best = bounds["worst"], bounds["best"]
    normalised = (episode_reward - worst) / (best - worst)
    score = max(_GRADER_EPS, min(1.0 - _GRADER_EPS, normalised))
    return round(score, 6)


# ─────────────────────────────────────────
# 🏗️ REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────

class GraderAction(BaseModel):
    strategy: str
    model_choice: str
    exit_flag: bool = False


class GraderRequest(BaseModel):
    task_id: str
    actions: List[GraderAction]


class GraderResponse(BaseModel):
    task_id: str
    score: float          # strictly in (0, 1)
    episode_reward: float
    worst_bound: float
    best_bound: float


# 1. Create the Master App FIRST
app = FastAPI()

# ✅ REQUIRED FOR EVALUATOR
@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────
# 📋 GET /tasks  – task catalogue
# ─────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """Return the catalogue of available tasks with grader bounds."""
    return {
        task_id: {
            "task_id": spec.task_id,
            "difficulty": spec.difficulty,
            "description": spec.description,
            "num_queries": len(spec.queries),
            "grader_bounds": _GRADER_BOUNDS[task_id],
        }
        for task_id, spec in TASKS.items()
    }


# ─────────────────────────────────────────
# 🏆 POST /grader  – deterministic grader
# ─────────────────────────────────────────

@app.post("/grader", response_model=GraderResponse)
def grade_episode(request: GraderRequest):
    """
    Deterministically grade a completed episode.

    Supply the task_id and the ordered list of actions taken during the
    episode.  The endpoint replays those actions in a fresh environment
    instance and returns a score strictly between 0 and 1.

    Score formula
    -------------
    score = soft_clamp(
        (episode_reward - WORST) / (BEST - WORST),
        lo=0.01, hi=0.99
    )

    where WORST and BEST are the empirically determined lower and upper
    bounds for each task (see GET /tasks for exact values).
    """
    task_id = request.task_id
    if task_id not in _GRADER_BOUNDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id {task_id!r}. Valid: {list(_GRADER_BOUNDS)}"
        )

    raw_actions = [a.model_dump() for a in request.actions]

    # Replay to measure episode_reward independently for the response
    env = EcoLLMInferenceRoutingEnvironment()
    obs = env.reset(task_id=task_id)
    episode_reward = 0.0
    for raw_action in raw_actions:
        if obs.done:
            break
        action = RLAction(
            strategy=Strategy(raw_action["strategy"]),
            model_choice=ModelChoice(raw_action["model_choice"]),
            exit_flag=bool(raw_action.get("exit_flag", False)),
        )
        obs = env.step(action)
        episode_reward += obs.reward_details.total_reward

    bounds = _GRADER_BOUNDS[task_id]
    worst, best = bounds["worst"], bounds["best"]
    normalised = (episode_reward - worst) / (best - worst)
    score = max(_GRADER_EPS, min(1.0 - _GRADER_EPS, normalised))

    return GraderResponse(
        task_id=task_id,
        score=round(score, 6),
        episode_reward=round(episode_reward, 6),
        worst_bound=worst,
        best_bound=best,
    )

# ─────────────────────────────────────────
# 🎯 GRADIO UI DEFINITION
# ─────────────────────────────────────────

def reset_demo(task_id: str, state: dict) -> tuple:
    demo_env = EcoLLMInferenceRoutingEnvironment()
    obs = demo_env.reset(task_id=task_id)
    updated_state = {"env": demo_env, "obs": obs, "history": [], "total_reward": 0.0}
    
    log = f"🔄 Reset — Task: {task_id}\n📋 Query: {obs.query}\n🌍 Carbon: {obs.carbon_intensity:.2f}"
    return log, "", 0.0, updated_state

def step_demo(strategy: str, model: str, exit_flag: bool, state: dict) -> tuple:
    demo_env = state.get("env")
    if not demo_env: return "⚠️ Reset first!", "", 0.0, state
    
    action = RLAction(strategy=Strategy(strategy), model_choice=ModelChoice(model), exit_flag=exit_flag)
    new_obs = demo_env.step(action)
    rd = new_obs.reward_details
    total_reward = state.get("total_reward", 0.0) + rd.total_reward
    
    log = f"Step Result: {'✅' if rd.correct else '❌'}\nReward: {rd.total_reward:+.3f}"
    return log, str(rd), round(total_reward, 3), {**state, "obs": new_obs, "total_reward": total_reward}

with gr.Blocks(title="Eco-LLM Routing") as demo:
    demo_state = gr.State({"env": None, "obs": None, "history": [], "total_reward": 0.0})
    gr.Markdown("# 🌱 Eco-LLM Inference Routing")
    
    with gr.Row():
        task_selector = gr.Dropdown(choices=["task_1", "task_2", "task_3"], value="task_1", label="Task")
        reset_btn = gr.Button("🔄 Reset", variant="primary")
    
    obs_box = gr.Textbox(label="State", lines=4, interactive=False)
    
    with gr.Row():
        strategy_dd = gr.Dropdown(choices=["NONE", "USE_CACHE", "DO_CASCADE", "EARLY_EXIT", "WAIT", "CALL_KB"], value="NONE", label="Strategy")
        model_dd = gr.Dropdown(choices=["SMALL", "MEDIUM", "LARGE"], value="SMALL", label="Model")
        exit_cb = gr.Checkbox(label="Exit Flag", value=False)
    
    step_btn = gr.Button("▶️ Step")
    reward_box = gr.Textbox(label="Step Detail", interactive=False)
    total_num = gr.Number(label="Total Reward", interactive=False)

    reset_btn.click(reset_demo, [task_selector, demo_state], [obs_box, reward_box, total_num, demo_state])
    step_btn.click(step_demo, [strategy_dd, model_dd, exit_cb, demo_state], [obs_box, reward_box, total_num, demo_state])

# ─────────────────────────────────────────
# 🚀 ROUTING SETUP
# ─────────────────────────────────────────

# 2. MOUNT GRADIO TO THE ROOT ("/") 
# This ensures Gradio's internal JS/CSS assets load without path issues.
app = mount_gradio_app(app, demo, path="/")

# 3. Create and mount OpenEnv to "/api" 
# This keeps the environment logic accessible but gets its UI out of the way.
openenv_logic = create_app(
    EcoLLMInferenceRoutingEnvironment,
    RLAction,
    RLObservation,
    env_name="eco_llm_inference_routing",
    max_concurrent_envs=4,
)
app.mount("/api", openenv_logic)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()