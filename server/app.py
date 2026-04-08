
"""FastAPI app for the Eco-LLM Inference Routing environment."""
from __future__ import annotations

from openenv.core.env_server.http_server import create_app
from fastapi.responses import RedirectResponse

from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import RLAction, RLObservation, Strategy, ModelChoice

import gradio as gr
from gradio.routes import mount_gradio_app


# 🔥 OpenEnv app (DO NOT CHANGE)
app = create_app(
    EcoLLMInferenceRoutingEnvironment,
    RLAction,
    RLObservation,
    env_name="eco_llm_inference_routing",
    max_concurrent_envs=4,
)


# ✅ REQUIRED FOR EVALUATOR
@app.get("/health")
def health():
    return {"status": "ok"}


# 🔥 FORCE ROOT → YOUR UI (IMPORTANT FIX)
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


# ─────────────────────────────────────────
# 🎯 YOUR UI
# ─────────────────────────────────────────

def reset_demo(task_id: str, state: dict) -> tuple:
    demo_env = EcoLLMInferenceRoutingEnvironment()
    obs = demo_env.reset(task_id=task_id)

    updated_state = {
        "env": demo_env,
        "obs": obs,
        "history": [],
        "total_reward": 0.0,
    }

    log = f"🔄 Reset — Task: {task_id} ({demo_env.current_task.difficulty})\n"
    log += f"📋 Query: {obs.query}\n"
    log += f"🌍 Carbon intensity: {obs.carbon_intensity:.2f}\n"
    log += f"💾 Cache: {obs.cache_contents or 'empty'}\n"
    log += f"🔌 KB available: {obs.kb_available}"

    return log, "", 0.0, updated_state


def step_demo(strategy: str, model: str, exit_flag: bool, state: dict) -> tuple:
    demo_env = state.get("env")

    if demo_env is None or state.get("obs") is None:
        return "⚠️ Click Reset first!", "", 0.0, state

    obs = state["obs"]

    if obs.done:
        return "✅ Episode done — click Reset to start again.", "", state.get("total_reward", 0.0), state

    action = RLAction(
        strategy=Strategy(strategy),
        model_choice=ModelChoice(model),
        exit_flag=exit_flag,
    )

    new_obs = demo_env.step(action)
    rd = new_obs.reward_details
    total_reward = state.get("total_reward", 0.0) + rd.total_reward

    history = list(state.get("history", []))

    history.append(
        f"Step {len(history)+1}: {strategy}+{model} → reward={rd.total_reward:+.3f} "
        f"({'✅' if rd.correct else '❌'})"
    )

    log = "\n".join(history) + "\n\n"

    if not new_obs.done:
        log += f"📋 Next query: {new_obs.query}\n"
        log += f"🌍 Carbon intensity: {new_obs.carbon_intensity:.2f}\n"
        log += f"💾 Cache: {new_obs.cache_contents or 'empty'}\n"
        log += f"🔌 KB available: {new_obs.kb_available}"
    else:
        log += "✅ Episode complete!"

    breakdown = (
        f"score={rd.score:.1f}  "
        f"energy_cost={rd.energy_cost:.2f}  "
        f"carbon_penalty={rd.carbon_penalty:.3f}  "
        f"latency_penalty={rd.latency_penalty:.3f}  "
        f"bonuses={rd.bonuses:.2f}  "
        f"→ total={rd.total_reward:+.3f}"
    )

    updated_state = {
        "env": demo_env,
        "obs": new_obs,
        "history": history,
        "total_reward": total_reward,
    }

    return log, breakdown, round(total_reward, 3), updated_state


# 🔥 UI
with gr.Blocks(title="Eco-LLM Routing") as demo:
    demo_state = gr.State({"env": None, "obs": None, "history": [], "total_reward": 0.0})

    gr.Markdown("# 🌱 Eco-LLM Inference Routing Environment")

    with gr.Row():
        task_selector = gr.Dropdown(
            choices=["task_1", "task_2", "task_3"],
            value="task_1",
            label="Task"
        )
        reset_btn = gr.Button("🔄 Reset", variant="primary")

    obs_box = gr.Textbox(label="State", lines=6, interactive=False)

    with gr.Row():
        strategy_dd = gr.Dropdown(
            choices=["NONE", "USE_CACHE", "DO_CASCADE", "EARLY_EXIT", "WAIT", "CALL_KB"],
            value="NONE",
            label="Strategy"
        )
        model_dd = gr.Dropdown(
            choices=["SMALL", "MEDIUM", "LARGE"],
            value="SMALL",
            label="Model"
        )
        exit_cb = gr.Checkbox(label="Exit Flag", value=False)

    step_btn = gr.Button("▶️ Step")
    reward_box = gr.Textbox(label="Reward", interactive=False)
    total_num = gr.Number(label="Total", interactive=False)

    reset_btn.click(reset_demo, [task_selector, demo_state], [obs_box, reward_box, total_num, demo_state])
    step_btn.click(step_demo, [strategy_dd, model_dd, exit_cb, demo_state], [obs_box, reward_box, total_num, demo_state])


# 🔥 Mount UI safely
app = mount_gradio_app(app, demo, path="/ui")


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
