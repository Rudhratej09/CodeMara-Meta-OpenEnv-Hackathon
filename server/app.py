"""Full FastAPI app for the Eco-LLM Inference Routing environment."""
from __future__ import annotations

import uvicorn
import gradio as gr
from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from openenv.core.env_server.http_server import create_app

# Your local logic imports
from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import RLAction, RLObservation, Strategy, ModelChoice

# 1. Create the Master App FIRST
app = FastAPI()

# ✅ REQUIRED FOR EVALUATOR
@app.get("/health")
def health():
    return {"status": "ok"}

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