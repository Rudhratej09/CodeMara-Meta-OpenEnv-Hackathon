"""FastAPI app for the Eco-LLM Inference Routing environment."""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install dependencies before running the server."
    ) from exc

from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import RLAction, RLObservation, Strategy, ModelChoice

app = create_app(
    EcoLLMInferenceRoutingEnvironment,
    RLAction,
    RLObservation,
    env_name="eco_llm_inference_routing",
    max_concurrent_envs=4,
)

@app.get("/")
def root():
    return {"status": "ok", "ui": "/ui"}


# ── Gradio demo — mounted at /ui, does NOT replace FastAPI ───────────────────
try:
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # One shared env instance for the demo (isolated from evaluator sessions)
    _demo_env = EcoLLMInferenceRoutingEnvironment()
    _demo_state: dict = {"obs": None, "task": "task_1", "history": [], "total_reward": 0.0}

    def reset_demo(task_id: str) -> tuple:
        _demo_state["task"] = task_id
        _demo_state["history"] = []
        _demo_state["total_reward"] = 0.0
        obs = _demo_env.reset(task_id=task_id)
        _demo_state["obs"] = obs
        log = f"🔄 Reset — Task: {task_id} ({obs.metadata['difficulty']})\n"
        log += f"📋 Query: {obs.query}\n"
        log += f"🌍 Carbon intensity: {obs.carbon_intensity:.2f}\n"
        log += f"💾 Cache: {obs.cache_contents or 'empty'}\n"
        log += f"🔌 KB available: {obs.kb_available}"
        return log, "", 0.0

    def step_demo(strategy: str, model: str, exit_flag: bool) -> tuple:
        if _demo_state["obs"] is None:
            return "⚠️ Click Reset first!", "", 0.0
        obs = _demo_state["obs"]
        if obs.done:
            return "✅ Episode done — click Reset to start again.", "", _demo_state["total_reward"]

        action = RLAction(
            strategy=Strategy(strategy),
            model_choice=ModelChoice(model),
            exit_flag=exit_flag,
        )
        new_obs = _demo_env.step(action)
        _demo_state["obs"] = new_obs
        rd = new_obs.reward_details
        _demo_state["total_reward"] += rd.total_reward

        history_line = (
            f"Step {len(_demo_state['history'])+1}: "
            f"{strategy}+{model} → reward={rd.total_reward:+.3f} "
            f"({'✅' if rd.correct else '❌'})"
        )
        _demo_state["history"].append(history_line)

        log = "\n".join(_demo_state["history"]) + "\n\n"
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
        return log, breakdown, round(_demo_state["total_reward"], 3)

    with gr.Blocks(title="Eco-LLM Routing Demo") as demo:
        gr.Markdown("""
        # 🌱 Eco-LLM Inference Routing Environment
        **Carbon-aware LLM query routing** — balance accuracy, energy, latency & carbon footprint.
        Choose a routing strategy and model for each query to maximise cumulative reward.
        """)

        with gr.Row():
            task_selector = gr.Dropdown(
                choices=["task_1", "task_2", "task_3"],
                value="task_1",
                label="Task (easy → hard)",
            )
            reset_btn = gr.Button("🔄 Reset Episode", variant="primary")

        obs_box = gr.Textbox(label="Environment State", lines=6, interactive=False)

        gr.Markdown("### Take an Action")
        with gr.Row():
            strategy_dd = gr.Dropdown(
                choices=["NONE", "USE_CACHE", "DO_CASCADE", "EARLY_EXIT", "WAIT", "CALL_KB"],
                value="NONE",
                label="Strategy",
            )
            model_dd = gr.Dropdown(
                choices=["SMALL", "MEDIUM", "LARGE"],
                value="SMALL",
                label="Model",
            )
            exit_cb = gr.Checkbox(label="exit_flag (cascade early stop)", value=False)
        step_btn = gr.Button("▶️ Step", variant="secondary")

        reward_breakdown = gr.Textbox(label="Reward Breakdown", interactive=False)
        total_reward_num = gr.Number(label="Cumulative Reward", interactive=False)

        gr.Markdown("""
        ### Reward Formula
        `reward = score − (carbon_intensity × energy_cost) − (0.1 × latency) + bonuses`

        | Model | Energy | Latency | | Strategy | Effect |
        |---|---|---|---|---|---|
        | SMALL | 0.1 | 1.0s | | USE_CACHE | +0.5 bonus if cached |
        | MEDIUM | 0.3 | 2.0s | | CALL_KB | cheap if kb_available |
        | LARGE | 0.6 | 5.0s | | WAIT | drops carbon next step |
        """)

        reset_btn.click(reset_demo, inputs=[task_selector], outputs=[obs_box, reward_breakdown, total_reward_num])
        step_btn.click(step_demo, inputs=[strategy_dd, model_dd, exit_cb], outputs=[obs_box, reward_breakdown, total_reward_num])

    app = mount_gradio_app(app, demo, path="/ui")

except ImportError:
    # Gradio not installed — API still works fine
    pass


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Run the development server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
