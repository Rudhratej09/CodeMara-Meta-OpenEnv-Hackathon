"""FastAPI app for the Eco-LLM Inference Routing environment."""

from __future__ import annotations

# CRITICAL: Import create_app from openenv
try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:
    raise ImportError(
        "openenv-core is required. Install dependencies before running the server."
    ) from exc

from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import RLAction, RLObservation

# ✅ CRITICAL: Use create_app() - this creates the /reset, /step, /state endpoints
app = create_app(
    EcoLLMInferenceRoutingEnvironment,
    RLAction,
    RLObservation,
    env_name="eco_llm_inference_routing",
    max_concurrent_envs=4,
)

@app.get("/")
<<<<<<< HEAD
def root():
    return {"status": "ok", "service": "eco_llm_inference_routing"}

@app.get("/health")
def health():
    return {"status": "healthy"}
=======
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")
>>>>>>> 587bffd (fix: redirect root to Gradio /ui)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: Gradio UI (does NOT interfere with API)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import gradio as gr
    from gradio.routes import mount_gradio_app

    def reset_demo(task_id: str, state: dict)
        })

        gr.Markdown("""
        # 🌱 Eco-LLM Inference Routing Environment
        **Carbon-aware LLM query routing** — balance accuracy, energy, latency & carbon footprint.
        """)

        with gr.Row():
            task_selector = gr.Dropdown(
                choices=["task_1", "task_2", "task_3"],
                value="task_1",
                label="Task",
            )
            reset_btn = gr.Button("🔄 Reset Episode", variant="primary")

        obs_box = gr.Textbox(label="Environment State", lines=8, interactive=False)

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
            exit_cb = gr.Checkbox(label="exit_flag", value=False)

        step -> tuple:
        """Reset the environment."""
        try:
            demo_env = EcoLLMInferenceRoutingEnvironment()
            obs = demo_env.reset(task_id=task_id)
            
            updated_state = {
                "episode_id": demo_env.state.episode_id,
                "task_id": task_id,
                "env_state": {
                    "query_index": 0,
                    "cache_contents": [],
                    "carbon_history_index": 0,
                    "step_count": 0,
                },
                "history": [],
                "total_reward": 0.0,
            }
            
            log = f"🔄 Reset — Task: {task_id} ({demo_env.current_task.difficulty})\n"
            log += f"📋 Query: {obs.query}\n"
            log += f"🌍 Carbon intensity: {obs.carbon_intensity:.2f}\n"
            log += f"💾 Cache: {obs.cache_contents or 'empty'}\n"
            log += f"🔌 KB available: {obs.kb_available}"
            
            return log, "", 0.0, updated_state
        except Exception as e:
            return f"❌ Error: {str(e)[:100]}", "", 0.0, state

    def step_demo(strategy: str, model: str, exit_flag: bool, state: dict) -> tuple:
        """Execute one step."""
        try:
            if not state.get("episode_id"):
                return "⚠️ Click Reset first!", "", 0.0, state
            
            # Recreate environment
            demo_env = EcoLLMInferenceRoutingEnvironment()
            demo_env.reset(task_id=state["task_id"])
            
            # Restore state
            env_state = state.get("env_state", {})
            if env_state:
                demo_env._state.query_index = env_state.get("query_index", 0)
                demo_env._state.cache_contents = env_state.get("cache_contents", [])
                demo_env._state.carbon_history_index = env_state.get("carbon_history_index", 0)
                demo_env._state.step_count = env_state.get("step_count", 0)
            
            obs = demo_env._build_observation(
                reward=demo_env._empty_reward(["restored"]),
                done=demo_env._episode_complete
            )
            
            if obs.done:
                return "✅ Episode complete!", "", round(state.get("total_reward", 0.0), 3), state
            
            from server.models import Strategy, ModelChoice
            action = RLAction(
                strategy=Strategy(strategy),
                model_choice=ModelChoice(model),
                exit_flag=exit_flag,
            )
            
            new_obs = demo_env.step(action)
            rd = new_obs.reward_details
            total_reward = state.get("total_reward", 0.0) + rd.total_reward
            
            history = list(state.get("history", []))
            history.append(f"Step {len(history)+1}: {strategy}+{model} → {rd.total_reward:+.3f}")
            
            log = "\n".join(history) + "\n\n"
            if not new_obs.done:
                log += f"📋 Query: {new_obs.query}\n"
                log += f"🌍 Carbon: {new_obs.carbon_intensity:.2f}\n"
                log += f"💾 Cache: {new_obs.cache_contents or 'empty'}"
            else:
                log += "✅ Episode complete!"
            
            breakdown = (
                f"score={rd.score:.1f} | "
                f"energy={rd.energy_cost:.2f} | "
                f"carbon={rd.carbon_penalty:.3f} | "
                f"latency={rd.latency_penalty:.3f} | "
                f"total={rd.total_reward:+.3f}"
            )
            
            updated_state = {
                "episode_id": demo_env.state.episode_id,
                "task_id": state["task_id"],
                "env_state": {
                    "query_index": demo_env._state.query_index,
                    "cache_contents": list(demo_env._state.cache_contents),
                    "carbon_history_index": demo_env._state.carbon_history_index,
                    "step_count": demo_env._state.step_count,
                },
                "history": history,
                "total_reward": round(total_reward, 3),
            }
            
            return log, breakdown, round(total_reward, 3), updated_state
            
        except Exception as e:
            return f"❌ Error: {str(e)[:100]}", "", state.get("total_reward", 0.0), state

    # Create Gradio interface
    with gr.Blocks(title="Eco-LLM Routing Demo") as demo:
        demo_state = gr.State({
            "episode_id": None,
            "task_id": "task_1",
            "env_state": {},
            "history": [],
            "total_reward": 0.0,
        })
        
        gr.Markdown("# 🌱 Eco-LLM Inference Routing")
        
        with gr.Row():
            task_selector = gr.Dropdown(
                ["task_1", "task_2", "task_3"],_btn = gr.Button("▶️ Step", variant="secondary")

        reward_breakdown = gr.Textbox(label="Reward Breakdown", interactive=False)
        total_reward_num = gr.Number(label="Cumulative Reward", interactive=False)

        gr.Markdown("""
        ### Reward Formula
        `reward = score − (carbon_intensity × energy_cost) − (0.1 × latency) + bonuses`
        """)

        reset_btn.click(
            reset_demo,
            inputs=[task_selector],
            outputs=[obs_box, reward_breakdown, total_reward_num, demo_state],
        )
        step_btn.click(
            step_demo,
            inputs=[strategy_dd, model_dd, exit_cb, demo_state],
            outputs=[obs_box, reward_breakdown, total_reward_num, demo_state],
        )

    app = mount_gradio_app(app, demo, path="/ui")

except ImportError:
    print("⚠️ Gradio not installed - API only mode")
    pass


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import
                value="task_1",
                label="Task"
            )
            reset_btn = gr.Button("🔄 Reset", variant="primary")
        
        obs_box = gr.Textbox(label="State", lines=6, interactive=False)
        
        with gr.Row():
            strategy_dd = gr.Dropdown(
                ["NONE", "USE_CACHE", "DO_CASCADE", "EARLY_EXIT", "WAIT", "CALL_KB"],
                value="NONE",
                label="Strategy"
            )
            model_dd = gr.Dropdown(
                ["SMALL", "MEDIUM", "LARGE"],
                value="SMALL",
                label="Model"
            )
            exit_cb = gr.Checkbox(label="Exit Flag", value=False)
        
        step_btn = gr.Button("▶️ Step")
        reward_box = gr.Textbox(label="Reward", interactive=False)
        total_num = gr.Number(label="Total Reward", interactive=False)
        
        reset_btn.click(
            reset_demo,
            inputs=[task_selector, demo_state],
            outputs=[obs_box, reward_box, total_num, demo_state],
        )
        step_btn.click(
            step_demo,
            inputs=[strategy_dd, model_dd, exit_cb, demo_state],
            outputs=[obs_box, reward_box, total_num, demo_state],
        )
    
    # Mount Gradio at /ui (

