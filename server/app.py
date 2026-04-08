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
<<<<<<< HEAD
from server.models import RLAction, RLObservation
=======
from server.models import ModelChoice, RLAction, RLObservation, Strategy
>>>>>>> cb2c156 (Fix OpenEnv app startup and serializable Gradio state)

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

=======
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "eco_llm_inference_routing",
        "ui": "/ui",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


>>>>>>> cb2c156 (Fix OpenEnv app startup and serializable Gradio state)
try:
    import gradio as gr
    from gradio.routes import mount_gradio_app

<<<<<<< HEAD
    def reset_demo(task_id: str, state: dict)
        })

        gr.Markdown("""
        # 🌱 Eco-LLM Inference Routing Environment
        **Carbon-aware LLM query routing** — balance accuracy, energy, latency & carbon footprint.
        """)
=======
    def _serialize_env_state(env: EcoLLMInferenceRoutingEnvironment) -> dict[str, object]:
        state = env.state
        return {
            "task_id": state.task_id,
            "step_count": state.step_count,
            "step_index": state.step_index,
            "query_index": state.query_index,
            "total_energy": float(state.total_energy),
            "total_latency": float(state.total_latency),
            "cache_contents": list(state.cache_contents),
            "carbon_history_index": state.carbon_history_index,
        }

    def _restore_env_state(
        env: EcoLLMInferenceRoutingEnvironment,
        serialized_state: dict[str, object],
    ) -> None:
        env._state.task_id = str(serialized_state["task_id"])
        env._state.step_count = int(serialized_state["step_count"])
        env._state.step_index = int(serialized_state["step_index"])
        env._state.query_index = int(serialized_state["query_index"])
        env._state.total_energy = float(serialized_state["total_energy"])
        env._state.total_latency = float(serialized_state["total_latency"])
        env._state.cache_contents = list(serialized_state["cache_contents"])
        env._state.carbon_history_index = int(serialized_state["carbon_history_index"])

    def reset_demo(task_id: str) -> tuple[str, str, float, dict[str, object]]:
        demo_env = EcoLLMInferenceRoutingEnvironment()
        obs = demo_env.reset(task_id=task_id)
        updated_state = {
            "task_id": task_id,
            "env_state": _serialize_env_state(demo_env),
            "history": [],
            "total_reward": 0.0,
            "episode_active": True,
        }
        log = (
            f"Reset task: {task_id}\n"
            f"Difficulty: {demo_env.current_task.difficulty}\n"
            f"Query: {obs.query}\n"
            f"Carbon intensity: {obs.carbon_intensity:.2f}\n"
            f"Cache: {obs.cache_contents or 'empty'}\n"
            f"KB available: {obs.kb_available}"
        )
        return log, "", 0.0, updated_state

    def step_demo(
        strategy: str,
        model: str,
        exit_flag: bool,
        state: dict[str, object],
    ) -> tuple[str, str, float, dict[str, object]]:
        if not state.get("episode_active") or not state.get("env_state"):
            return "Click Reset first.", "", 0.0, state

        demo_env = EcoLLMInferenceRoutingEnvironment()
        demo_env.reset(task_id=str(state["task_id"]))
        _restore_env_state(demo_env, dict(state["env_state"]))

        obs = demo_env._build_observation(
            reward=demo_env._empty_reward(["restored"]),
            done=demo_env._episode_complete,
        )
        if obs.done:
            return (
                "Episode complete. Click Reset to start again.",
                "",
                round(float(state.get("total_reward", 0.0)), 3),
                state,
            )

        action = RLAction(
            strategy=Strategy(strategy),
            model_choice=ModelChoice(model),
            exit_flag=exit_flag,
        )
        new_obs = demo_env.step(action)
        reward = new_obs.reward_details
        total_reward = float(state.get("total_reward", 0.0)) + reward.total_reward

        history = list(state.get("history", []))
        history.append(
            f"Step {len(history) + 1}: {strategy}+{model} reward={reward.total_reward:+.3f}"
        )

        log = "\n".join(history) + "\n\n"
        if new_obs.done:
            log += "Episode complete."
        else:
            log += (
                f"Query: {new_obs.query}\n"
                f"Carbon intensity: {new_obs.carbon_intensity:.2f}\n"
                f"Cache: {new_obs.cache_contents or 'empty'}\n"
                f"KB available: {new_obs.kb_available}"
            )

        breakdown = (
            f"score={reward.score:.1f}  "
            f"energy={reward.energy_cost:.2f}  "
            f"carbon={reward.carbon_penalty:.3f}  "
            f"latency={reward.latency_penalty:.3f}  "
            f"bonuses={reward.bonuses:.2f}  "
            f"total={reward.total_reward:+.3f}"
        )
        updated_state = {
            "task_id": str(state["task_id"]),
            "env_state": _serialize_env_state(demo_env),
            "history": history,
            "total_reward": round(total_reward, 3),
            "episode_active": not new_obs.done,
        }
        return log, breakdown, round(total_reward, 3), updated_state

    with gr.Blocks(title="Eco-LLM Routing Demo") as demo:
        demo_state = gr.State(
            {
                "task_id": "task_1",
                "env_state": None,
                "history": [],
                "total_reward": 0.0,
                "episode_active": False,
            }
        )
        gr.Markdown(
            """
            # Eco-LLM Inference Routing
            API-first OpenEnv app with an optional local demo.
            """
        )
>>>>>>> cb2c156 (Fix OpenEnv app startup and serializable Gradio state)

        with gr.Row():
            task_selector = gr.Dropdown(
                choices=["task_1", "task_2", "task_3"],
                value="task_1",
                label="Task",
            )
            reset_btn = gr.Button("Reset", variant="primary")

<<<<<<< HEAD
        obs_box = gr.Textbox(label="Environment State", lines=8, interactive=False)
=======
        obs_box = gr.Textbox(label="State", lines=8, interactive=False)
>>>>>>> cb2c156 (Fix OpenEnv app startup and serializable Gradio state)

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
<<<<<<< HEAD
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
=======
            exit_cb = gr.Checkbox(label="Exit flag", value=False)

        step_btn = gr.Button("Step")
        reward_box = gr.Textbox(label="Reward", interactive=False)
        total_num = gr.Number(label="Total reward", interactive=False)
>>>>>>> cb2c156 (Fix OpenEnv app startup and serializable Gradio state)

        reset_btn.click(
            reset_demo,
            inputs=[task_selector],
<<<<<<< HEAD
            outputs=[obs_box, reward_breakdown, total_reward_num, demo_state],
=======
            outputs=[obs_box, reward_box, total_num, demo_state],
>>>>>>> cb2c156 (Fix OpenEnv app startup and serializable Gradio state)
        )
        step_btn.click(
            step_demo,
            inputs=[strategy_dd, model_dd, exit_cb, demo_state],
            outputs=[obs_box, reward_box, total_num, demo_state],
        )

    app = mount_gradio_app(app, demo, path="/ui")
<<<<<<< HEAD

except ImportError:
    print("⚠️ Gradio not installed - API only mode")
=======
except ImportError:  # pragma: no cover
>>>>>>> cb2c156 (Fix OpenEnv app startup and serializable Gradio state)
    pass


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Run the server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
<<<<<<< HEAD
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
=======
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
>>>>>>> cb2c156 (Fix OpenEnv app startup and serializable Gradio state)
