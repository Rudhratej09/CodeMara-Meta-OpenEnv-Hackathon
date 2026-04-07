"""
inference.py — Eco-LLM Inference Routing Environment
=====================================================
Spec-compliant inference script for the OpenEnv hackathon.

Environment variables (required):
    HF_TOKEN / API_KEY   Your Hugging Face or API key
    API_BASE_URL         LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME           Model ID       (default: Qwen/Qwen2.5-72B-Instruct)
    ECO_LLM_TASK         Task to run    (default: task_1  |  task_1 / task_2 / task_3)
    LOCAL_IMAGE_NAME     Docker image name (if using from_docker_image)

Stdout format (mandatory):
    [START] task=<task> env=eco_llm_inference_routing model=<model>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── env / LLM config ──────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME", "")

TASK_ID      = os.getenv("ECO_LLM_TASK", "task_1")   # task_1 / task_2 / task_3
BENCHMARK    = "eco_llm_inference_routing"

# ── episode config ────────────────────────────────────────────────────────────
MAX_STEPS               = 50          # matches MAX_STEPS_PER_EPISODE in tasks.py
SUCCESS_SCORE_THRESHOLD = 0.5         # normalised score in [0,1]
TEMPERATURE             = 0.2         # low for deterministic routing decisions
MAX_TOKENS              = 80

# ── reward normalisation ──────────────────────────────────────────────────────
# Best-case per step: score=1.0, cache bonus=0.5, no penalties → 1.5
# Realistic ceiling for normalisation (conservative)
_MAX_REWARD_PER_STEP = 1.5
TASK_QUERY_COUNTS    = {"task_1": 1, "task_2": 3, "task_3": 5}
_MAX_TOTAL_REWARD    = TASK_QUERY_COUNTS.get(TASK_ID, 5) * _MAX_REWARD_PER_STEP

# ── prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an LLM inference router for the Eco-LLM environment.

    At each step you receive:
    - The current query
    - Carbon intensity of the grid (0.0 = clean, 1.0 = very dirty)
    - Cache contents (queries already answered correctly — re-use them for free)
    - Whether a knowledge base (KB) is available for this query

    Your goal is to maximise total reward:
        reward = score − carbon_penalty − latency_penalty + bonuses
    where:
        score          = 1.0 if model tier >= query minimum, else 0.0
        carbon_penalty = carbon_intensity × energy_cost   (capped at 0.8)
        latency_penalty= 0.1 × latency_cost
        cache_bonus    = +0.5 for a cache hit
        early_exit_bonus = +0.1 for EARLY_EXIT when correct

    Model costs:
        SMALL  — energy 0.1, latency 1.0  (answers queries needing SMALL)
        MEDIUM — energy 0.3, latency 2.0  (answers queries needing SMALL or MEDIUM)
        LARGE  — energy 0.6, latency 5.0  (answers any query, but expensive)

    Strategies:
        NONE       — direct inference with chosen model
        USE_CACHE  — serve from cache (only if query is in cache_contents)
        DO_CASCADE — try models small→chosen, stop on exit_flag if correct
        EARLY_EXIT — direct inference; bonus if correct and exits early
        WAIT       — skip step to wait for lower carbon (useful when carbon > 0.7)
        CALL_KB    — use knowledge base (only when kb_available=true)

    Rules:
    - Prefer SMALL models unless the query requires more capability.
    - Use USE_CACHE immediately if the query is already in cache_contents.
    - Use CALL_KB only when kb_available is true.
    - WAIT only on non-final queries when carbon_intensity > 0.7.
    - DO_CASCADE with exit_flag=true saves energy by stopping at the first correct answer.

    Respond with ONLY valid JSON, no markdown, no extra text:
    {
      "strategy": "<NONE|USE_CACHE|DO_CASCADE|EARLY_EXIT|WAIT|CALL_KB>",
      "model_choice": "<SMALL|MEDIUM|LARGE>",
      "exit_flag": <true|false>,
      "reasoning": "<one short sentence>"
    }
""").strip()


def build_user_prompt(
    step: int,
    query: str,
    carbon_intensity: float,
    cache_contents: List[str],
    kb_available: bool,
    last_reward: float,
    history: List[str],
) -> str:
    cache_str  = json.dumps(cache_contents) if cache_contents else "[]"
    hist_block = "\n".join(history[-5:]) if history else "None"
    in_cache   = query in cache_contents
    return textwrap.dedent(f"""
        Step: {step}
        Query: {query!r}
        In cache: {str(in_cache).lower()}
        Cache contents: {cache_str}
        Carbon intensity: {carbon_intensity:.2f}
        KB available: {str(kb_available).lower()}
        Last step reward: {last_reward:.2f}

        Recent history:
        {hist_block}

        Choose the most energy-efficient correct routing action.
    """).strip()


# ── stdout helpers ─────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM decision ──────────────────────────────────────────────────────────────

def get_routing_action(
    client: OpenAI,
    step: int,
    query: str,
    carbon_intensity: float,
    cache_contents: List[str],
    kb_available: bool,
    last_reward: float,
    history: List[str],
) -> dict:
    """Call the LLM and parse a routing action. Falls back to a safe heuristic."""
    user_prompt = build_user_prompt(
        step, query, carbon_intensity, cache_contents, kb_available, last_reward, history
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        strategy     = parsed.get("strategy",    "NONE").upper()
        model_choice = parsed.get("model_choice", "SMALL").upper()
        exit_flag    = bool(parsed.get("exit_flag", False))

        valid_strategies = {"NONE", "USE_CACHE", "DO_CASCADE", "EARLY_EXIT", "WAIT", "CALL_KB"}
        valid_models     = {"SMALL", "MEDIUM", "LARGE"}
        if strategy not in valid_strategies:
            strategy = "NONE"
        if model_choice not in valid_models:
            model_choice = "SMALL"

        return {"strategy": strategy, "model_choice": model_choice, "exit_flag": exit_flag}

    except Exception as exc:
        print(f"[DEBUG] LLM routing error at step {step}: {exc}", flush=True)
        # Safe heuristic fallback
        if query in cache_contents:
            return {"strategy": "USE_CACHE", "model_choice": "SMALL", "exit_flag": False}
        if kb_available:
            return {"strategy": "CALL_KB",   "model_choice": "SMALL", "exit_flag": False}
        if carbon_intensity > 0.7:
            return {"strategy": "WAIT",      "model_choice": "SMALL", "exit_flag": False}
        return {"strategy": "NONE", "model_choice": "SMALL", "exit_flag": True}


def action_to_str(action: dict) -> str:
    """Compact single-token representation for [STEP] line."""
    s   = action["strategy"]
    m   = action["model_choice"]
    ex  = str(action["exit_flag"]).lower()
    return f"strategy={s},model={m},exit={ex}"


# ── main episode loop ─────────────────────────────────────────────────────────

async def run_episode(client: OpenAI) -> None:
    """
    Connect to the Eco-LLM environment (docker or direct) and run one episode,
    emitting mandatory [START] / [STEP] / [END] lines throughout.
    """
    # ── environment setup ──────────────────────────────────────────────────
    # Try docker image first (HF Spaces / CI). Fall back to direct import.
    env = None
    try:
        from openenv.client import AsyncEnvClient  # type: ignore
        env = AsyncEnvClient(base_url=os.getenv("ENV_BASE_URL", "http://localhost:8000"))
    except Exception:
        # Direct import for local dev / evaluation without a running server
        from server.env import EcoLLMInferenceRoutingEnvironment  # type: ignore
        env = EcoLLMInferenceRoutingEnvironment()

    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── reset ──────────────────────────────────────────────────────────
        if asyncio.iscoroutinefunction(getattr(env, "reset", None)):
            obs = await env.reset(task_id=TASK_ID)
        else:
            obs = env.reset(task_id=TASK_ID)

        # Normalise observation access (client dict vs dataclass)
        def _obs_field(o, *keys):
            for k in keys:
                if isinstance(o, dict):
                    if k in o: return o[k]
                else:
                    if hasattr(o, k): return getattr(o, k)
            return None

        last_reward = 0.0
        done        = bool(_obs_field(obs, "done") or False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            query            = _obs_field(obs, "query") or ""
            carbon_intensity = float(_obs_field(obs, "carbon_intensity") or 0.5)
            cache_contents   = list(_obs_field(obs, "cache_contents") or [])

            # kb_available is now a first-class field on the observation
            kb_available = bool(_obs_field(obs, "kb_available") or False)

            action = get_routing_action(
                client, step, query, carbon_intensity,
                cache_contents, kb_available, last_reward, history,
            )
            action_str = action_to_str(action)

            # ── build RLAction and step ────────────────────────────────────
            error_msg = None
            try:
                from server.models import ModelChoice, RLAction, Strategy  # type: ignore
                rl_action = RLAction(
                    strategy=Strategy(action["strategy"]),
                    model_choice=ModelChoice(action["model_choice"]),
                    exit_flag=action["exit_flag"],
                )
                if asyncio.iscoroutinefunction(getattr(env, "step", None)):
                    obs = await env.step(rl_action)
                else:
                    obs = env.step(rl_action)
            except Exception as e:
                error_msg = str(e).replace("\n", " ")[:120]
                print(f"[DEBUG] step error: {error_msg}", flush=True)
                done = True
                log_step(step=step, action=action_str, reward=0.0, done=True, error=error_msg)
                break

            reward      = float(_obs_field(obs, "reward") or 0.0)
            done        = bool(_obs_field(obs, "done")   or False)
            last_reward = reward

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            history.append(
                f"step={step} strategy={action['strategy']} model={action['model_choice']} "
                f"reward={reward:+.2f} correct={getattr(getattr(obs, 'reward_details', None), 'correct', '?')}"
            )

            if done:
                break

        # ── score normalisation ────────────────────────────────────────────
        raw_total = sum(rewards)
        max_total = _MAX_TOTAL_REWARD if _MAX_TOTAL_REWARD > 0 else 1.0
        score     = min(max(raw_total / max_total, 0.0), 1.0)
        success   = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # Always close — mandatory per spec
        try:
            if asyncio.iscoroutinefunction(getattr(env, "close", None)):
                await env.close()
            elif hasattr(env, "close"):
                env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    asyncio.run(run_episode(client))


if __name__ == "__main__":
    main()
