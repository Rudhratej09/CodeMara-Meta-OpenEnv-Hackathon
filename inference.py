"""
inference.py - Eco-LLM Inference Routing Environment
====================================================
Spec-compliant inference script for the OpenEnv hackathon.

Environment variables:
    HF_TOKEN       Your HuggingFace API key  (REQUIRED, no default)
    API_BASE_URL   LLM endpoint              (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier          (default: Qwen/Qwen2.5-72B-Instruct)
    ECO_LLM_TASK   Task to run              (default: task_1 | task_2 | task_3 | all)

Stdout format (mandatory):
    [START] task=<task> env=eco_llm_inference_routing model=<model>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

FIXES vs previous version
──────────────────────────
1. score and success initialised to 0.0/False BEFORE the try block so the
   finally clause never raises UnboundLocalError when an exception occurs
   before the score= assignment line.
2. _MAX_TOTAL_REWARD computed inside run_episode (per task), not at module
   load time — eliminates stale-value bug when task_id changes.
3. HF_TOKEN check kept in main() only (not at module load) so the script
   can be imported for testing without a token.
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────

HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
API_KEY: Optional[str] = HF_TOKEN or os.getenv("OPENAI_API_KEY")
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str  = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_ID: str     = os.getenv("ECO_LLM_TASK", "task_1")
BENCHMARK: str   = "eco_llm_inference_routing"

MAX_STEPS: int = 50
SUCCESS_SCORE_THRESHOLD: float = 0.5
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 80

# Max reward per query step = score(1.0) + cache_bonus(0.5) = 1.5
_MAX_REWARD_PER_STEP: float = 1.5
TASK_QUERY_COUNTS: dict[str, int] = {"task_1": 1, "task_2": 3, "task_3": 5}

SYSTEM_PROMPT = textwrap.dedent("""
    You are an LLM inference router for the Eco-LLM environment.

    At each step you receive:
    - The current query
    - Carbon intensity of the grid (0.0 = clean, 1.0 = very dirty)
    - Cache contents (queries already answered correctly — re-use for free)
    - Whether a knowledge base (KB) is available for this query

    Your goal is to maximise total reward:
        reward = score - carbon_penalty - latency_penalty + bonuses
    where:
        score           = 1.0 if model tier >= query minimum, else 0.0
        carbon_penalty  = carbon_intensity * energy_cost (capped at 0.8)
        latency_penalty = 0.1 * latency_cost
        cache_bonus     = +0.5 for a cache hit
        early_exit_bonus= +0.1 for EARLY_EXIT when correct

    Model costs:
        SMALL  — energy 0.1, latency 1.0
        MEDIUM — energy 0.3, latency 2.0
        LARGE  — energy 0.6, latency 5.0

    Strategies:
        NONE       — direct inference with chosen model
        USE_CACHE  — serve from cache (only if query is in cache_contents)
        DO_CASCADE — try models small→chosen, stop on exit_flag if correct
        EARLY_EXIT — direct inference; bonus if correct and exits early
        WAIT       — skip step to wait for lower carbon (useful when carbon > 0.7)
        CALL_KB    — use knowledge base (only when kb_available=true)

    Respond with ONLY valid JSON, no markdown:
    {
      "strategy": "<NONE|USE_CACHE|DO_CASCADE|EARLY_EXIT|WAIT|CALL_KB>",
      "model_choice": "<SMALL|MEDIUM|LARGE>",
      "exit_flag": <true|false>
    }
""").strip()


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """
    [END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM action ───────────────────────────────────────────────────────────────

def build_user_prompt(
    step: int, query: str, carbon_intensity: float,
    cache_contents: List[str], kb_available: bool,
    last_reward: float, history: List[str],
) -> str:
    cache_str  = json.dumps(cache_contents) if cache_contents else "[]"
    hist_block = "\n".join(history[-5:]) if history else "None"
    in_cache   = query in cache_contents
    return textwrap.dedent(f"""
        Step: {step}
        Query: {query!r}
        In cache: {str(in_cache).lower()}
        Cache: {cache_str}
        Carbon intensity: {carbon_intensity:.2f}
        KB available: {str(kb_available).lower()}
        Last reward: {last_reward:.2f}

        Recent history:
        {hist_block}

        Choose the most energy-efficient routing action.
    """).strip()


def get_routing_action(
    client: OpenAI, step: int, query: str, carbon_intensity: float,
    cache_contents: List[str], kb_available: bool,
    last_reward: float, history: List[str],
) -> dict:
    """Call LLM for routing decision; falls back to safe heuristic on any error."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(
                    step, query, carbon_intensity, cache_contents,
                    kb_available, last_reward, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            timeout=10.0,
        )
        raw = (completion.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        strategy     = parsed.get("strategy",     "NONE").upper()
        model_choice = parsed.get("model_choice", "SMALL").upper()
        exit_flag    = bool(parsed.get("exit_flag", False))

        if strategy     not in {"NONE","USE_CACHE","DO_CASCADE","EARLY_EXIT","WAIT","CALL_KB"}:
            strategy = "NONE"
        if model_choice not in {"SMALL", "MEDIUM", "LARGE"}:
            model_choice = "SMALL"

        return {"strategy": strategy, "model_choice": model_choice, "exit_flag": exit_flag}

    except (asyncio.TimeoutError, TimeoutError) as exc:
        print(f"[DEBUG] LLM timeout at step {step}: {exc}", flush=True)
        if query in cache_contents:
            return {"strategy": "USE_CACHE", "model_choice": "SMALL", "exit_flag": False}
        if carbon_intensity > 0.7:
            return {"strategy": "WAIT", "model_choice": "SMALL", "exit_flag": False}
        return {"strategy": "NONE", "model_choice": "SMALL", "exit_flag": False}

    except Exception as exc:
        print(f"[DEBUG] LLM error at step {step}: {exc}", flush=True)
        if query in cache_contents:
            return {"strategy": "USE_CACHE", "model_choice": "SMALL", "exit_flag": False}
        if kb_available:
            return {"strategy": "CALL_KB",   "model_choice": "SMALL", "exit_flag": False}
        if carbon_intensity > 0.7:
            return {"strategy": "WAIT",      "model_choice": "SMALL", "exit_flag": False}
        return {"strategy": "NONE", "model_choice": "SMALL", "exit_flag": True}


def action_to_str(action: dict) -> str:
    return (f"strategy={action['strategy']},"
            f"model={action['model_choice']},"
            f"exit={str(action['exit_flag']).lower()}")


# ── Episode runner ────────────────────────────────────────────────────────────

async def run_episode(client: OpenAI, task_id: str) -> float:
    """
    Run one complete episode.  Emits [START], N×[STEP], [END].
    Returns normalised score in [0.0, 1.0].

    FIX: score and success are initialised here (before the try block)
    so the finally clause can always call log_end() without risk of
    UnboundLocalError even when an exception is raised before score= line.
    """
    # ✅ Initialise BEFORE try so finally never hits UnboundLocalError
    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    success:     bool        = False
    score:       float       = 0.0                        # ← key fix

    # Compute max reward for THIS task (not at module load)
    max_total: float = TASK_QUERY_COUNTS.get(task_id, 5) * _MAX_REWARD_PER_STEP

    # Try AsyncEnvClient (HF Space), fall back to local env
    env = None
    try:
        from openenv.client import AsyncEnvClient  # type: ignore
        env = AsyncEnvClient(base_url=os.getenv("ENV_BASE_URL", "http://localhost:8000"))
    except Exception:
        from server.env import EcoLLMInferenceRoutingEnvironment  # type: ignore
        env = EcoLLMInferenceRoutingEnvironment()

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # reset
        if asyncio.iscoroutinefunction(getattr(env, "reset", None)):
            obs = await env.reset(task_id=task_id)
        else:
            obs = env.reset(task_id=task_id)

        def _field(o: object, *keys: str) -> object:
            for k in keys:
                if isinstance(o, dict):
                    if k in o: return o[k]
                elif hasattr(o, k):
                    return getattr(o, k)
            return None

        last_reward = 0.0
        done = bool(_field(obs, "done") or False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            query   = str(_field(obs, "query") or "")
            carbon  = float(_field(obs, "carbon_intensity") or 0.5)
            cache   = list(_field(obs, "cache_contents") or [])
            kb      = bool(_field(obs, "kb_available") or False)

            action     = get_routing_action(client, step, query, carbon, cache, kb, last_reward, history)
            action_str = action_to_str(action)
            error_msg: Optional[str] = None

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
            except Exception as exc:
                error_msg = str(exc).replace("\n", " ")[:120]
                print(f"[DEBUG] step error: {error_msg}", flush=True)
                done = True
                log_step(step=step, action=action_str, reward=0.0, done=True, error=error_msg)
                break

            reward      = float(_field(obs, "reward") or 0.0)
            done        = bool(_field(obs, "done") or False)
            last_reward = reward

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
            history.append(
                f"step={step} strategy={action['strategy']} "
                f"model={action['model_choice']} reward={reward:+.2f}"
            )
            if done:
                break

        raw_total = sum(rewards)
        score   = float(min(max(raw_total / max_total, 0.0), 1.0)) if max_total > 0 else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            if asyncio.iscoroutinefunction(getattr(env, "close", None)):
                await env.close()
            elif hasattr(env, "close"):
                env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

        # score and success are always defined here (initialised above)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY or not str(API_KEY).strip():
        raise ValueError(
            "HF_TOKEN (or OPENAI_API_KEY) is required.\n"
            "Set: export HF_TOKEN='your_token'"
        )
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # ECO_LLM_TASK=all runs all three tasks sequentially
    tasks = ["task_1", "task_2", "task_3"] if TASK_ID == "all" \
            else [TASK_ID if TASK_ID in TASK_QUERY_COUNTS else "task_1"]

    for t in tasks:
        asyncio.run(run_episode(client, t))


if __name__ == "__main__":
    main()
