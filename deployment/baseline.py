#!/usr/bin/env python3
"""
Baseline evaluation script for the Eco-LLM Inference Routing Environment.

Policies:
  - random:    Random valid strategy + model each step.
  - heuristic: SMALL by default; escalate to MEDIUM/LARGE only if needed;
               use cache when available; wait when carbon > 0.7.
  - llm:       Uses Anthropic Claude via the API to decide strategy + model.

Usage:
    python baseline.py --task easy
    python baseline.py --task medium --policy heuristic
    python baseline.py --task hard   --policy llm --episodes 5
"""

import argparse
import random
import statistics
from typing import Tuple

# ── Inline env import (no server needed for local eval) ──────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.deployment_environment import EcoLLMEnvironment
from models import EcoLLMAction, EcoLLMObservation

STRATEGIES = ["NONE", "USE_CACHE", "DO_CASCADE", "EARLY_EXIT", "WAIT", "CALL_KB"]
MODELS = ["SMALL", "MEDIUM", "LARGE"]


# ── Policies ─────────────────────────────────────────────────────────────────

def random_policy(obs: EcoLLMObservation) -> EcoLLMAction:
    return EcoLLMAction(
        strategy=random.choice(STRATEGIES),
        model_choice=random.choice(MODELS),
    )


def heuristic_policy(obs: EcoLLMObservation) -> EcoLLMAction:
    """Rule-based heuristic."""
    # Wait if carbon is very high and we haven't used all steps
    if obs.carbon_intensity > 0.7 and obs.step < obs.total_steps - 1:
        return EcoLLMAction(strategy="WAIT", model_choice="SMALL")

    # Use cache if query already answered
    if obs.query in obs.cache_contents:
        return EcoLLMAction(strategy="USE_CACHE", model_choice="SMALL")

    # For hard task, use CALL_KB for weather-like queries
    if obs.task == "hard" and "weather" in obs.query.lower():
        return EcoLLMAction(strategy="CALL_KB", model_choice="SMALL")

    # Default: try early-exit (SMALL first, saves energy)
    return EcoLLMAction(strategy="EARLY_EXIT", model_choice="MEDIUM")


def llm_policy(obs: EcoLLMObservation) -> EcoLLMAction:
    """Use Claude claude-sonnet-4-20250514 to decide routing strategy."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (
            f"You are an LLM inference router. Choose the best strategy and model.\n"
            f"Query: {obs.query}\n"
            f"Carbon intensity (0=clean, 1=dirty): {obs.carbon_intensity:.2f}\n"
            f"Cache contains: {obs.cache_contents}\n"
            f"Step {obs.step+1}/{obs.total_steps}, Task: {obs.task}\n\n"
            f"Reply with EXACTLY two lines:\n"
            f"STRATEGY: <one of NONE|USE_CACHE|DO_CASCADE|EARLY_EXIT|WAIT|CALL_KB>\n"
            f"MODEL: <one of SMALL|MEDIUM|LARGE>"
        )
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        lines = {l.split(":")[0].strip(): l.split(":")[1].strip()
                 for l in text.splitlines() if ":" in l}
        strategy = lines.get("STRATEGY", "NONE").upper()
        model = lines.get("MODEL", "SMALL").upper()
        if strategy not in STRATEGIES:
            strategy = "NONE"
        if model not in MODELS:
            model = "SMALL"
        return EcoLLMAction(strategy=strategy, model_choice=model)
    except Exception as e:
        print(f"  [LLM policy error: {e}] falling back to heuristic")
        return heuristic_policy(obs)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task: str, policy_fn) -> Tuple[float, int]:
    env = EcoLLMEnvironment(task=task)
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    while not obs.done:
        action = policy_fn(obs)
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1
    max_possible = len(env._queries) * (1.0 + 0.5)  # max per step ≈ 1.5 (cache bonus)
    normalized = total_reward / max_possible if max_possible > 0 else 0.0
    return normalized, steps


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Eco-LLM Routing Environment Baseline")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--policy", choices=["random", "heuristic", "llm"], default="heuristic")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    policy_map = {
        "random": random_policy,
        "heuristic": heuristic_policy,
        "llm": llm_policy,
    }
    policy_fn = policy_map[args.policy]

    print(f"\n{'='*55}")
    print(f"  Eco-LLM Routing Baseline")
    print(f"  Task: {args.task}  |  Policy: {args.policy}  |  Episodes: {args.episodes}")
    print(f"{'='*55}")

    scores = []
    for ep in range(1, args.episodes + 1):
        score, steps = run_episode(args.task, policy_fn)
        scores.append(score)
        print(f"  Episode {ep:3d}: score={score:.3f}  steps={steps}")

    print(f"\n  Mean score : {statistics.mean(scores):.3f}")
    print(f"  Std dev    : {statistics.stdev(scores):.3f}" if len(scores) > 1 else "")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
