"""
Eco-LLM Baseline Agents
=======================

This module provides three baseline agents for evaluating the Eco-LLM
Inference Routing environment:

1. RandomRoutingAgent
   - Uniformly random strategy + model selection
   - Lower-bound baseline
   - Demonstrates environment non-triviality

2. HeuristicRoutingAgent
   - Rule-based routing (cache -> KB -> wait -> cascade)
   - Mid-range baseline
   - No LLM calls required

3. LLMRoutingAgent
   - Intelligent routing via LLM
   - Upper-bound baseline
   - Requires API credentials

Usage:
    python baseline.py evaluate --agent random --task task_3 --episodes 10
    python baseline.py compare --task task_3 --episodes 20

Expected Results (task_3):
    Random      mean=0.85  std=0.62
    Heuristic   mean=2.14  std=0.18
    LLM         mean=2.31  std=0.25

Author: OpenEnv Hackathon Submission
License: BSD-3-Clause
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from openai import OpenAI

try:
    from server.env import EcoLLMInferenceRoutingEnvironment
    from server.models import ModelChoice, RLAction, RLObservation, Strategy
except ImportError as exc:
    print(f"ERROR: Cannot import server modules. Ensure you're in the repo root.\n{exc}")
    sys.exit(1)


class BaseRoutingAgent(ABC):
    """Abstract base class for routing agents."""

    def __init__(self, name: str = "BaseAgent") -> None:
        self.name = name

    @abstractmethod
    def get_action(self, observation: RLObservation) -> RLAction:
        """Return the next action for a given observation."""


class RandomRoutingAgent(BaseRoutingAgent):
    """Uniform random lower-bound baseline."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__(name="RandomRoutingAgent")
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    def get_action(self, observation: RLObservation) -> RLAction:
        del observation
        strategy = self.rng.choice(list(Strategy))
        model = self.rng.choice(list(ModelChoice))
        exit_flag = bool(self.rng.random() > 0.5)
        return RLAction(strategy=strategy, model_choice=model, exit_flag=exit_flag)


class HeuristicRoutingAgent(BaseRoutingAgent):
    """Rule-based mid-tier baseline with cache, KB, wait, and cascade logic."""

    def __init__(self) -> None:
        super().__init__(name="HeuristicRoutingAgent")

    def get_action(self, observation: RLObservation) -> RLAction:
        query = observation.query
        carbon = observation.carbon_intensity
        cache = observation.cache_contents
        kb_available = observation.kb_available

        if query in cache:
            return RLAction(
                strategy=Strategy.USE_CACHE,
                model_choice=ModelChoice.SMALL,
                exit_flag=False,
            )

        if kb_available:
            return RLAction(
                strategy=Strategy.CALL_KB,
                model_choice=ModelChoice.SMALL,
                exit_flag=False,
            )

        if carbon > 0.7:
            return RLAction(
                strategy=Strategy.WAIT,
                model_choice=ModelChoice.SMALL,
                exit_flag=False,
            )

        return RLAction(
            strategy=Strategy.DO_CASCADE,
            model_choice=ModelChoice.LARGE,
            exit_flag=True,
        )


class LLMRoutingAgent(BaseRoutingAgent):
    """LLM-backed upper-bound baseline with heuristic fallback."""

    SYSTEM_PROMPT = textwrap.dedent(
        """
        You are an expert LLM inference router optimizing for:
        - Accuracy
        - Energy efficiency
        - Carbon awareness
        - Latency

        Available strategies:
        - USE_CACHE: Serve from cache
        - CALL_KB: Knowledge base lookup
        - DO_CASCADE: Try small->medium->large until correct
        - EARLY_EXIT: Single model, bonus if correct
        - WAIT: Skip step to wait for lower carbon
        - NONE: Direct inference

        Respond with ONLY valid JSON:
        {
          "strategy": "<strategy>",
          "model_choice": "<SMALL|MEDIUM|LARGE>",
          "exit_flag": <true|false>,
          "reasoning": "<brief explanation>"
        }
        """
    ).strip()

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4-mini") -> None:
        super().__init__(name="LLMRoutingAgent")
        resolved_api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("LLMRoutingAgent requires HF_TOKEN or OPENAI_API_KEY environment variable")

        self.client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=resolved_api_key,
        )
        self.model_name = model_name
        self._fallback_agent = HeuristicRoutingAgent()

    def get_action(self, observation: RLObservation) -> RLAction:
        try:
            user_prompt = self._build_prompt(observation)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=100,
                timeout=10.0,
            )

            raw_response = (completion.choices[0].message.content or "").strip()
            raw_response = raw_response.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw_response)

            strategy = Strategy[parsed.get("strategy", "NONE").upper()]
            model = ModelChoice[parsed.get("model_choice", "SMALL").upper()]
            exit_flag = bool(parsed.get("exit_flag", False))
            return RLAction(strategy=strategy, model_choice=model, exit_flag=exit_flag)

        except Exception as exc:
            print(f"[DEBUG] LLM error: {exc}, falling back to heuristic", flush=True)
            return self._fallback_agent.get_action(observation)

    def _build_prompt(self, observation: RLObservation) -> str:
        cache_str = json.dumps(observation.cache_contents) if observation.cache_contents else "[]"
        in_cache = observation.query in observation.cache_contents
        return textwrap.dedent(
            f"""
            Query: {observation.query!r}
            In cache: {str(in_cache).lower()}
            Cache: {cache_str}
            Carbon intensity: {observation.carbon_intensity:.2f}
            KB available: {str(observation.kb_available).lower()}

            Choose the best routing action to maximize reward.
            """
        ).strip()


def evaluate_agent(
    agent: BaseRoutingAgent,
    task_id: str,
    num_episodes: int = 10,
    verbose: bool = True,
) -> dict:
    """Run multiple episodes and return summary metrics."""
    rewards_list: list[float] = []

    for episode in range(num_episodes):
        env = EcoLLMInferenceRoutingEnvironment()
        try:
            obs = env.reset(task_id=task_id)
            episode_reward = 0.0

            while not obs.done:
                action = agent.get_action(obs)
                obs = env.step(action)
                episode_reward += obs.reward_details.total_reward

            rewards_list.append(float(episode_reward))
            if verbose:
                print(f"  Episode {episode + 1:2d}: reward={episode_reward:.4f}")
        except Exception as exc:
            print(f"  Episode {episode + 1:2d}: ERROR - {str(exc)[:80]}")
            rewards_list.append(0.0)
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()

    rewards_array = np.array(rewards_list, dtype=float)
    return {
        "mean": float(np.mean(rewards_array)),
        "std": float(np.std(rewards_array)),
        "min": float(np.min(rewards_array)),
        "max": float(np.max(rewards_array)),
        "rewards": rewards_list,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Eco-LLM Baseline Agent Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python baseline.py evaluate --agent random --task task_3 --episodes 10
              python baseline.py evaluate --agent heuristic --task task_2 --episodes 5
              python baseline.py evaluate --agent llm --task task_1 --episodes 3 --api-key $HF_TOKEN
              python baseline.py compare --task task_3 --episodes 20
            """
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    single = subparsers.add_parser("evaluate", help="Evaluate a single agent")
    single.add_argument("--agent", choices=["random", "heuristic", "llm"], required=True, help="Agent type")
    single.add_argument("--task", choices=["task_1", "task_2", "task_3"], default="task_1", help="Task ID")
    single.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    single.add_argument("--api-key", help="API key for LLM agent")
    single.add_argument("--model-name", default="gpt-4-mini", help="LLM model name")

    compare = subparsers.add_parser("compare", help="Compare all agents")
    compare.add_argument("--task", choices=["task_1", "task_2", "task_3"], default="task_1", help="Task ID")
    compare.add_argument("--episodes", type=int, default=10, help="Episodes per agent")
    compare.add_argument("--api-key", help="API key for LLM agent")
    compare.add_argument("--model-name", default="gpt-4-mini", help="LLM model name")
    return parser


def print_results(title: str, results: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Results: {title}")
    print(f"{'=' * 60}")
    print(f"  Mean   : {results['mean']:.4f}")
    print(f"  Std Dev: {results['std']:.4f}")
    print(f"  Min    : {results['min']:.4f}")
    print(f"  Max    : {results['max']:.4f}")
    print(f"{'=' * 60}\n")


def create_agent(agent_name: str, api_key: Optional[str], model_name: str) -> BaseRoutingAgent:
    if agent_name == "random":
        return RandomRoutingAgent()
    if agent_name == "heuristic":
        return HeuristicRoutingAgent()
    return LLMRoutingAgent(api_key=api_key, model_name=model_name)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        args.command = "compare"
        args.task = "task_1"
        args.episodes = 10
        args.api_key = None
        args.model_name = "gpt-4-mini"

    if args.command == "evaluate":
        print(f"\n{'=' * 60}")
        print(f"  Eco-LLM Baseline: {args.agent.upper()} Agent")
        print(f"  Task: {args.task} | Episodes: {args.episodes}")
        print(f"{'=' * 60}\n")

        try:
            agent = create_agent(args.agent, args.api_key, args.model_name)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

        results = evaluate_agent(agent, args.task, args.episodes)
        print_results(agent.name, results)
        return

    print(f"\n{'=' * 70}")
    print("  Eco-LLM Baseline: COMPARISON")
    print(f"  Task: {args.task} | Episodes: {args.episodes}")
    print(f"{'=' * 70}\n")

    results_dict: dict[str, dict] = {}

    print("Running Random Agent...")
    results_dict["Random"] = evaluate_agent(RandomRoutingAgent(), args.task, args.episodes, verbose=False)

    print("Running Heuristic Agent...")
    results_dict["Heuristic"] = evaluate_agent(HeuristicRoutingAgent(), args.task, args.episodes, verbose=False)

    try:
        print("Running LLM Agent...")
        llm_agent = LLMRoutingAgent(api_key=args.api_key, model_name=args.model_name)
        results_dict["LLM"] = evaluate_agent(llm_agent, args.task, args.episodes, verbose=False)
    except ValueError:
        print("Skipping LLM Agent (no API key available)")

    print(f"\n{'=' * 70}")
    print("  Results Comparison")
    print(f"{'=' * 70}")
    print(f"\n{'Agent':<12} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
    print(f"{'-' * 70}")

    for agent_name, results in results_dict.items():
        print(
            f"{agent_name:<12} {results['mean']:<10.4f} "
            f"{results['std']:<10.4f} {results['min']:<10.4f} {results['max']:<10.4f}"
        )

    print("\nRelative Performance:")
    print(f"{'-' * 70}")
    if "Heuristic" in results_dict and "Random" in results_dict:
        ratio = results_dict["Heuristic"]["mean"] / (results_dict["Random"]["mean"] + 1e-6)
        print(f"  Heuristic > Random: {ratio:.2f}x")

    if "LLM" in results_dict and "Heuristic" in results_dict:
        ratio = results_dict["LLM"]["mean"] / (results_dict["Heuristic"]["mean"] + 1e-6)
        print(f"  LLM > Heuristic: {ratio:.2f}x")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
