"""
baseline.py - Eco-LLM Baseline Agents
=====================================
Three independent baseline agents for evaluation and comparison.

Usage:
    python baseline.py evaluate --agent random --task task_1 --episodes 10
    python baseline.py evaluate --agent heuristic --task task_3 --episodes 10
    python baseline.py compare --task task_3 --episodes 20

Expected Results (task_3):
    Random agent:     ~0.85 +/- 0.62
    Heuristic agent:  ~2.14 +/- 0.18
"""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod

import numpy as np

try:
    from server.env import EcoLLMInferenceRoutingEnvironment
    from server.models import ModelChoice, RLAction, RLObservation, Strategy
except ImportError as exc:
    print(f"ERROR: Cannot import server modules: {exc}")
    print("Make sure you're in the repo root directory.")
    sys.exit(1)


class BaseRoutingAgent(ABC):
    """Abstract base class for routing agents."""

    def __init__(self, name: str = "BaseAgent") -> None:
        self.name = name

    @abstractmethod
    def get_action(self, observation: RLObservation) -> RLAction:
        """Given an observation, return the next action."""


class RandomRoutingAgent(BaseRoutingAgent):
    """
    Random baseline agent.

    Policy: Uniformly random strategy + model selection.
    Purpose: Lower-bound baseline (no intelligence).
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(name="RandomRoutingAgent")
        self.rng = np.random.RandomState(seed)

    def get_action(self, observation: RLObservation) -> RLAction:
        """Select a random strategy and model."""
        del observation
        strategies = list(Strategy)
        models = list(ModelChoice)
        strategy = strategies[int(self.rng.randint(len(strategies)))]
        model_choice = models[int(self.rng.randint(len(models)))]
        exit_flag = bool(self.rng.random() > 0.5)
        return RLAction(strategy=strategy, model_choice=model_choice, exit_flag=exit_flag)


class HeuristicRoutingAgent(BaseRoutingAgent):
    """
    Heuristic baseline agent.

    Policy: Rule-based routing with priorities.
    Priority 1: Use cache
    Priority 2: Wait if carbon > 0.7
    Priority 3: Cascade with early exit
    """

    def __init__(self) -> None:
        super().__init__(name="HeuristicRoutingAgent")

    def get_action(self, observation: RLObservation) -> RLAction:
        """Apply heuristic routing policy."""
        query = observation.query
        carbon = observation.carbon_intensity
        cache = observation.cache_contents

        if query in cache:
            return RLAction(
                strategy=Strategy.USE_CACHE,
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


def evaluate_agent(
    agent: BaseRoutingAgent,
    task_id: str,
    num_episodes: int = 10,
    verbose: bool = True,
) -> dict:
    """Run an agent on a task for multiple episodes."""
    env = EcoLLMInferenceRoutingEnvironment()
    rewards_list: list[float] = []

    for episode in range(num_episodes):
        try:
            obs = env.reset(task_id=task_id)
            episode_reward = 0.0

            while not obs.done:
                action = agent.get_action(obs)
                obs = env.step(action)
                episode_reward += obs.reward_details.total_reward

            rewards_list.append(episode_reward)
            if verbose:
                print(f"  Episode {episode + 1:2d}/{num_episodes}: reward={episode_reward:.4f}", flush=True)

        except Exception as exc:
            print(f"  Episode {episode + 1:2d}/{num_episodes}: ERROR - {str(exc)[:80]}", flush=True)
            rewards_list.append(0.0)

    rewards_array = np.array(rewards_list)
    return {
        "mean": float(np.mean(rewards_array)),
        "std": float(np.std(rewards_array)),
        "min": float(np.min(rewards_array)),
        "max": float(np.max(rewards_array)),
        "rewards": rewards_list,
    }


def main() -> None:
    """Command-line interface for baseline evaluation."""
    parser = argparse.ArgumentParser(
        description="Eco-LLM Baseline Agent Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python baseline.py evaluate --agent random --task task_1 --episodes 10
  python baseline.py evaluate --agent heuristic --task task_3 --episodes 10
  python baseline.py compare --task task_3 --episodes 20
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a single agent")
    evaluate_parser.add_argument(
        "--agent",
        choices=["random", "heuristic"],
        default="heuristic",
        help="Agent type",
    )
    evaluate_parser.add_argument(
        "--task",
        choices=["task_1", "task_2", "task_3"],
        default="task_1",
        help="Task to evaluate on",
    )
    evaluate_parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare all agents")
    compare_parser.add_argument(
        "--task",
        choices=["task_1", "task_2", "task_3"],
        default="task_1",
        help="Task to evaluate on",
    )
    compare_parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per agent",
    )

    args = parser.parse_args()

    if args.command == "evaluate":
        print(f"\n{'=' * 70}")
        print("  Eco-LLM Baseline Evaluation")
        print(f"  Agent: {args.agent.upper()}")
        print(f"  Task: {args.task} | Episodes: {args.episodes}")
        print(f"{'=' * 70}\n")

        if args.agent == "random":
            agent = RandomRoutingAgent()
        else:
            agent = HeuristicRoutingAgent()

        results = evaluate_agent(agent, args.task, args.episodes)

        print(f"\n{'=' * 70}")
        print(f"  Results: {agent.name}")
        print(f"{'=' * 70}")
        print(f"  Mean   : {results['mean']:.4f}")
        print(f"  Std Dev: {results['std']:.4f}")
        print(f"  Min    : {results['min']:.4f}")
        print(f"  Max    : {results['max']:.4f}")
        print(f"{'=' * 70}\n")

    elif args.command == "compare":
        print(f"\n{'=' * 70}")
        print("  Eco-LLM Baseline Comparison")
        print(f"  Task: {args.task} | Episodes: {args.episodes}")
        print(f"{'=' * 70}\n")

        results_dict: dict[str, dict] = {}

        print("Running Random Agent...")
        agent_random = RandomRoutingAgent()
        results_dict["Random"] = evaluate_agent(agent_random, args.task, args.episodes, verbose=False)

        print("\nRunning Heuristic Agent...")
        agent_heuristic = HeuristicRoutingAgent()
        results_dict["Heuristic"] = evaluate_agent(
            agent_heuristic,
            args.task,
            args.episodes,
            verbose=False,
        )

        print(f"\n{'=' * 70}")
        print("  Results Comparison")
        print(f"{'=' * 70}\n")
        print(f"{'Agent':<15} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
        print(f"{'-' * 70}")

        for agent_name, results in results_dict.items():
            print(
                f"{agent_name:<15} {results['mean']:<12.4f} "
                f"{results['std']:<12.4f} {results['min']:<12.4f} {results['max']:<12.4f}"
            )

        print(f"\n{'Relative Performance':<15}")
        print(f"{'-' * 70}")
        if "Heuristic" in results_dict and "Random" in results_dict:
            ratio = results_dict["Heuristic"]["mean"] / (results_dict["Random"]["mean"] + 1e-6)
            print(f"  Heuristic > Random: {ratio:.2f}x")

        print(f"{'=' * 70}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
