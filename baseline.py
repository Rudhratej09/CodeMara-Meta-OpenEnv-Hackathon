"""Baseline agents for the Eco-LLM Inference Routing environment.

Usage:
    python baseline.py --task task_1 --policy random
    python baseline.py --task task_2 --policy heuristic --episodes 10
    python baseline.py --task task_3 --policy llm --episodes 3
"""

from __future__ import annotations

import argparse
import random
import statistics
from typing import Callable

from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import ModelChoice, RLAction, RLObservation, Strategy
from server.tasks import MAX_STEPS_PER_EPISODE

# ── Policies ─────────────────────────────────────────────────────────────────

class RandomPolicyAgent:
    """Baseline agent that samples uniformly from the action space."""

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, env: EcoLLMInferenceRoutingEnvironment) -> RLAction:
        return RLAction(
            strategy=self._rng.choice(list(Strategy)),
            model_choice=self._rng.choice(list(ModelChoice)),
            exit_flag=self._rng.choice([True, False]),
        )

    def observe(self, was_correct: bool) -> None:
        pass


class HeuristicEscalationAgent:
    """Use SMALL by default; escalate after wrong answers; cache and wait when useful."""

    def __init__(self) -> None:
        self._current_model = ModelChoice.SMALL

    def reset(self) -> None:
        self._current_model = ModelChoice.SMALL

    def act(self, env: EcoLLMInferenceRoutingEnvironment) -> RLAction:
        query = env.current_query

        # Cache hit — free answer
        if query.text in env.cache_contents:
            return RLAction(strategy=Strategy.USE_CACHE, model_choice=self._current_model, exit_flag=False)

        # KB available and query supports it
        if query.kb_available:
            return RLAction(strategy=Strategy.CALL_KB, model_choice=self._current_model, exit_flag=False)

        # Wait when carbon is very high and not on last query
        task_queries = env.current_task.queries
        if (
            env.current_carbon_intensity > 0.7
            and env._state.query_index < len(task_queries) - 1
        ):
            return RLAction(strategy=Strategy.WAIT, model_choice=self._current_model, exit_flag=False)

        return RLAction(strategy=Strategy.NONE, model_choice=self._current_model, exit_flag=True)

    def observe(self, was_correct: bool) -> None:
        if was_correct:
            self._current_model = ModelChoice.SMALL
        elif self._current_model == ModelChoice.SMALL:
            self._current_model = ModelChoice.MEDIUM
        elif self._current_model == ModelChoice.MEDIUM:
            self._current_model = ModelChoice.LARGE


class LLMPolicyAgent:
    """Use Claude to decide strategy and model at each step."""

    def __init__(self) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except ImportError:
            raise ImportError("anthropic package required for LLM policy: pip install anthropic")

    def reset(self) -> None:
        pass

    def act(self, env: EcoLLMInferenceRoutingEnvironment) -> RLAction:
        query = env.current_query
        prompt = (
            f"You are an LLM inference router making a single routing decision.\n\n"
            f"Query: {query.text}\n"
            f"Carbon intensity (0=clean, 1=dirty): {env.current_carbon_intensity:.2f}\n"
            f"Cache contains: {env.cache_contents}\n"
            f"KB available for this query: {query.kb_available}\n"
            f"Task: {env.current_task.task_id} ({env.current_task.difficulty})\n\n"
            f"Choose the most energy-efficient correct strategy.\n"
            f"Reply with EXACTLY:\n"
            f"STRATEGY: <NONE|USE_CACHE|DO_CASCADE|EARLY_EXIT|WAIT|CALL_KB>\n"
            f"MODEL: <SMALL|MEDIUM|LARGE>\n"
            f"EXIT_FLAG: <true|false>"
        )
        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            lines = {}
            for line in text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    lines[k.strip().upper()] = v.strip().upper()

            strategy = lines.get("STRATEGY", "NONE")
            model = lines.get("MODEL", "SMALL")
            exit_flag = lines.get("EXIT_FLAG", "FALSE") == "TRUE"

            if strategy not in Strategy._value2member_map_:
                strategy = "NONE"
            if model not in ModelChoice._value2member_map_:
                model = "SMALL"

            return RLAction(strategy=Strategy(strategy), model_choice=ModelChoice(model), exit_flag=exit_flag)
        except Exception as e:
            print(f"  [LLM policy error: {e}] falling back to heuristic")
            return RLAction(strategy=Strategy.NONE, model_choice=ModelChoice.SMALL, exit_flag=True)

    def observe(self, was_correct: bool) -> None:
        pass


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task_id: str, agent) -> float:
    env = EcoLLMInferenceRoutingEnvironment()
    env.reset(task_id=task_id)
    agent.reset()
    total_reward = 0.0
    for _ in range(MAX_STEPS_PER_EPISODE):   
        action = agent.act(env)
        obs = env.step(action)
        total_reward += float(obs.reward or 0.0)
        agent.observe(obs.reward_details.correct)
        if obs.done:
            break
    return total_reward


# ── CLI ────────────────────────────────────────────────────────────────────────

TASK_MAP = {
    "easy":   "task_1",
    "medium": "task_2",
    "hard":   "task_3",
    "task_1": "task_1",
    "task_2": "task_2",
    "task_3": "task_3",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Eco-LLM Routing Environment Baseline")
    parser.add_argument("--task", default="task_1",
                        choices=["easy", "medium", "hard", "task_1", "task_2", "task_3"])
    parser.add_argument("--policy", default="heuristic",
                        choices=["random", "heuristic", "llm"])
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    task_id = TASK_MAP[args.task]

    agent_map: dict[str, Callable] = {
        "random":    lambda: RandomPolicyAgent(),
        "heuristic": lambda: HeuristicEscalationAgent(),
        "llm":       lambda: LLMPolicyAgent(),
    }
    agent = agent_map[args.policy]()

    print(f"\n{'='*55}")
    print(f"  Eco-LLM Routing Baseline")
    print(f"  Task: {task_id}  |  Policy: {args.policy}  |  Episodes: {args.episodes}")
    print(f"{'='*55}")

    scores = []
    for ep in range(1, args.episodes + 1):
        score = run_episode(task_id, agent)
        scores.append(score)
        print(f"  Episode {ep:3d}: reward={score:.4f}")

    print(f"\n  Mean  : {statistics.mean(scores):.4f}")
    if len(scores) > 1:
        print(f"  Stdev : {statistics.stdev(scores):.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()