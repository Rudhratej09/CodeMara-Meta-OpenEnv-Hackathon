"""Baseline agents for the Eco-LLM Inference Routing environment."""

from __future__ import annotations

import random
from typing import Iterable

from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import ModelChoice, RLAction, Strategy


class RandomPolicyAgent:
    """Baseline agent that samples uniformly from the action space."""

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)

    def act(self) -> RLAction:
        return RLAction(
            strategy=self._rng.choice(list(Strategy)),
            model_choice=self._rng.choice(list(ModelChoice)),
            exit_flag=self._rng.choice([True, False]),
        )


class HeuristicEscalationAgent:
    """Use SMALL by default and escalate after incorrect answers."""

    def __init__(self) -> None:
        self._current_model = ModelChoice.SMALL

    def reset(self) -> None:
        self._current_model = ModelChoice.SMALL

    def act(self, env: EcoLLMInferenceRoutingEnvironment) -> RLAction:
        query = env.current_query
        strategy = Strategy.USE_CACHE if query.text in env.cache_contents else Strategy.NONE
        if env.current_task.task_id == "task_3" and env.current_carbon_intensity > 0.7:
            strategy = Strategy.WAIT
        return RLAction(strategy=strategy, model_choice=self._current_model, exit_flag=True)

    def observe(self, was_correct: bool) -> None:
        if was_correct:
            self._current_model = ModelChoice.SMALL
            return
        if self._current_model == ModelChoice.SMALL:
            self._current_model = ModelChoice.MEDIUM
        elif self._current_model == ModelChoice.MEDIUM:
            self._current_model = ModelChoice.LARGE


def run_episode(
    env: EcoLLMInferenceRoutingEnvironment,
    actions: Iterable[RLAction],
) -> float:
    """Execute a sequence of actions and return cumulative reward."""
    total_reward = 0.0
    env.reset(task_id=env.current_task.task_id)
    for action in actions:
        observation = env.step(action)
        total_reward += float(observation.reward or 0.0)
        if observation.done:
            break
    return total_reward


if __name__ == "__main__":
    env = EcoLLMInferenceRoutingEnvironment()

    random_agent = RandomPolicyAgent()
    env.reset(task_id="task_2")
    for _ in range(6):
        obs = env.step(random_agent.act())
        print("random", obs.reward, obs.done, obs.query)
        if obs.done:
            break

    heuristic_agent = HeuristicEscalationAgent()
    env.reset(task_id="task_3")
    heuristic_agent.reset()
    for _ in range(10):
        action = heuristic_agent.act(env)
        obs = env.step(action)
        heuristic_agent.observe(obs.reward_details.correct)
        print("heuristic", obs.reward, obs.done, obs.query)
        if obs.done:
            break
