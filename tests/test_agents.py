"""Unit tests for baseline agents."""

from baseline import HeuristicRoutingAgent, RandomRoutingAgent
from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import Strategy


def test_random_agent_creates_valid_actions() -> None:
    """Random agent produces valid RLAction values."""
    agent = RandomRoutingAgent()
    env = EcoLLMInferenceRoutingEnvironment()
    obs = env.reset(task_id="task_1")

    action = agent.get_action(obs)
    assert action.strategy is not None
    assert action.model_choice is not None
    assert isinstance(action.exit_flag, bool)


def test_heuristic_agent_prefers_cache() -> None:
    """Heuristic agent should prioritize cache hits."""
    agent = HeuristicRoutingAgent()
    env = EcoLLMInferenceRoutingEnvironment()
    obs = env.reset(task_id="task_1")

    obs.cache_contents.append(obs.query)
    action = agent.get_action(obs)

    assert action.strategy == Strategy.USE_CACHE


def test_random_agent_deterministic_with_seed() -> None:
    """Random agent should be deterministic with the same seed."""
    agent1 = RandomRoutingAgent(seed=42)
    agent2 = RandomRoutingAgent(seed=42)

    env = EcoLLMInferenceRoutingEnvironment()
    obs = env.reset(task_id="task_1")

    action1 = agent1.get_action(obs)
    action2 = agent2.get_action(obs)

    assert action1.strategy == action2.strategy
    assert action1.model_choice == action2.model_choice
    assert action1.exit_flag == action2.exit_flag
