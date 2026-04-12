"""Compatibility graders for OpenEnv hackathon validators."""

from __future__ import annotations

from typing import Any

_MAX_REWARD_PER_QUERY: float = 1.5

_MAX_REWARDS: dict[str, float] = {
    "task_1": 1.0 * _MAX_REWARD_PER_QUERY,
    "task_2": 3.0 * _MAX_REWARD_PER_QUERY,
    "task_3": 5.0 * _MAX_REWARD_PER_QUERY,
}

_SCORE_MIN: float = 0.01
_SCORE_MAX: float = 0.99


def _compute_score(task_id: str, rewards: list[float]) -> float:
    max_reward = _MAX_REWARDS.get(task_id, _MAX_REWARDS["task_1"])
    total_reward = sum(rewards)
    raw_score = total_reward / max_reward if max_reward > 0 else 0.0
    return float(max(_SCORE_MIN, min(_SCORE_MAX, raw_score)))


def _extract_rewards(episode_data: dict[str, Any]) -> list[float]:
    rewards = episode_data.get("rewards")
    if rewards is None:
        rewards = episode_data.get("per_step_rewards")
    if rewards is None:
        rewards = []
    return [float(reward) for reward in rewards]


def _make_result(task_id: str, score: float, rewards: list[float]) -> dict[str, Any]:
    return {
        "score": round(score, 4),
        "success": score >= 0.5,
        "task_id": task_id,
        "total_reward": round(sum(rewards), 4),
        "steps": len(rewards),
        "details": {
            "per_step_rewards": [round(reward, 4) for reward in rewards],
            "max_possible_reward": _MAX_REWARDS.get(task_id, _MAX_REWARDS["task_1"]),
            "score_range": [_SCORE_MIN, _SCORE_MAX],
        },
    }


class _GraderMeta(type):
    """Allow both Task1Grader() and Task1Grader(episode_data)."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], dict) and len(args) == 1 and not kwargs:
            instance = super().__call__()
            return instance(args[0])
        return super().__call__(*args, **kwargs)


class BaseGrader(metaclass=_GraderMeta):
    task_id: str = "task_1"

    def __call__(self, episode_data: dict[str, Any]) -> dict[str, Any]:
        rewards = _extract_rewards(episode_data)
        score = _compute_score(self.task_id, rewards)
        return _make_result(self.task_id, score, rewards)

    def grade(self, episode_data: dict[str, Any]) -> dict[str, Any]:
        return self(episode_data)

    def score(self, episode_data: dict[str, Any]) -> float:
        return float(self(episode_data)["score"])


class Task1Grader(BaseGrader):
    task_id = "task_1"


class Task2Grader(BaseGrader):
    task_id = "task_2"


class Task3Grader(BaseGrader):
    task_id = "task_3"


GRADERS: dict[str, type[BaseGrader]] = {
    "task_1": Task1Grader,
    "task_2": Task2Grader,
    "task_3": Task3Grader,
}


def get_grader(task_id: str) -> BaseGrader:
    grader_cls = GRADERS.get(task_id, Task1Grader)
    return grader_cls()
