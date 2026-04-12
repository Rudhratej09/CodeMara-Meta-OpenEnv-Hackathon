"""
Grader classes for eco_llm_inference_routing.

The validator calls: float(getattr(module, ClassName)().grade(None))
So each grader must be a CLASS with:
  - .grade(episode_data) -> float   (handles None input)
  - float(result) must work -> return plain float, not dict
"""

from __future__ import annotations

from typing import Any

_MAX_REWARD_PER_QUERY: float = 1.5
_MAX_REWARDS: dict[str, float] = {
    "easy":   1 * _MAX_REWARD_PER_QUERY,
    "medium": 3 * _MAX_REWARD_PER_QUERY,
    "hard":   5 * _MAX_REWARD_PER_QUERY,
}
_SCORE_MIN: float = 0.1
_SCORE_MAX: float = 0.99


def _compute_score(task_key: str, episode_data: Any) -> float:
    if episode_data is None:
        return _SCORE_MIN
    if isinstance(episode_data, (int, float)):
        raw = float(episode_data)
        return round(max(_SCORE_MIN, min(_SCORE_MAX, raw)), 4)
    if not isinstance(episode_data, dict):
        return _SCORE_MIN
    # Try direct reward field first
    if "reward" in episode_data:
        raw = float(episode_data["reward"])
        return round(max(_SCORE_MIN, min(_SCORE_MAX, raw)), 4)
    # Try rewards list
    rewards = episode_data.get("rewards") or episode_data.get("per_step_rewards") or []
    total = sum(float(r) for r in rewards)
    max_r = _MAX_REWARDS.get(task_key, _MAX_REWARDS["easy"])
    raw = total / max_r if max_r > 0 else 0.0
    return round(max(_SCORE_MIN, min(_SCORE_MAX, raw)), 4)


class _BaseGrader:
    task_key: str = "easy"

    def grade(self, episode_data: Any) -> float:
        return _compute_score(self.task_key, episode_data)

    def __call__(self, episode_data: Any) -> float:
        return self.grade(episode_data)

    def score(self, episode_data: Any) -> float:
        return self.grade(episode_data)


class EasyGrader(_BaseGrader):
    task_key = "easy"


class MediumGrader(_BaseGrader):
    task_key = "medium"


class HardGrader(_BaseGrader):
    task_key = "hard"


# Aliases
Task1Rubric = EasyGrader
Task2Rubric = MediumGrader
Task3Rubric = HardGrader
grade_easy   = EasyGrader
grade_medium = MediumGrader
grade_hard   = HardGrader

GRADERS: dict[str, type[_BaseGrader]] = {
    "easy":   EasyGrader,
    "medium": MediumGrader,
    "hard":   HardGrader,
    "task_1": EasyGrader,
    "task_2": MediumGrader,
    "task_3": HardGrader,
}

TASK_GRADER_PAIRS = [
    ("easy",   EasyGrader),
    ("medium", MediumGrader),
    ("hard",   HardGrader),
]

__all__ = [
    "EasyGrader",
    "MediumGrader",
    "HardGrader",
    "Task1Rubric",
    "Task2Rubric",
    "Task3Rubric",
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "GRADERS",
    "TASK_GRADER_PAIRS",
]
