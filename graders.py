"""Top-level grader registry for validator compatibility."""

from __future__ import annotations

from typing import Any

from server.rubrics import get_grader


def _clamp_score(value: float) -> float:
    return round(max(0.1, min(0.99, float(value))), 4)


def _grade(task_id: str, *args: Any, **kwargs: Any) -> float:
    if len(args) >= 2 and isinstance(args[1], (int, float)):
        return _clamp_score(float(args[1]))

    episode_data: dict[str, Any]
    if args and isinstance(args[0], dict):
        episode_data = args[0]
    else:
        episode_data = kwargs.get("episode_data", {}) if isinstance(kwargs.get("episode_data"), dict) else {}

    if "reward" in episode_data and isinstance(episode_data["reward"], (int, float)):
        return _clamp_score(float(episode_data["reward"]))

    grader = get_grader(task_id)
    result = grader(episode_data)
    return _clamp_score(float(result.get("score", 0.1)))


def grade_easy(state: dict[str, Any], reward: float) -> float:
    del state
    return _grade("easy", {}, reward)


def grade_medium(state: dict[str, Any], reward: float) -> float:
    del state
    return _grade("medium", {}, reward)


def grade_hard(state: dict[str, Any], reward: float) -> float:
    del state
    return _grade("hard", {}, reward)


def grade_task_1(state: dict[str, Any], reward: float) -> float:
    return grade_easy(state, reward)


def grade_task_2(state: dict[str, Any], reward: float) -> float:
    return grade_medium(state, reward)


def grade_task_3(state: dict[str, Any], reward: float) -> float:
    return grade_hard(state, reward)


def EasyGrader(*args: Any, **kwargs: Any) -> float:
    return _grade("easy", *args, **kwargs)


def MediumGrader(*args: Any, **kwargs: Any) -> float:
    return _grade("medium", *args, **kwargs)


def HardGrader(*args: Any, **kwargs: Any) -> float:
    return _grade("hard", *args, **kwargs)


Task1Rubric = EasyGrader
Task2Rubric = MediumGrader
Task3Rubric = HardGrader
Task1Grader = EasyGrader
Task2Grader = MediumGrader
Task3Grader = HardGrader


GRADERS = {
    "easy": EasyGrader,
    "medium": MediumGrader,
    "hard": HardGrader,
}


__all__ = [
    "EasyGrader",
    "MediumGrader",
    "HardGrader",
    "Task1Rubric",
    "Task2Rubric",
    "Task3Rubric",
    "Task1Grader",
    "Task2Grader",
    "Task3Grader",
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "grade_task_1",
    "grade_task_2",
    "grade_task_3",
    "GRADERS",
]
