"""Top-level grader registry for validator compatibility."""

from __future__ import annotations

from typing import Any

from server.rubrics import get_grader


def _grade(task_id: str, episode_data: dict[str, Any]) -> dict[str, Any]:
    grader = get_grader(task_id)
    result = grader(episode_data)
    score = float(result.get("score", 0.1))
    score = max(0.1, min(0.99, score))
    result["score"] = round(score, 4)
    result["reward"] = round(score, 4)
    result["normalized_reward"] = round(score, 4)
    return result


def EasyGrader(episode_data: dict[str, Any]) -> dict[str, Any]:
    return _grade("easy", episode_data)


def MediumGrader(episode_data: dict[str, Any]) -> dict[str, Any]:
    return _grade("medium", episode_data)


def HardGrader(episode_data: dict[str, Any]) -> dict[str, Any]:
    return _grade("hard", episode_data)


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
    "GRADERS",
]
