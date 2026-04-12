"""
graders.py  (ROOT level — NOT server/graders.py)
================================================
Grader functions for the Eco-LLM Inference Routing environment.

CRITICAL: This file MUST be at the repository ROOT, not inside server/.
The validator imports it as `graders` (no package prefix).
Putting it inside server/ causes a namespace collision that silently
drops tasks — confirmed by reallyaarush commit:
"Fix namespace collision causing Not enough tasks with graders error"

FUNCTION SIGNATURE (required by validator):
    grade_task_N(state: dict, reward: float) -> float

The validator calls each grader as:
    score = grade_task_1(current_state_dict, episode_total_reward)

Returns a float in [0.0, 1.0].

GRADING FORMULA:
    score = clamp(total_reward / max_possible_reward, 0.0, 1.0)

    task_1: max = 1 query  × 1.5 = 1.5
    task_2: max = 3 queries × 1.5 = 4.5
    task_3: max = 5 queries × 1.5 = 7.5
"""

from __future__ import annotations

_MAX_REWARDS = {"task_1": 1.5, "task_2": 4.5, "task_3": 7.5}


def _normalize_reward(task_id: str, reward: float) -> float:
    """Normalise total reward to [0.0, 1.0]."""
    max_r = _MAX_REWARDS.get(task_id, 1.5)
    if max_r <= 0:
        return 0.0
    return float(min(max(reward / max_r, 0.0), 1.0))


def grade_task_1(state: dict, reward: float) -> float:
    """
    Grader for task_1 — Single Query Routing (Easy).
    score = clamp(reward / 1.5, 0.0, 1.0)
    """
    return _normalize_reward("task_1", reward)


def grade_task_2(state: dict, reward: float) -> float:
    """
    Grader for task_2 — Multi-Query Episode (Medium).
    score = clamp(reward / 4.5, 0.0, 1.0)
    """
    return _normalize_reward("task_2", reward)


def grade_task_3(state: dict, reward: float) -> float:
    """
    Grader for task_3 — Stateful Carbon-Aware Routing (Hard).
    score = clamp(reward / 7.5, 0.0, 1.0)
    """
    return _normalize_reward("task_3", reward)


# Registry for convenience access
GRADERS = {
    "task_1": grade_task_1,
    "task_2": grade_task_2,
    "task_3": grade_task_3,
}

# Required by openenv validator for task discovery
TASK_GRADER_PAIRS = [
    ("task_1", grade_task_1),
    ("task_2", grade_task_2),
    ("task_3", grade_task_3),
]

__all__ = [
    "grade_task_1",
    "grade_task_2",
    "grade_task_3",
    "GRADERS",
    "TASK_GRADER_PAIRS",
]
