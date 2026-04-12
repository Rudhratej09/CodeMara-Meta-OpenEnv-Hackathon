"""Top-level grader functions for validator compatibility.

The validator resolves 'graders:grade_easy' by:
  1. import graders
  2. grade_easy = getattr(graders, 'grade_easy')
  3. result = grade_easy(episode_data)   ← single dict argument

Each function accepts a single episode_data dict and returns a dict
with a 'score' key in [0.1, 0.99].
"""

from __future__ import annotations

from typing import Any

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_REWARDS: dict[str, float] = {
    "easy":   1.5,   # 1 query  × 1.5
    "medium": 4.5,   # 3 queries × 1.5
    "hard":   7.5,   # 5 queries × 1.5
    "task_1": 1.5,
    "task_2": 4.5,
    "task_3": 7.5,
}

_SCORE_MIN: float = 0.1
_SCORE_MAX: float = 0.99


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_rewards(episode_data: dict[str, Any]) -> list[float]:
    rewards = episode_data.get("rewards") or episode_data.get("per_step_rewards")
    if rewards:
        return [float(r) for r in rewards]
    # Single reward field
    if "reward" in episode_data:
        return [float(episode_data["reward"])]
    return []


def _compute_score(task_id: str, episode_data: dict[str, Any]) -> float:
    rewards = _extract_rewards(episode_data)
    total = sum(rewards)
    max_r = _MAX_REWARDS.get(task_id, 1.5)
    raw = total / max_r if max_r > 0 else 0.0
    return round(max(_SCORE_MIN, min(_SCORE_MAX, raw)), 4)


def _make_result(task_id: str, score: float, episode_data: dict[str, Any]) -> dict[str, Any]:
    rewards = _extract_rewards(episode_data)
    return {
        "score": score,
        "reward": score,
        "normalized_reward": score,
        "success": score >= 0.5,
        "task_id": task_id,
        "total_reward": round(sum(rewards), 4),
        "steps": len(rewards),
        "details": {
            "per_step_rewards": [round(r, 4) for r in rewards],
            "max_possible_reward": _MAX_REWARDS.get(task_id, 1.5),
            "score_range": [_SCORE_MIN, _SCORE_MAX],
        },
    }


# ── Grader functions ───────────────────────────────────────────────────────────
# Each accepts a single episode_data dict (validator pattern) OR
# (state, reward) positional args (legacy pattern) — both work.

def grade_easy(episode_data_or_state: dict[str, Any] = None,
               reward: float = None, **kwargs: Any) -> dict[str, Any]:
    """Grader for easy task — Single Query Routing."""
    if reward is not None:
        # Legacy (state, reward) call pattern
        episode_data: dict[str, Any] = {"reward": reward}
    else:
        episode_data = episode_data_or_state or {}
    score = _compute_score("easy", episode_data)
    return _make_result("easy", score, episode_data)


def grade_medium(episode_data_or_state: dict[str, Any] = None,
                 reward: float = None, **kwargs: Any) -> dict[str, Any]:
    """Grader for medium task — Multi-Query Episode."""
    if reward is not None:
        episode_data: dict[str, Any] = {"reward": reward}
    else:
        episode_data = episode_data_or_state or {}
    score = _compute_score("medium", episode_data)
    return _make_result("medium", score, episode_data)


def grade_hard(episode_data_or_state: dict[str, Any] = None,
               reward: float = None, **kwargs: Any) -> dict[str, Any]:
    """Grader for hard task — Stateful Carbon-Aware Routing."""
    if reward is not None:
        episode_data: dict[str, Any] = {"reward": reward}
    else:
        episode_data = episode_data_or_state or {}
    score = _compute_score("hard", episode_data)
    return _make_result("hard", score, episode_data)


# ── Aliases ────────────────────────────────────────────────────────────────────

grade_task_1 = grade_easy
grade_task_2 = grade_medium
grade_task_3 = grade_hard

EasyGrader   = grade_easy
MediumGrader = grade_medium
HardGrader   = grade_hard

Task1Rubric  = grade_easy
Task2Rubric  = grade_medium
Task3Rubric  = grade_hard

Task1Grader  = grade_easy
Task2Grader  = grade_medium
Task3Grader  = grade_hard

GRADERS: dict[str, Any] = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
    "task_1": grade_easy,
    "task_2": grade_medium,
    "task_3": grade_hard,
}

__all__ = [
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "grade_task_1",
    "grade_task_2",
    "grade_task_3",
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
