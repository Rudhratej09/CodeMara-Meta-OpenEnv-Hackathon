"""
server/graders.py
=================
Grader CLASSES for the Eco-LLM Inference Routing tasks.

WHY CLASSES (not functions)
────────────────────────────
The OpenEnv Phase 2 validator resolves the grader field in openenv.yaml by:
  1. Splitting  "server.graders:Task1Grader"  on ":"
  2. importlib.import_module("server.graders")
  3. getattr(module, "Task1Grader")
  4. instance = Task1Grader()          ← instantiates the class
  5. result   = instance(episode_data) ← calls it with episode data

A plain function reference like "server.app:grade_episode" fails at step 4
because grade_episode() requires positional arguments and cannot be called
as grade_episode() with no args to produce an instance.

WHY SCORES IN [0.01, 0.99]
────────────────────────────
The Phase 2 validator applies a strict inequality check: 0.0 < score < 1.0.
Returning exactly 0.0 or 1.0 triggers "Score Out of Range" and fails the task.
Community-verified fix (April 2026): clamp to [0.01, 0.99].

GRADING FORMULA
────────────────
score = clamp(total_reward / max_possible_reward, 0.01, 0.99)

  task_1 → max = 1 query  × 1.5 = 1.5
  task_2 → max = 3 queries × 1.5 = 4.5
  task_3 → max = 5 queries × 1.5 = 7.5

The grader accepts any of these call signatures:
  grader(episode_data_dict)
  grader.grade(episode_data_dict)
  grader.score(episode_data_dict)
"""

from __future__ import annotations

from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum achievable reward per query:
#   score(1.0) + cache_bonus(0.5) − 0 penalties = 1.5
_MAX_REWARD_PER_QUERY: float = 1.5

_MAX_REWARDS: dict[str, float] = {
    "task_1": 1.0 * _MAX_REWARD_PER_QUERY,   # 1.5
    "task_2": 3.0 * _MAX_REWARD_PER_QUERY,   # 4.5
    "task_3": 5.0 * _MAX_REWARD_PER_QUERY,   # 7.5
}

# Phase 2 validator requires STRICT inequality: 0.0 < score < 1.0
# Clamp to [0.01, 0.99] to guarantee this regardless of episode outcome.
_SCORE_MIN: float = 0.01
_SCORE_MAX: float = 0.99


def _compute_score(task_id: str, rewards: list[float]) -> float:
    """Normalise total reward to [0.01, 0.99]."""
    max_r = _MAX_REWARDS.get(task_id, _MAX_REWARDS["task_1"])
    total = sum(rewards)
    raw   = total / max_r if max_r > 0 else 0.0
    return float(max(_SCORE_MIN, min(_SCORE_MAX, raw)))


def _extract_rewards(episode_data: dict[str, Any]) -> list[float]:
    """Pull per-step rewards out of whatever the validator sends."""
    rewards = episode_data.get("rewards") or episode_data.get("per_step_rewards") or []
    return [float(r) for r in rewards]


def _make_result(task_id: str, score: float, rewards: list[float]) -> dict[str, Any]:
    return {
        "score":        round(score, 4),
        "success":      score >= 0.5,
        "task_id":      task_id,
        "total_reward": round(sum(rewards), 4),
        "steps":        len(rewards),
        "details": {
            "per_step_rewards":    [round(r, 4) for r in rewards],
            "max_possible_reward": _MAX_REWARDS.get(task_id, 1.5),
            "score_range":         [_SCORE_MIN, _SCORE_MAX],
        },
    }


# ── Grader Classes ────────────────────────────────────────────────────────────
# Each class:
#   - Is instantiable with no arguments: Task1Grader()
#   - Is callable:  instance(episode_data) → result dict
#   - Exposes .grade() and .score() as aliases
# ──────────────────────────────────────────────────────────────────────────────

class Task1Grader:
    """
    Grader for task_1 — Single Query Routing (Easy).

    Score = clamp(total_reward / 1.5, 0.01, 0.99)
    """

    task_id: str = "task_1"

    def __call__(self, episode_data: dict[str, Any]) -> dict[str, Any]:
        rewards = _extract_rewards(episode_data)
        score   = _compute_score(self.task_id, rewards)
        return _make_result(self.task_id, score, rewards)

    def grade(self, episode_data: dict[str, Any]) -> dict[str, Any]:
        return self(episode_data)

    def score(self, episode_data: dict[str, Any]) -> float:
        return self(episode_data)["score"]


class Task2Grader:
    """
    Grader for task_2 — Multi-Query Episode (Medium).

    Score = clamp(total_reward / 4.5, 0.01, 0.99)
    Note: LARGE model incurs -0.2 penalty per use, making 4.5 the ceiling
    only when LARGE is never used and all 3 queries are cache-hit.
    """

    task_id: str = "task_2"

    def __call__(self, episode_data: dict[str, Any]) -> dict[str, Any]:
        rewards = _extract_rewards(episode_data)
        score   = _compute_score(self.task_id, rewards)
        return _make_result(self.task_id, score, rewards)

    def grade(self, episode_data: dict[str, Any]) -> dict[str, Any]:
        return self(episode_data)

    def score(self, episode_data: dict[str, Any]) -> float:
        return self(episode_data)["score"]


class Task3Grader:
    """
    Grader for task_3 — Stateful Carbon-Aware Routing (Hard).

    Score = clamp(total_reward / 7.5, 0.01, 0.99)
    5-query episode with caching, KB lookups, and carbon-aware waiting.
    """

    task_id: str = "task_3"

    def __call__(self, episode_data: dict[str, Any]) -> dict[str, Any]:
        rewards = _extract_rewards(episode_data)
        score   = _compute_score(self.task_id, rewards)
        return _make_result(self.task_id, score, rewards)

    def grade(self, episode_data: dict[str, Any]) -> dict[str, Any]:
        return self(episode_data)

    def score(self, episode_data: dict[str, Any]) -> float:
        return self(episode_data)["score"]


# ── Registry (convenience) ────────────────────────────────────────────────────

GRADERS: dict[str, type] = {
    "task_1": Task1Grader,
    "task_2": Task2Grader,
    "task_3": Task3Grader,
}


def get_grader(task_id: str) -> Task1Grader | Task2Grader | Task3Grader:
    """Return an instantiated grader for the given task_id."""
    cls = GRADERS.get(task_id, Task1Grader)
    return cls()
