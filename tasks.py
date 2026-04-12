"""Top-level task exports for validator compatibility."""

from __future__ import annotations

from graders import grade_easy, grade_hard, grade_medium
from server.tasks import TASKS as TASK_REGISTRY
from server.tasks import get_task


TASKS = [
    {
        "id": "easy",
        "task_id": "easy",
        "name": "easy",
        "title": "Single Query Routing",
        "difficulty": "easy",
        "description": "Route a single LLM query to the optimal model tier while minimising carbon footprint and latency.",
        "max_steps": 10,
        "grader": grade_easy,
        "grader_path": "graders:grade_easy",
        "graders": [grade_easy],
        "grader_paths": ["graders:grade_easy"],
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "medium",
        "task_id": "medium",
        "name": "medium",
        "title": "Multi-Query Episode",
        "difficulty": "medium",
        "description": "Route 3 queries of varying complexity. LARGE model incurs -0.2 penalty per use.",
        "max_steps": 20,
        "grader": grade_medium,
        "grader_path": "graders:grade_medium",
        "graders": [grade_medium],
        "grader_paths": ["graders:grade_medium"],
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "hard",
        "task_id": "hard",
        "name": "hard",
        "title": "Stateful Carbon-Aware Routing",
        "difficulty": "hard",
        "description": "5-query episode with caching, KB lookups, cascade strategies, and carbon-aware waiting.",
        "max_steps": 50,
        "grader": grade_hard,
        "grader_path": "graders:grade_hard",
        "graders": [grade_hard],
        "grader_paths": ["graders:grade_hard"],
        "reward_range": [0.0, 1.0],
    },
]

TASK_IDS = [task["task_id"] for task in TASKS]

TASK_MAP = {task["task_id"]: task for task in TASKS}
easy = TASK_MAP["easy"]
medium = TASK_MAP["medium"]
hard = TASK_MAP["hard"]

__all__ = [
    "TASKS",
    "TASK_IDS",
    "TASK_MAP",
    "TASK_REGISTRY",
    "easy",
    "medium",
    "hard",
    "get_task",
]
