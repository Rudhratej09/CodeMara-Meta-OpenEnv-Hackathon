"""Top-level task exports for validator compatibility."""

from __future__ import annotations

from graders import grade_task_1, grade_task_2, grade_task_3
from server.tasks import TASKS as TASK_REGISTRY
from server.tasks import get_task


TASKS = [
    {
        "id": "task_1",
        "task_id": "task_1",
        "name": "Single Query Routing",
        "title": "Single Query Routing",
        "difficulty": "easy",
        "description": "Route a single LLM query to the optimal model tier while minimising carbon footprint and latency.",
        "max_steps": 10,
        "grader": grade_task_1,
        "grader_path": "graders:grade_task_1",
        "graders": [grade_task_1],
        "grader_paths": ["graders:grade_task_1"],
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "task_2",
        "task_id": "task_2",
        "name": "Multi-Query Episode",
        "title": "Multi-Query Episode",
        "difficulty": "medium",
        "description": "Route 3 queries of varying complexity. LARGE model incurs -0.2 penalty per use.",
        "max_steps": 20,
        "grader": grade_task_2,
        "grader_path": "graders:grade_task_2",
        "graders": [grade_task_2],
        "grader_paths": ["graders:grade_task_2"],
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "task_3",
        "task_id": "task_3",
        "name": "Stateful Carbon-Aware Routing",
        "title": "Stateful Carbon-Aware Routing",
        "difficulty": "hard",
        "description": "5-query episode with caching, KB lookups, cascade strategies, and carbon-aware waiting.",
        "max_steps": 50,
        "grader": grade_task_3,
        "grader_path": "graders:grade_task_3",
        "graders": [grade_task_3],
        "grader_paths": ["graders:grade_task_3"],
        "reward_range": [0.0, 1.0],
    },
]

TASK_IDS = [task["task_id"] for task in TASKS]

__all__ = ["TASKS", "TASK_IDS", "TASK_REGISTRY", "get_task"]
