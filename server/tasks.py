"""Deterministic task definitions for the Eco-LLM routing environment."""

from __future__ import annotations

from dataclasses import dataclass

from server.models import ModelChoice


ENERGY_COSTS = {
    ModelChoice.SMALL: 0.1,
    ModelChoice.MEDIUM: 0.3,
    ModelChoice.LARGE: 0.6,
}

LATENCY_COSTS = {
    ModelChoice.SMALL: 1.0,
    ModelChoice.MEDIUM: 2.0,
    ModelChoice.LARGE: 5.0,
}

MODEL_ORDER = [ModelChoice.SMALL, ModelChoice.MEDIUM, ModelChoice.LARGE]
MODEL_RANK = {model: index for index, model in enumerate(MODEL_ORDER)}
CARBON_SCHEDULE = [0.8, 0.6, 0.4, 0.3, 0.5]
LATENCY_MU = 0.1        
MAX_CARBON_PENALTY = 0.8
KB_ENERGY_COST = 0.05
KB_LATENCY_COST = 1.5
MAX_STEPS_PER_EPISODE = 50


@dataclass(frozen=True)
class QuerySpec:
    text: str
    correct_answer: str
    minimum_model: ModelChoice
    kb_available: bool = False


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    queries: tuple[QuerySpec, ...]
    large_model_penalty: float = 0.0
    description: str = ""


TASKS: dict[str, TaskSpec] = {
    "easy": TaskSpec(
        task_id="easy",
        difficulty="easy",
        queries=(
            QuerySpec(
                text="What is the most efficient insulation material for a passive house retrofit?",
                correct_answer="Vacuum insulated panels",
                minimum_model=ModelChoice.MEDIUM,
            ),
        ),
        description="Single-query routing with accuracy-only emphasis.",
    ),
    "medium": TaskSpec(
        task_id="medium",
        difficulty="medium",
        queries=(
            QuerySpec(
                text="Summarize the trade-off between quantization and factual accuracy in a customer support bot.",
                correct_answer="Aggressive quantization lowers energy use but can reduce factual reliability.",
                minimum_model=ModelChoice.SMALL,
            ),
            QuerySpec(
                text="Explain when speculative decoding improves throughput on medium-sized models.",
                correct_answer="It helps when draft tokens are likely to be accepted by the target model.",
                minimum_model=ModelChoice.MEDIUM,
            ),
            QuerySpec(
                text="Design a safety fallback for multilingual medical triage.",
                correct_answer="Use high-capability review with escalation and verified retrieval before final output.",
                minimum_model=ModelChoice.LARGE,
            ),
        ),
        large_model_penalty=-0.2,
        description="Three-query episode with an extra penalty for using the LARGE model.",
    ),
    "hard": TaskSpec(
        task_id="hard",
        difficulty="hard",
        queries=(
            QuerySpec(
                text="Classify whether this FAQ request can be answered from the billing knowledge base.",
                correct_answer="Yes, use the billing KB article.",
                minimum_model=ModelChoice.SMALL,
                kb_available=True,
            ),
            QuerySpec(
                text="Route a long-form reasoning question about carbon-aware serving policy.",
                correct_answer="Use cascade with escalation to a larger model if smaller ones fail.",
                minimum_model=ModelChoice.MEDIUM,
            ),
            QuerySpec(
                text="Classify whether this FAQ request can be answered from the billing knowledge base.",
                correct_answer="Yes, use the billing KB article.",
                minimum_model=ModelChoice.SMALL,
                kb_available=True,
            ),
            QuerySpec(
                text="Generate a compliant answer for a finance-regulated enterprise user.",
                correct_answer="Use the large model with a retrieval-backed compliance layer.",
                minimum_model=ModelChoice.LARGE,
            ),
            QuerySpec(
                text="Route a long-form reasoning question about carbon-aware serving policy.",
                correct_answer="Use cascade with escalation to a larger model if smaller ones fail.",
                minimum_model=ModelChoice.MEDIUM,
            ),
        ),
        description="Stateful five-query episode with repeats, caching, cascading, early exit, and waiting.",
    ),
}


def get_task(task_id: str) -> TaskSpec:
    """Return a task by id, defaulting to the easy task for unknown ids."""
    return TASKS.get(task_id, TASKS["easy"])
