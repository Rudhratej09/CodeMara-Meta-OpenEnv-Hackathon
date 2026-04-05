"""Environment logic for Eco-LLM inference routing."""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from server.models import ModelChoice, RLAction, RLObservation, RLReward, RLState, Strategy
from server.tasks import (
    CARBON_SCHEDULE,
    ENERGY_COSTS,
    KB_ENERGY_COST,
    KB_LATENCY_COST,
    LATENCY_COSTS,
    LATENCY_MU,
    MODEL_ORDER,
    MODEL_RANK,
    QuerySpec,
    TaskSpec,
    get_task,
)


class EcoLLMInferenceRoutingEnvironment(Environment[RLAction, RLObservation, RLState]):
    """Deterministic hierarchical RL environment for LLM routing."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.current_task: TaskSpec = get_task("task_1")
        self._state = RLState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self.current_task.task_id,
            step_index=0,
            query_index=0,
            total_energy=0.0,
            total_latency=0.0,
            cache_contents=[],
            carbon_history_index=0,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "task_1",
        **kwargs: object,
    ) -> RLObservation:
        del seed, kwargs
        self.current_task = get_task(task_id)
        self._state = RLState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self.current_task.task_id,
            step_index=0,
            query_index=0,
            total_energy=0.0,
            total_latency=0.0,
            cache_contents=[],
            carbon_history_index=0,
        )
        return self._build_observation(
            reward=self._empty_reward(["reset"]),
            done=False,
        )

    def step(
        self,
        action: RLAction,
        timeout_s: Optional[float] = None,
        **kwargs: object,
    ) -> RLObservation:
        del timeout_s, kwargs

        if self._episode_complete:
            return self._build_observation(
                reward=self._empty_reward(["step_after_done"]),
                done=True,
            )

        self._state.step_count += 1
        trace = [f"strategy={action.strategy.value}", f"model={action.model_choice.value}"]
        query = self.current_query
        current_carbon = self.current_carbon_intensity

        if action.strategy == Strategy.WAIT:
            self._advance_carbon()
            self._state.step_index += 1
            reward = RLReward(
                score=0.0,
                energy_cost=0.0,
                latency_cost=0.0,
                carbon_penalty=0.0,
                latency_penalty=0.0,
                bonuses=0.0,
                total_reward=0.0,
                correct=False,
                strategy_trace=trace + ["wait"],
            )
            return self._build_observation(reward=reward, done=False)

        correct = False
        energy_cost = 0.0
        latency_cost = 0.0
        bonuses = 0.0

        if action.strategy == Strategy.USE_CACHE and query.text in self.cache_contents:
            correct = True
            bonuses += 0.5
            trace.append("cache_hit")
        elif action.strategy == Strategy.CALL_KB:
            energy_cost += KB_ENERGY_COST
            latency_cost += KB_LATENCY_COST
            correct = query.kb_available
            trace.append("kb_lookup")
        elif action.strategy == Strategy.DO_CASCADE:
            cascade_models = self._cascade_models(action.model_choice)
            for cascade_model in cascade_models:
                energy_cost += ENERGY_COSTS[cascade_model]
                latency_cost += LATENCY_COSTS[cascade_model]
                trace.append(f"cascade:{cascade_model.value}")
                model_correct = self._model_answers_correctly(query, cascade_model)
                if model_correct:
                    correct = True
                    if action.exit_flag:
                        trace.append("cascade_exit")
                        break
        else:
            chosen_model = action.model_choice
            energy_cost += ENERGY_COSTS[chosen_model]
            latency_cost += LATENCY_COSTS[chosen_model]
            correct = self._model_answers_correctly(query, chosen_model)
            trace.append(f"infer:{chosen_model.value}")
            if action.strategy == Strategy.EARLY_EXIT and correct:
                trace.append("early_exit_success")

        if action.strategy == Strategy.EARLY_EXIT and correct:
            bonuses += 0.1

        if action.model_choice == ModelChoice.LARGE:
            bonuses += self.current_task.large_model_penalty
            if self.current_task.large_model_penalty:
                trace.append("task_large_penalty")

        score = 1.0 if correct else 0.0
        carbon_penalty = current_carbon * energy_cost
        latency_penalty = LATENCY_MU * latency_cost
        total_reward = score - carbon_penalty - latency_penalty + bonuses

        self._state.total_energy += energy_cost
        self._state.total_latency += latency_cost
        self._state.step_index += 1

        if correct and query.text not in self.cache_contents:
            self._state.cache_contents.append(query.text)

        if action.strategy != Strategy.WAIT:
            self._advance_query()
            self._advance_carbon()

        done = self._episode_complete
        reward = RLReward(
            score=score,
            energy_cost=round(energy_cost, 4),
            latency_cost=round(latency_cost, 4),
            carbon_penalty=round(carbon_penalty, 4),
            latency_penalty=round(latency_penalty, 4),
            bonuses=round(bonuses, 4),
            total_reward=round(total_reward, 4),
            correct=correct,
            strategy_trace=trace,
        )
        return self._build_observation(reward=reward, done=done)

    @property
    def state(self) -> RLState:
        return self._state

    @property
    def cache_contents(self) -> list[str]:
        return list(self._state.cache_contents)

    @property
    def current_query(self) -> QuerySpec:
        if self._state.query_index >= len(self.current_task.queries):
            return self.current_task.queries[-1]
        return self.current_task.queries[self._state.query_index]

    @property
    def current_carbon_intensity(self) -> float:
        return CARBON_SCHEDULE[self._state.carbon_history_index % len(CARBON_SCHEDULE)]

    @property
    def _episode_complete(self) -> bool:
        return self._state.query_index >= len(self.current_task.queries)

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Eco-LLM Inference Routing Environment",
            description=(
                "Hierarchical RL environment for carbon-aware LLM query routing "
                "across accuracy, latency, energy, and caching objectives."
            ),
            version="1.0.0",
        )

    def _build_observation(self, reward: RLReward, done: bool) -> RLObservation:
        query = self.current_query
        return RLObservation(
            query=query.text,
            cache_contents=self.cache_contents,
            carbon_intensity=self.current_carbon_intensity,
            done=done,
            reward=reward.total_reward,
            correct_answer=query.correct_answer,
            reward_details=reward,
            metadata={
                "task_id": self.current_task.task_id,
                "difficulty": self.current_task.difficulty,
                "state": self._state.public_payload(),
            },
        )

    def _empty_reward(self, trace: list[str]) -> RLReward:
        return RLReward(
            score=0.0,
            energy_cost=0.0,
            latency_cost=0.0,
            carbon_penalty=0.0,
            latency_penalty=0.0,
            bonuses=0.0,
            total_reward=0.0,
            correct=False,
            strategy_trace=trace,
        )

    def _advance_query(self) -> None:
        if self._state.query_index < len(self.current_task.queries):
            self._state.query_index += 1

    def _advance_carbon(self) -> None:
        self._state.carbon_history_index = (self._state.carbon_history_index + 1) % len(CARBON_SCHEDULE)

    def _cascade_models(self, upper_bound: ModelChoice) -> list[ModelChoice]:
        max_index = MODEL_RANK[upper_bound]
        return MODEL_ORDER[: max_index + 1]

    def _model_answers_correctly(self, query: QuerySpec, model: ModelChoice) -> bool:
        return MODEL_RANK[model] >= MODEL_RANK[query.minimum_model]
