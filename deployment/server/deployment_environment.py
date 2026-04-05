# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Eco-LLM Inference Routing Environment.

Hierarchical multi-objective LLM query routing with:
- Task 1 (easy):   1-step, pick the right model for a single query.
- Task 2 (medium): 3-step sequence, accuracy + energy penalty for LARGE.
- Task 3 (hard):   5-step cascade with cache, cascade, early-exit, wait, CALL_KB.

Reward per step:
    R_t = s_t - λ·(E_t * CI_t) - μ·L_t + bonus_t - penalty_t
"""

import random
from uuid import uuid4
from typing import List, Dict, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EcoLLMAction, EcoLLMObservation
except ImportError:
    from models import EcoLLMAction, EcoLLMObservation


# ── Model cost tables ────────────────────────────────────────────────────────
MODEL_ENERGY: Dict[str, float] = {"SMALL": 0.1, "MEDIUM": 0.3, "LARGE": 0.6}   # kWh
MODEL_LATENCY: Dict[str, float] = {"SMALL": 1.0, "MEDIUM": 2.0, "LARGE": 5.0}  # seconds

# Reward hyper-parameters
LAMBDA = 1.0   # weight on energy*CI
MU = 0.01      # weight on latency
GAMMA = 1.0    # penalty for wrong answer
CACHE_BONUS = 0.5
WAIT_TIME_COST = 0.5  # latency penalty for WAIT

# ── Task datasets ────────────────────────────────────────────────────────────
# Each entry: (query, correct_model, correct_answer_stub)
TASK1_QUERIES: List[Tuple[str, str, str]] = [
    ("What is photosynthesis?",          "SMALL",  "photosynthesis"),
    ("Summarise the French Revolution.", "MEDIUM", "french revolution"),
    ("Prove the Riemann Hypothesis.",    "LARGE",  "riemann"),
    ("What is 2+2?",                     "SMALL",  "4"),
    ("Explain transformer attention.",   "MEDIUM", "attention"),
]

TASK2_QUERIES: List[Tuple[str, str, str]] = [
    ("What is 2+2?",                            "SMALL",  "4"),
    ("Summarise climate change in 3 sentences.", "MEDIUM", "climate"),
    ("List all prime numbers less than 20.",     "SMALL",  "2,3,5,7,11,13,17,19"),
]

TASK3_QUERIES: List[Tuple[str, str, str]] = [
    ("Who won the FIFA World Cup 2022?",   "SMALL",  "argentina"),
    ("Who won the FIFA World Cup 2022?",   "SMALL",  "argentina"),   # repeat → cache hit expected
    ("Solve: x^2 - 5x + 6 = 0",           "MEDIUM", "x=2,x=3"),
    ("What is the weather like tomorrow?", "CALL_KB","weather"),
    ("Summarise the last 5 US presidents.","LARGE",  "presidents"),
]


def _carbon_intensity(step: int) -> float:
    """Simulate carbon intensity varying over time (deterministic for reproducibility)."""
    # Oscillates between 0.2 and 0.8 over 10 steps
    return 0.5 + 0.3 * ((-1) ** step)


def _model_correct(strategy: str, model_choice: str, correct_model: str, answer: str) -> float:
    """
    Simulate whether the chosen model answers correctly.
    Returns 1.0 if correct, 0.0 otherwise.
    SMALL is correct only for SMALL tasks, MEDIUM for MEDIUM+SMALL, LARGE always correct.
    """
    capability = {"SMALL": 1, "MEDIUM": 2, "LARGE": 3}
    required   = {"SMALL": 1, "MEDIUM": 2, "LARGE": 3}
    req = required.get(correct_model, 1)
    cap = capability.get(model_choice, 1)
    return 1.0 if cap >= req else 0.0


class EcoLLMEnvironment(Environment):
    """
    Eco-LLM Inference Routing Environment.

    The agent observes a query, carbon intensity, and cache state, then
    chooses a routing strategy and model. Reward balances correctness,
    energy, latency, and carbon intensity.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "easy"):
        self._task = task
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cache: List[str] = []
        self._total_reward: float = 0.0
        self._queries: List[Tuple[str, str, str]] = []
        self._current_ci: float = 0.5
        self._done: bool = False

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> EcoLLMObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cache = []
        self._total_reward = 0.0
        self._done = False

        if self._task == "easy":
            self._queries = [random.choice(TASK1_QUERIES)]
        elif self._task == "medium":
            self._queries = list(TASK2_QUERIES)
        else:
            self._queries = list(TASK3_QUERIES)

        self._current_ci = _carbon_intensity(0)
        return self._observe()

    def step(self, action: EcoLLMAction) -> EcoLLMObservation:  # type: ignore[override]
        if self._done:
            return self._observe(info={"warning": "Episode already done. Call reset()."})

        step = self._state.step_count
        query, correct_model, answer_stub = self._queries[step]

        strategy = action.strategy.upper()
        model = action.model_choice.upper()

        # Validate
        if strategy not in ("NONE", "USE_CACHE", "DO_CASCADE", "EARLY_EXIT", "WAIT", "CALL_KB"):
            strategy = "NONE"
        if model not in ("SMALL", "MEDIUM", "LARGE"):
            model = "SMALL"

        reward, info = self._compute_reward(strategy, model, query, correct_model, answer_stub, step)

        # Update cache if we got a correct answer this step
        if info.get("correct") and query not in self._cache:
            self._cache.append(query)

        # Carbon intensity for next step (WAIT causes it to drop)
        if strategy == "WAIT":
            self._current_ci = max(0.1, self._current_ci - 0.2)
        else:
            self._current_ci = _carbon_intensity(step + 1)

        self._total_reward += reward
        self._state.step_count += 1

        self._done = self._state.step_count >= len(self._queries)
        return self._observe(reward=reward, info=info)

    @property
    def state(self) -> State:
        return self._state

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _observe(self, reward: float = 0.0, info: dict = None) -> EcoLLMObservation:
        step = self._state.step_count
        if step < len(self._queries):
            query = self._queries[step][0]
        else:
            query = ""

        return EcoLLMObservation(
            query=query,
            cache_contents=list(self._cache),
            carbon_intensity=self._current_ci,
            step=step,
            total_steps=len(self._queries),
            task=self._task,
            done=self._done,
            reward=reward,
            info=info or {},
        )

    def _compute_reward(
        self,
        strategy: str,
        model: str,
        query: str,
        correct_model: str,
        answer_stub: str,
        step: int,
    ) -> Tuple[float, dict]:
        """
        Compute per-step reward.

        R = s - λ·(E·CI) - μ·L + bonus - penalty
        """
        ci = self._current_ci
        info: dict = {"strategy": strategy, "model": model, "ci": ci}

        # ── WAIT: no computation, just delay ──────────────────────────────
        if strategy == "WAIT":
            reward = -MU * WAIT_TIME_COST
            info["correct"] = False
            info["note"] = "Waited. Carbon will drop next step."
            return reward, info

        # ── USE_CACHE ────────────────────────────────────────────────────
        if strategy == "USE_CACHE":
            if query in self._cache:
                reward = 1.0 + CACHE_BONUS   # correct answer free from cache
                info["correct"] = True
                info["note"] = "Cache hit!"
            else:
                reward = -GAMMA              # penalise invalid cache call
                info["correct"] = False
                info["note"] = "Cache miss — no answer cached for this query."
            return reward, info

        # ── CALL_KB: external knowledge base (simulate always correct) ────
        if strategy == "CALL_KB":
            E = MODEL_ENERGY["SMALL"]        # KB call is cheap
            L = MODEL_LATENCY["SMALL"]
            reward = 1.0 - LAMBDA * E * ci - MU * L
            info["correct"] = True
            info["note"] = "Knowledge base returned answer."
            return reward, info

        # ── DO_CASCADE: SMALL → MEDIUM → LARGE until correct ─────────────
        if strategy == "DO_CASCADE":
            cumulative_E = 0.0
            cumulative_L = 0.0
            correct = False
            for m in ["SMALL", "MEDIUM", "LARGE"]:
                cumulative_E += MODEL_ENERGY[m]
                cumulative_L += MODEL_LATENCY[m]
                if _model_correct(strategy, m, correct_model, answer_stub):
                    correct = True
                    break
            s = 1.0 if correct else 0.0
            reward = s - LAMBDA * cumulative_E * ci - MU * cumulative_L
            info["correct"] = correct
            info["cascade_energy"] = cumulative_E
            info["note"] = f"Cascade finished. Correct={correct}"
            return reward, info

        # ── EARLY_EXIT: try SMALL, exit if correct ────────────────────────
        if strategy == "EARLY_EXIT":
            if _model_correct(strategy, "SMALL", correct_model, answer_stub):
                E = MODEL_ENERGY["SMALL"]
                L = MODEL_LATENCY["SMALL"]
                reward = 1.0 - LAMBDA * E * ci - MU * L
                info["correct"] = True
                info["note"] = "Early exit succeeded with SMALL model."
            else:
                # Fall back to chosen model
                E = MODEL_ENERGY["SMALL"] + MODEL_ENERGY[model]
                L = MODEL_LATENCY["SMALL"] + MODEL_LATENCY[model]
                correct = _model_correct(strategy, model, correct_model, answer_stub)
                s = 1.0 if correct else 0.0
                reward = s - LAMBDA * E * ci - MU * L - (GAMMA if not correct else 0)
                info["correct"] = correct
                info["note"] = f"Early exit failed; fell back to {model}."
            return reward, info

        # ── NONE: direct model call ───────────────────────────────────────
        correct = bool(_model_correct(strategy, model, correct_model, answer_stub))
        s = 1.0 if correct else 0.0
        E = MODEL_ENERGY[model]
        L = MODEL_LATENCY[model]
        penalty = 0.0

        # Task 2: extra energy penalty for unnecessary LARGE usage
        if self._task == "medium" and model == "LARGE":
            penalty = 0.2

        reward = s - LAMBDA * E * ci - MU * L - penalty - (GAMMA if not correct else 0)
        info["correct"] = correct
        info["energy"] = E
        info["latency"] = L
        info["note"] = f"Direct call to {model}. Correct={correct}"
        return reward, info


# ── Factory functions used by app.py ─────────────────────────────────────────

def make_easy_env() -> "EcoLLMEnvironment":
    return EcoLLMEnvironment(task="easy")

def make_medium_env() -> "EcoLLMEnvironment":
    return EcoLLMEnvironment(task="medium")

def make_hard_env() -> "EcoLLMEnvironment":
    return EcoLLMEnvironment(task="hard")
