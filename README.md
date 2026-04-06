# Eco-LLM Inference Routing Environment

> An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment for carbon-aware, multi-objective LLM query routing.

---

## Overview

This environment simulates the problem of routing incoming LLM queries to the most appropriate model tier (small, medium, or large) while jointly optimising for **accuracy**, **energy efficiency**, **latency**, and **carbon footprint**.

At each step, an agent observes a query, the current carbon intensity of the grid, and the contents of a response cache. It then selects a routing *strategy* and a target *model*, receiving a structured reward signal that reflects all four objectives.

The environment is fully deterministic — given the same actions, every episode plays out identically — making it suitable for reproducible evaluation and grading.

---

## Environment Design

### Action Space

Each action is an `RLAction` with three fields:

| Field | Type | Description |
|---|---|---|
| `strategy` | `Strategy` | How to handle the query (see below) |
| `model_choice` | `ModelChoice` | Target model: `SMALL`, `MEDIUM`, or `LARGE` |
| `exit_flag` | `bool` | Whether to terminate a cascade early on first correct answer |

**Available strategies:**

| Strategy | Behaviour |
|---|---|
| `NONE` | Direct inference with the chosen model |
| `USE_CACHE` | Serve from cache if this query was answered correctly before |
| `DO_CASCADE` | Try models from smallest up to the chosen model, stopping on `exit_flag` |
| `EARLY_EXIT` | Infer with chosen model; bonus reward if correct and stops early |
| `CALL_KB` | Look up a knowledge base; only succeeds if `kb_available` for the query |
| `WAIT` | Skip this step to wait for lower carbon intensity |

### Observation Space

Each `RLObservation` contains:

| Field | Description |
|---|---|
| `query` | The current query text |
| `carbon_intensity` | Grid carbon intensity this step (0.0 = clean, 1.0 = dirty) |
| `cache_contents` | List of queries already answered correctly (available for free re-use) |
| `correct_answer` | Ground-truth answer (used for grading) |
| `reward_details` | Full structured `RLReward` breakdown |
| `done` | Whether the episode has ended |

### Reward Function

```
reward = score − carbon_penalty − latency_penalty + bonuses
```

| Component | Formula | Notes |
|---|---|---|
| `score` | `1.0` if correct, else `0.0` | Primary accuracy signal |
| `carbon_penalty` | `carbon_intensity × energy_cost` | Penalises dirty-grid inference |
| `latency_penalty` | `0.01 × latency_cost` | Soft latency pressure |
| `bonuses` | Cache hit: `+0.5`, Early exit: `+0.1` | Efficiency rewards |
| Task penalty | Configurable per task (e.g. `−0.2` for LARGE on task 2) | Discourages over-provisioning |

**Energy and latency costs per model:**

| Model | Energy cost | Latency cost |
|---|---|---|
| `SMALL` | 0.1 | 1.0 |
| `MEDIUM` | 0.3 | 2.0 |
| `LARGE` | 0.6 | 5.0 |

### Tasks

Three tasks of increasing difficulty are included:

| Task | Difficulty | Queries | Description |
|---|---|---|---|
| `task_1` | Easy | 1 | Single-query routing; accuracy emphasis |
| `task_2` | Medium | 3 | Multi-query episode; LARGE model penalised |
| `task_3` | Hard | 5 | Stateful episode with repeated queries, caching, cascading, KB lookups, and carbon-aware waiting |

---

## Project Structure

```
eco_llm_inference_routing/
├── openenv.yaml          # OpenEnv spec manifest
├── pyproject.toml        # Python package and dependency config
├── uv.lock               # Locked dependency tree
├── baseline.py           # Baseline agents and CLI runner
├── server/
│   ├── app.py            # FastAPI app (OpenEnv HTTP server)
│   ├── env.py            # EcoLLMInferenceRoutingEnvironment (core logic)
│   ├── models.py         # RLAction, RLObservation, RLState, RLReward
│   └── tasks.py          # Task definitions, query specs, reward constants
└── deployment/
    └── Dockerfile        # Multi-stage build from openenv-base
```

---

## Setup

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- Docker (for containerised deployment)

### Install dependencies

```bash
# With uv (recommended)
uv sync

# With pip
pip install -e .
```

### Run the server locally

```bash
python -m server.app
# or
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The server exposes OpenEnv-standard endpoints at `http://localhost:8000`:
- `POST /reset` — start a new episode
- `POST /step` — take an action
- `GET /state` — inspect current state

---

## Running Baselines

The `baseline.py` script provides three agents and a full CLI:

```bash
# Heuristic agent on the hard task, 10 episodes
python baseline.py --task task_3 --policy heuristic --episodes 10

# Random agent on easy task
python baseline.py --task easy --policy random --episodes 5

# LLM agent (calls Claude via Anthropic API)
python baseline.py --task task_2 --policy llm --episodes 3
```

**Available options:**

| Flag | Values | Default | Description |
|---|---|---|---|
| `--task` | `easy`, `medium`, `hard`, `task_1`, `task_2`, `task_3` | `task_1` | Which task to run |
| `--policy` | `random`, `heuristic`, `llm` | `heuristic` | Which agent to use |
| `--episodes` | integer | `10` | Number of episodes to run |

**Example output:**
```
=======================================================
  Eco-LLM Routing Baseline
  Task: task_3  |  Policy: heuristic  |  Episodes: 10
=======================================================
  Episode   1: reward=2.3150
  Episode   2: reward=2.4870
  ...
  Mean  : 2.3940
  Stdev : 0.0812
=======================================================
```

### Agent descriptions

**`RandomPolicyAgent`** — samples strategies and models uniformly at random. Seeded for reproducibility. Useful as a lower-bound baseline.

**`HeuristicEscalationAgent`** — starts with the SMALL model and escalates after incorrect answers. Checks the cache before inferring, uses the KB when available, and waits when carbon intensity is high and the episode is not on its final query.

**`LLMPolicyAgent`** — uses Claude (via the Anthropic SDK) to choose a strategy and model at each step. Constructs a structured prompt with the current query, carbon intensity, cache state, and KB availability. Falls back to the SMALL/NONE heuristic on API errors. Requires `pip install anthropic` and a valid `ANTHROPIC_API_KEY`.

---

## Docker Deployment

Build and run the environment as a container:

```bash
# Build
docker build -t eco-llm-routing -f deployment/Dockerfile .

# Run
docker run -p 8000:8000 eco-llm-routing
```

The container serves the OpenEnv HTTP interface at port 8000 and includes a health check at `GET /health`.

To deploy to Hugging Face Spaces via the OpenEnv CLI:

```bash
pip install openenv-core
openenv push --repo-id your-org/eco-llm-routing
```

---

## OpenEnv Compliance

This environment conforms to the [OpenEnv spec](https://github.com/meta-pytorch/OpenEnv):

- `openenv.yaml` with `spec_version: 1`, `type: space`, `runtime: fastapi`
- `EcoLLMInferenceRoutingEnvironment` extends `openenv.core.env_server.interfaces.Environment`
- Typed `Action`, `Observation`, and `State` via Pydantic models
- Server bootstrapped with `create_app(...)` from `openenv.core.env_server.http_server`
- Multi-stage Dockerfile based on `ghcr.io/meta-pytorch/openenv-base`

---

## License

BSD-style license — see [LICENSE](./LICENSE) for full terms.  
Copyright (c) Meta Platforms, Inc. and affiliates.
