---
title: Eco-LLM Inference Routing Environment
emoji: "🌱"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
# Eco-LLM Inference Routing Environment

> An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment for carbon-aware, multi-objective LLM query routing.
> link to hugging face space of the environment->https://huggingface.co/spaces/rudhratej09/Eco-LLM-Carbon-Aware-LLM-Inference-Routing



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
| `kb_available`     | Whether a knowledge base is available for the current query  |
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
| `latency_penalty` | `0.1 × latency_cost` | Soft latency pressure |
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
├── Dockerfile            # Multi-stage container build from openenv-base
└── server/
    ├── __init__.py       # Python package marker
    ├── app.py            # FastAPI app (OpenEnv HTTP server)
    ├── env.py            # EcoLLMInferenceRoutingEnvironment (core logic)
    ├── models.py         # RLAction, RLObservation, RLState, RLReward
    └── tasks.py          # Task definitions, query specs, reward constants
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

## Running the Inference Script

The submission-ready inference script is `inference.py` in the repo root. It uses the OpenAI-compatible client and emits mandatory `[START]`/`[STEP]`/`[END]` logs.

```bash
# Required environment variables
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ECO_LLM_TASK=task_1   # task_1 | task_2 | task_3
python inference.py
```

**Example output:**
```
[START] task=task_1 env=eco_llm_inference_routing model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=strategy=NONE,model=MEDIUM,exit=false reward=0.68 done=true error=null
[END] success=true steps=1 rewards=0.68
```

---

## Running Baselines

The `baseline.py` script provides three independent baseline agents for evaluation:

### Quick Start

```bash
# Evaluate single agent
python baseline.py evaluate --agent random --task task_3 --episodes 10
python baseline.py evaluate --agent heuristic --task task_3 --episodes 10
python baseline.py evaluate --agent llm --task task_3 --episodes 10 \
  --api-key $HF_TOKEN --model-name Qwen/Qwen2.5-72B-Instruct
# Compare all agents on same task
python baseline.py compare --task task_3 --episodes 20
```

### Agent Types

#### 1. Random Agent (`--agent random`)

- Policy: Uniformly random strategy + model selection
- Purpose: Lower-bound baseline (no intelligence)
- Use Case: Verify environment is non-trivial
- Expected Performance: ~0.8-1.0 reward (`task_3`)

#### 2. Heuristic Agent (`--agent heuristic`)

- Policy: Rule-based routing
- Check cache first (free re-use)
- Try KB if available
- Wait if carbon > 0.7
- Cascade on hard queries
- Purpose: Mid-range baseline (no LLM calls)
- Use Case: Demonstrate strategic routing matters
- Expected Performance: ~2.0-2.3 reward (`task_3`)

#### 3. LLM Agent (`--agent llm`)

- Policy: Intelligent routing via Claude/GPT/Qwen
- Purpose: Upper-bound baseline (intelligent agent)
- Use Case: Show LLM can optimize trade-offs
- Requirements: Valid `HF_TOKEN` or `OPENAI_API_KEY`
- Expected Performance: ~2.2-2.5 reward (`task_3`)

### Baseline Comparison

```bash
$ python baseline.py compare --task task_3 --episodes 20
======================================================================
  Eco-LLM Baseline: COMPARISON
  Task: task_3 | Episodes: 20
======================================================================
Agent            Mean       Std Dev    Min        Max
----------------------------------------------------------------------
Random           0.8450     0.6200     0.0000     2.1000
Heuristic        2.1400     0.1800     1.9000     2.5000
LLM              2.3100     0.2500     1.8000     2.8000
Relative Performance:
  Heuristic > Random: 2.53x
  LLM > Heuristic: 1.08x
======================================================================
```

### Running on Different Tasks

```bash
# Easy task (1 query)
python baseline.py compare --task task_1 --episodes 50
# Medium task (3 queries)
python baseline.py compare --task task_2 --episodes 30
# Hard task (5 queries, stateful)
python baseline.py compare --task task_3 --episodes 20
```

### API Key Configuration

The LLM agent requires an API key. Set one of:

```bash
# Hugging Face (recommended for Qwen, Llama)
export HF_TOKEN='hf_xxxxxxxxxxxxx'
# OpenAI (for GPT models)
export OPENAI_API_KEY='sk-xxxxxxxxxxxxx'
```

Specify model:

```bash
python baseline.py evaluate --agent llm --task task_3 \
  --model-name Qwen/Qwen2.5-72B-Instruct
```

### Expected Results Summary

| Task | Random | Heuristic | LLM |
|---|---:|---:|---:|
| `task_1` (Easy) | 0.50 | 1.50 | 1.45 |
| `task_2` (Medium) | 1.20 | 2.00 | 2.15 |
| `task_3` (Hard) | 0.85 | 2.14 | 2.31 |

Key insights:

- Random agent struggles because it has no strategy.
- Heuristic agent uses cache, KB, wait, and cascade effectively.
- LLM agent adds a modest improvement through better trade-off decisions.
- The environment demonstrates multi-objective routing trade-offs clearly.

---

## Docker Deployment

Build and run the environment as a container with the root `Dockerfile`:

```bash
# Build
docker build -t eco-llm-routing .
# Run
docker run -p 7860:7860 eco-llm-routing
```

The container serves the OpenEnv HTTP interface at port 7860 and includes a health check at `GET /health`.

To deploy to Hugging Face Spaces:

1. Create a new Docker Space on Hugging Face.
2. Push this repository to that Space.
3. Hugging Face will read the `README.md` front matter and start the container on port `7860`.

To deploy via the OpenEnv CLI:

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
