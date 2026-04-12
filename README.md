---
title: Eco-LLM Inference Routing Environment
emoji: "🌱"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - llm-routing
  - carbon-aware
  - systems
---

# Eco-LLM Inference Routing Environment

Eco-LLM Inference Routing is an OpenEnv-compatible reinforcement learning environment for carbon-aware LLM serving. The agent must decide how to route each query across model tiers and routing strategies while balancing answer quality, latency, energy use, and grid carbon intensity.

Public Space:
https://huggingface.co/spaces/rudhratej09/Eco-LLM-Carbon-Aware-LLM-Inference-Routing

Live endpoints:
- `/tasks`: https://rudhratej09-eco-llm-carbon-aware-llm-inference-routing.hf.space/tasks
- `/metadata`: https://rudhratej09-eco-llm-carbon-aware-llm-inference-routing.hf.space/metadata
- `/docs`: https://rudhratej09-eco-llm-carbon-aware-llm-inference-routing.hf.space/docs

## Why This Environment

Modern LLM systems do not have a single objective. A router often has to decide whether a request should go to a small model, a larger model, a cache, a retrieval layer, or be delayed until carbon intensity improves. This environment turns that operational tradeoff into a deterministic RL benchmark with structured rewards and graded tasks.

The design stays close to a real serving stack:
- multiple model tiers with different energy and latency costs
- a cache that can eliminate repeat inference cost
- a knowledge-base path for FAQ-style requests
- cascade and early-exit strategies for adaptive compute
- a carbon schedule that makes timing matter

## Tasks

Three tasks are included, each with a deterministic grader returning a score in `[0.0, 1.0]`.

- `task_1`: single-query routing with a simple accuracy-versus-cost tradeoff
- `task_2`: three-query episode with an explicit penalty for overusing the large model
- `task_3`: five-query stateful episode with repeated queries, cache reuse, knowledge-base lookups, cascading, and carbon-aware waiting

Task definitions live in [server/tasks.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/server/tasks.py).

## Action Space

Defined in [server/models.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/server/models.py).

Each action has:
- `strategy`: `NONE`, `USE_CACHE`, `DO_CASCADE`, `EARLY_EXIT`, `WAIT`, or `CALL_KB`
- `model_choice`: `SMALL`, `MEDIUM`, or `LARGE`
- `exit_flag`: whether a cascade should stop early after the first correct answer

## Observation Space

Each observation includes:
- current query text
- current grid carbon intensity
- cache contents
- whether a knowledge base is available
- scalar reward
- detailed reward breakdown
- done flag
- task and state metadata

## Reward Design

The environment is dense-reward, not purely terminal.

Per step:
- `score = 1.0` when the chosen strategy reaches a model capable of solving the query, else `0.0`
- `carbon_penalty = carbon_intensity * energy_cost`, capped at `0.8`
- `latency_penalty = 0.1 * latency_cost`
- cache hits add `+0.5`
- successful early exit adds `+0.1`
- some tasks add an explicit large-model penalty

Model costs:

| Model | Energy cost | Latency cost |
|---|---:|---:|
| `SMALL` | 0.1 | 1.0 |
| `MEDIUM` | 0.3 | 2.0 |
| `LARGE` | 0.6 | 5.0 |

This makes the benchmark about policy quality, not just raw correctness.

## Core Files

- [openenv.yaml](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/openenv.yaml): OpenEnv manifest
- [server/app.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/server/app.py): FastAPI app and HTTP endpoints
- [server/env.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/server/env.py): environment logic
- [server/models.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/server/models.py): typed action, observation, and state models
- [server/tasks.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/server/tasks.py): deterministic task catalog and cost tables
- [server/rubrics.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/server/rubrics.py): rubric-style graders
- [graders.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/graders.py): top-level grader compatibility shims
- [inference.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/inference.py): root-level baseline runner
- [baseline.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/baseline.py): random, heuristic, and LLM baselines

## Local Run

Install dependencies:

```bash
uv sync
```

Start the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Quick checks:

```bash
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/metadata
curl http://127.0.0.1:7860/tasks
```

## Baseline Inference

The required root-level baseline script is [inference.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/inference.py).

Example:

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ECO_LLM_TASK=task_3
python inference.py
```

The script emits OpenEnv-style logs:

```text
[START] task=task_3 env=eco_llm_inference_routing model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=strategy=CALL_KB,model=SMALL,exit=false reward=1.30 done=false error=null
[END] success=true steps=5 score=0.72 rewards=1.30,0.48,1.50,0.20,1.90
```

## Baselines

The repository includes three reference agents:
- random policy
- heuristic router
- LLM-based router

Run comparison:

```bash
python baseline.py compare --task task_3 --episodes 20
```

These baselines are useful because they show the environment is not just a one-step classification toy. Cache-aware and carbon-aware decisions materially change returns on the harder task.

## API Surface

Main endpoints:
- `GET /`
- `GET /health`
- `GET /metadata`
- `GET /schema`
- `GET /tasks`
- `GET /tasks/meta`
- `GET /tasks/{task_id}`
- `POST /reset`
- `POST /step`
- `POST /grader`
- `POST /grade`
- `POST /grader/replay`
- `GET /docs`

## OpenEnv Compatibility

This submission includes the expected OpenEnv pieces:
- `openenv.yaml`
- root-level `inference.py`
- root-level task and grader compatibility exports
- typed action, observation, and state models
- Docker runtime for Hugging Face Spaces deployment

## Docker

Build:

```bash
docker build -t eco-llm-routing .
```

Run:

```bash
docker run -p 7860:7860 eco-llm-routing
```

## Validation Notes

The repository includes lightweight tests in [tests/test_agents.py](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/tests/test_agents.py). For a submission review, the most important checks are:
- server boots cleanly
- `/tasks`, `/metadata`, and `/docs` resolve
- the grader returns normalized scores
- the root-level inference script remains runnable

## License

BSD-style license. See [LICENSE](/C:/Users/Rudhratej/CodeMara-Meta-OpenEnv-Hackathon/LICENSE).
