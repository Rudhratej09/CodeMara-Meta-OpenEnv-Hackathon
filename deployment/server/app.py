# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Eco-LLM Inference Routing Environment.

Endpoints:
    POST /reset  – Reset environment (accepts ?task=easy|medium|hard)
    POST /step   – Execute an action
    GET  /state  – Current state
    GET  /schema – Action/observation schemas
    WS   /ws     – WebSocket for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with:\n    uv sync\n"
    ) from e

try:
    from ..models import EcoLLMAction, EcoLLMObservation
    from .deployment_environment import EcoLLMEnvironment
except ModuleNotFoundError:
    from models import EcoLLMAction, EcoLLMObservation
    from server.deployment_environment import EcoLLMEnvironment


app = create_app(
    EcoLLMEnvironment,
    EcoLLMAction,
    EcoLLMObservation,
    env_name="eco-llm-routing-env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
