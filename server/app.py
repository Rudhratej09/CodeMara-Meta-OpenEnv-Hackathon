"""FastAPI app for the Eco-LLM Inference Routing environment."""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required for the web interface. Install dependencies before running the server."
    ) from exc

from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import RLAction, RLObservation


app = create_app(
    EcoLLMInferenceRoutingEnvironment,
    RLAction,
    RLObservation,
    env_name="eco_llm_inference_routing",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the development server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port == 8000:
        main()
    else:
        main(port=args.port)
