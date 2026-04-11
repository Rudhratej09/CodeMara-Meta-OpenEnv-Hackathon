import uvicorn
from fastapi import HTTPException
from openenv.core.env_server.http_server import create_app
from pydantic import BaseModel
from typing import List

from server.env import EcoLLMInferenceRoutingEnvironment
from server.models import RLAction, RLObservation, Strategy, ModelChoice
from server.tasks import TASKS

# 1. Create native OpenEnv app
app = create_app(
    EcoLLMInferenceRoutingEnvironment,
    RLAction,
    RLObservation,
    env_name="eco_llm_inference_routing",
    max_concurrent_envs=4,
)

# 2. Grading Metadata - UPDATED BOUNDS
_GRADER_BOUNDS = {
    "task_1": {"worst": 0.0, "best": 1.0},
    "task_2": {"worst": 0.5, "best": 2.5},
    "task_3": {"worst": 1.0, "best": 5.0},
}

class GraderAction(BaseModel):
    strategy: str
    model_choice: str
    exit_flag: bool = False

class GraderRequest(BaseModel):
    task_id: str
    actions: List[GraderAction]

# 3. Add required extra endpoints to the OpenEnv app
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def list_tasks():
    return {tid: {"task_id": tid, "bounds": _GRADER_BOUNDS.get(tid)} for tid in TASKS.keys()}

@app.post("/grader")
def grade_episode(request: GraderRequest):
    # Validate task_id
    if request.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {request.task_id}")
    
    env = EcoLLMInferenceRoutingEnvironment()
    obs = env.reset(task_id=request.task_id)
    total_reward = 0.0
    
    for a in request.actions:
        if obs.done: 
            break
        try:
            action = RLAction(
                strategy=Strategy(a.strategy), 
                model_choice=ModelChoice(a.model_choice), 
                exit_flag=a.exit_flag
            )
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")
        
        obs = env.step(action)
        total_reward += obs.reward_details.total_reward
    
    bounds = _GRADER_BOUNDS[request.task_id]
    score = (total_reward - bounds["worst"]) / (bounds["best"] - bounds["worst"])
    
    # Normalize to (0.0, 1.0) - NO CLAMPING to 0.01-0.99
    score = max(0.0, min(1.0, score))
    
    return {
        "task_id": request.task_id, 
        "score": score,
        "reward": total_reward
    }

def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Run the development server."""
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
