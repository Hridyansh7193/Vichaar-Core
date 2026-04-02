import logging
import copy
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env import Env
from multi_agent import get_multi_agent_action

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Multi-Agent RL Simulation API")

# Global environment instance
global_env = Env()

# Models
class Metrics(BaseModel):
    expected_profit: float
    legal_risk: float
    env_impact: float
    public_sentiment: float
    cost: float

class Observation(BaseModel):
    scenario: str
    metrics: Metrics
    history: List[str]
    step_count: int

class Action(BaseModel):
    action: str

class ResetRequest(BaseModel):
    task_id: str = "easy"

class RewardResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class RunRequest(BaseModel):
    task_id: str = "easy"
    max_steps: int = 3

class RunSummary(BaseModel):
    total_steps: int
    final_reward: float
    performance: str

class RunResponse(BaseModel):
    history: List[Dict[str, Any]]
    final_state: Observation
    final_reward: float
    total_step_reward: float
    summary: RunSummary

@app.get("/")
async def root():
    return {"message": "API working"}

@app.post("/reset", response_model=Observation)
async def reset_env(req: ResetRequest):
    logger.info(f"Resetting environment with task_id: {req.task_id}")
    obs = global_env.reset(req.task_id)
    return copy.deepcopy(obs)

@app.post("/step", response_model=RewardResponse)
async def step_env():
    # Get Action from Agents Based on Current State
    current_state = global_env.state()
    if not current_state.get("scenario") or current_state.get("scenario") == "Uninitialized":
        # Auto-reset if not initialized
        current_state = global_env.reset("easy")
        logger.info("Environment auto-reset to 'easy' since it wasn't initialized.")
    
    action = await get_multi_agent_action(current_state)
    logger.info(f"Agents selected action: {action}")
    
    # Step environment
    obs, reward, done, info = global_env.step(action)
    logger.info(f"Reward updated: {reward:.2f}")
    
    safe_obs = copy.deepcopy(obs)
    return RewardResponse(
        observation=safe_obs,
        reward=reward,
        done=done,
        info=info
    )

@app.post("/run", response_model=RunResponse)
async def run_episode(req: RunRequest):
    logger.info(f"Running full episode for task_id: {req.task_id} with max_steps: {req.max_steps}")
    state = global_env.reset(req.task_id)
    
    history_log = []
    total_step_reward = 0.0
    
    from grader import grade_episode
    
    for step_num in range(req.max_steps):
        action = await get_multi_agent_action(state)
        # Avoid crashes safely
        obs, reward, done, info = global_env.step(action)
        
        # Safe dict access to metrics and history tracking
        metrics = obs.get("metrics", {})
        
        step_result = {
            "step": step_num + 1,
            "action": action,
            "reward": reward,
            "metrics": metrics.copy() if metrics else {}
        }
        history_log.append(step_result)
        
        logger.info(f"Step {step_num + 1}: Action [{action}] -> Step Reward: {reward:.2f}")
        
        total_step_reward += reward
        state = obs
        
        if done:
            break
            
    final_score = grade_episode(state, req.task_id)
    
    if final_score >= 0.7:
        perf = "good"
    elif final_score >= 0.4:
        perf = "average"
    else:
        perf = "poor"

    summary = {
        "total_steps": len(history_log),
        "final_reward": final_score,
        "performance": perf
    }
    
    # Ensure fully independent copy for the response
    safe_state = copy.deepcopy(global_env.state())
    
    return {
        "history": history_log,
        "final_state": safe_state,
        "final_reward": final_score,
        "total_step_reward": round(total_step_reward, 2),
        "summary": summary
    }
