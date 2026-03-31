import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

from reward import evaluate_decision

try:
    import env
except ImportError:

    env = None

app = FastAPI(
    title="Evaluation & Simulation API",
    description="Demo-ready API for evaluating agentic decisions on Profit, Ethics, PR, and Legal fronts."
)

class ActionRequest(BaseModel):
    action: Dict[str, Any]

class RunResponse(BaseModel):
    interaction_history: List[Dict[str, Any]]
    final_decision: Dict[str, Any]
    reward_score: float

@app.post("/reset")
def reset_environment():
    if not env:
        raise HTTPException(status_code=500, detail="env module not found. Ensure env.py exists in the project.")
    try:
        state = env.reset()
        return {"status": "success", "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step_environment(req: ActionRequest):
    if not env:
        raise HTTPException(status_code=500, detail="env module not found.")
    try:
        result = env.step(req.action)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def get_state():
    if not env:
        raise HTTPException(status_code=500, detail="env module not found.")
    try:
        # Tries env.get_state() or env.state based on typical Python setups
        state = env.get_state() if hasattr(env, "get_state") else getattr(env, "state", {})
        return {"status": "success", "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MockAgent:
    def __init__(self, role: str):
        self.role = role
        
    def act(self, current_state: Any) -> Dict[str, Any]:
        """Returns a mock decision prioritizing the agent's role."""
        return {
            "agent_role": self.role,
            "proposed_action": f"{self.role} optimized strategy",
            "justification": f"Taking action to maximize {self.role} metrics."
        }

@app.post("/run", response_model=RunResponse)
def run_simulation():
    if not env:
        raise HTTPException(status_code=500, detail="env module not found. Cannot run simulation.")
        
    try:
        current_state = env.reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Environment reset failed: {e}")

    history = []
    
    round_agents = [MockAgent("Profit"), MockAgent("PR"), MockAgent("Ethics")]
    
    for round_num in range(1, 3):
        for agent in round_agents:
            action = agent.act(current_state)
            
            try:
                step_result = env.step(action)

                if isinstance(step_result, tuple):
                    current_state = step_result[0]
                else:
                    current_state = step_result
            except Exception as e:
                 raise HTTPException(status_code=500, detail=f"Environment step failed with action {action}: {e}")

            history.append({
                "round": round_num,
                "agent": agent.role,
                "action": action,
                "current_state": current_state
            })

    final_agent = MockAgent("Final Action Evaluator")
    final_decision = final_agent.act(current_state)
    
    history.append({
        "round": "Final",
        "agent": final_agent.role,
        "action": final_decision,
        "current_state": current_state
    })

    if isinstance(current_state, dict) and "profit" in current_state:
        state_metrics = current_state  
    else:

        state_metrics = {
             "profit": 0.8,
             "legal_risk": 0.1,
             "pr_risk": 0.2,
             "env_risk": 0.3
        }

    reward = evaluate_decision(
        final_decision=final_decision,
        state_metrics=state_metrics,
        agent_outputs=history
    )

    return RunResponse(
        interaction_history=history,
        final_decision=final_decision,
        reward_score=reward
    )
