import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# We import the core engine from the script we just wrote
from multi_agent import orchestrate_decision, FinalDecision

# Initialize the FastAPI App
app = FastAPI(
    title="Multi-Agent Decision Engine API", 
    description="A concurrent LLM decision making backend.",
    version="1.0"
)

# ==========================================
# STRICT INPUT SCHEMAS
# ==========================================
# FastAPI will automatically validate requests against these structures.
# If a user forgets "cost", it will throw a beautiful 422 error before even hitting the LLM.

class MetricsInput(BaseModel):
    cost: float
    expected_profit: float
    legal_risk: float
    env_impact: float
    public_sentiment: float

class UncertaintyInput(BaseModel):
    legal_conf: float
    env_conf: float

class DecisionRequest(BaseModel):
    scenario: str
    metrics: MetricsInput
    uncertainty: UncertaintyInput
    # History starts empty, but we include it the schema for extensibility
    history: Optional[List[Dict[str, Any]]] = []

# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/api/v1/evaluate", response_model=FinalDecision)
async def evaluate_scenario(request: DecisionRequest):
    """
    Takes in a business scenario and strictly typed metrics.
    Spins up Profit, PR, and Ethics agents concurrently, then combines
    their views to output a balanced and reasoned Final Decision.
    """
    try:
        # Convert the strictly typed Pydantic object back into a flat dictionary
        # because our multi_agent.py expects a payload state of Dict[str, Any]
        state_payload = request.model_dump()
        
        # We await the async orchestrator. While this runs, FastAPI can efficiently 
        # handle other incoming HTTP requests!
        final_output = await orchestrate_decision(state_payload)
        
        # FastAPI will automatically validate that final_output matches FinalDecision
        return final_output
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Processing Error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Multi-Agent System Online."}

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Starting FastAPI Sub-Agent Backend...")
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8000, reload=True)
