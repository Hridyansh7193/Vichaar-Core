import uvicorn
from fastapi import FastAPI
from env import Env
from grader import compute_reward

app = FastAPI(
    title="Multi-Agent Decision Pipeline",
    description="Full agent deliberation workflow with reward grading.",
    version="1.0"
)

# Global environment instance
env_instance = Env()

@app.post("/run")
async def run():
    """
    Triggers a full simulation using the stepped Env API:
    1. Resets the state (picks a scenario)
    2. Runs 2 deliberation rounds
    3. Runs the final decision
    4. Returns the full integrated result
    """
    # 1. Reset
    obs = env_instance.reset()
    
    # 2. Deliberation Rounds
    for _ in range(2):
        await env_instance.step({"action_type": "step"})
    
    # 3. Final Decision and Scoring
    # Step automatically calculates final reward based on task_id
    response = await env_instance.step({"action_type": "finalize"})
    
    # Get the final state for the response record
    state = env_instance.state()

    return {
        "scenario": state["scenario"],
        "history": state["history"],
        "final_decision": response.info.get("final_decision"),
        "reward": response.reward
    }

@app.get("/health")
def health():
    return {"status": "ok", "message": "Pipeline active."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
