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
    Triggers a full simulation:
    1. Resets the state (picks a scenario)
    2. Runs the agent workflow (Round 1 + Round 2 + Final)
    3. Grades the decision using the reward function
    4. Returns the full deliberation record
    """
    # Initialize the scenario and metrics
    state = env_instance.reset()
    
    # Run the full deliberative workflow (two rounds + final)
    # This must be awaited since the agents call OpenAI LLMs
    state, final_output = await env_instance.run_episode()
    
    # Evaluate the result using the grader
    reward = compute_reward(state, final_output)

    # Return the clean, flattened result required
    return {
        "scenario": state["scenario"],
        "history": state["history"],
        "final_decision": final_output,
        "reward": reward
    }

@app.get("/health")
def health():
    return {"status": "ok", "message": "Pipeline active."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
