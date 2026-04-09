from fastapi import FastAPI, Body
from typing import Dict, Any

from core.env import Env as VichaarEnv
from decision.policy import Policy
from agents import make_agents
from configs.env_config import ACTIONS

app = FastAPI(title="Vichaar-Core OpenEnv Server")

# -------------------------------
# Initialize system
# -------------------------------
env = VichaarEnv()
agents = make_agents()
policy = Policy(agents)


# -------------------------------
# ROOT (IMPORTANT FOR HF)
# -------------------------------
@app.get("/")
def root():
    return {"message": "Vichaar-Core API running"}


# -------------------------------
# RESET
# -------------------------------
@app.post("/reset")
def reset(payload: Dict[str, Any] = Body(default={})):
    task_id = payload.get("task_id", "easy")
    obs = env.reset(task_id=task_id)
    return {"observation": obs}


# -------------------------------
# STEP (FINAL SAFE VERSION)
# -------------------------------
@app.post("/step")
async def step(action: Dict[str, Any] = Body(default={})):
    """
    OpenEnv-compatible step endpoint.
    Always ensures valid action.
    Never crashes.
    """

    # -------------------------------
    # 1. Safe extraction
    # -------------------------------
    act_str = ""

    if isinstance(action, dict):
        act_str = action.get("action", "")
    else:
        act_str = str(action)

    act_str = str(act_str).strip()

    # -------------------------------
    # 2. Validate action
    # -------------------------------
    is_valid = act_str in ACTIONS and act_str != ""

    if not is_valid:
        state = env.state()

        try:
            act_str, board, votes = await policy.run_step(state)

            # 🔥 double safety (VERY IMPORTANT)
            if act_str not in ACTIONS or not act_str:
                act_str = "invest_in_safety"
                board = []
                votes = {}

        except Exception:
            # 🔥 NEVER crash
            act_str = "invest_in_safety"
            board = []
            votes = {}

        obs, reward, done, info = env.step(
            act_str,
            messages=board,
            agent_votes=votes
        )

    else:
        # -------------------------------
        # 3. Valid action
        # -------------------------------
        obs, reward, done, info = env.step(act_str)

    # -------------------------------
    # 4. Response (STRICT FORMAT)
    # -------------------------------
    info["action"] = act_str
    return {
        "observation": obs,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }


# -------------------------------
# STATE
# -------------------------------
@app.get("/state")
def get_state():
    return env.state()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()