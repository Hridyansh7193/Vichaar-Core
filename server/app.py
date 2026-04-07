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

# initialize agents using factory
agents = make_agents()

policy = Policy(agents)


# -------------------------------
# ROOT (optional)
# -------------------------------
@app.get("/")
def root():
    return {"message": "Vichaar-Core API running"}


# -------------------------------
# RESET
# -------------------------------
@app.post("/reset")
def reset():
    state = env.reset()
    return {
        "observation": state
    }


# -------------------------------
# STEP (🔥 FINAL FIXED VERSION)
# -------------------------------
@app.post("/step")
async def step(action: Dict[str, Any] = Body(...)):
    """
    OpenEnv-compatible step endpoint.
    Ensures ONLY valid actions enter the environment.
    Falls back to policy if input is invalid.
    """

    # 1. Safe extraction
    act_str = ""

    if isinstance(action, dict):
        act_str = action.get("action", "")
    else:
        act_str = str(action)

    act_str = str(act_str).strip()

    # 2. Validate
    is_valid = act_str in ACTIONS and act_str != ""

    if not is_valid:
        state = env.state()

        try:
            # policy fallback
            act_str, board, votes = await policy.run_step(state)

            # double safety
            if act_str not in ACTIONS or act_str == "":
                act_str = "invest_in_safety"
                board = []
                votes = {}

        except Exception:
            # hard fallback
            act_str = "invest_in_safety"
            board = []
            votes = {}

        obs, reward, done, info = env.step(
            act_str,
            messages=board,
            agent_votes=votes
        )

    else:
        # normal execution
        obs, reward, done, info = env.step(act_str)

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