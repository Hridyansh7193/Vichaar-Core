from fastapi import FastAPI
from core.env import Env

app = FastAPI()
env = Env()

@app.post("/reset")
async def reset():
    obs = env.reset()
    return {"observation": obs}

@app.post("/step")
async def step(action: dict):
    # Depending on how the environment parses actions, 
    # action might be wrapped (e.g. {"action": "some_action"})
    act_str = action.get("action", "") if isinstance(action, dict) else str(action)
    obs, reward, done, info = env.step(act_str)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
async def state():
    return env.state()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
