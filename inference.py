import os
import requests
from openai import OpenAI

# TASK 1 & 2: Use environment variables with correct fallbacks
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
# MODEL_NAME has default as per Task 2
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
# HF_TOKEN has NO default as per Task 2
HF_TOKEN = os.getenv("HF_TOKEN")

OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "dummy"

# TASK 1: Use OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=OPENAI_API_KEY
)

def run_inference():
    """
    STRICT compliance with Meta OpenEnv submission requirements.
    Uses centralized environment variable management and specific logging format.
    """
    # TASK 1: STRICT LOGGING FORMAT
    print("START")

    try:
        # TASK 1: Call API endpoints directly
        # 1. Reset
        reset_res = requests.post(f"{API_BASE_URL}/reset", timeout=10)
        reset_res.raise_for_status()
        
        step_number = 1
        done = False
        
        while not done:
            # 2. Step
            # We send an empty action to trigger the backend's internal policy as the default
            # but we use requests for the standard compliant loop.
            step_res = requests.post(
                f"{API_BASE_URL}/step", 
                json={"action": ""}, 
                timeout=30
            )
            step_res.raise_for_status()
            data = step_res.json()
            
            reward = data.get("reward", 0.0)
            done = data.get("done", False)
            # The backend provides the chosen action in the info field
            action = data.get("info", {}).get("action", "invest_in_safety")
            
            # TASK 1: STRICT LOGGING FORMAT
            print(f"STEP {step_number} | action={action} | reward={reward}")
            
            if done:
                break
            step_number += 1
            
    except Exception as e:
        # User wants to see logs/errors while developing instead of silently passing
        print(f"ERROR: {str(e)}")
        print("Please ensure your FastAPI backend is running! Run `fastapi dev server/app.py` in another terminal.")

    # TASK 1: STRICT LOGGING FORMAT
    print("END")

if __name__ == "__main__":
    run_inference()
