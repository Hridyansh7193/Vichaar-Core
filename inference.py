import asyncio
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# Required Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME")
TASK_NAME = os.environ.get("TASK_ID", "easy")

# Initialize OpenAI client
try:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception as e:
    print(f"[DEBUG] Client init failed: {e}")
    client = None

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

async def get_llm_action(step: int, obs: dict) -> str:
    """Calls LLM every step to choose action."""
    if not client:
        return "invest_in_safety"
    
    prompt = f"Step: {step}\nObservation: {obs}\nChoose one action from: invest_in_safety, green_innovation, reduce_cost, pr_campaign, market_research, launch_fast, delay_launch, vulnerability_audit, align_values, expand_research."
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a CEO. Reply with ONLY the action name string."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        action = response.choices[0].message.content.strip()
        # Validation: ensure it's one of the valid actions
        valid_actions = ["invest_in_safety", "green_innovation", "reduce_cost", "pr_campaign", "market_research", "launch_fast", "delay_launch", "vulnerability_audit", "align_values", "expand_research"]
        for valid in valid_actions:
            if valid in action.lower():
                return valid
        return "invest_in_safety"
    except Exception as e:
        return "invest_in_safety"

async def main():
    # Detect Benchmark
    benchmark = "openenv"
    
    log_start(task=TASK_NAME, env=benchmark, model=MODEL_NAME)
    
    # We will use the VichaarEnv directly or hit the Space URL if running remotely
    # But usually inference script hits the local import to be fast
    from core.env import Env as VichaarEnv
    env = VichaarEnv()
    
    history: List[dict] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    try:
        obs = env.reset(task_id=TASK_NAME)
        done = False
        
        # Max steps depend on task, but we clamp for safety
        max_steps_limit = 20
        
        for step_num in range(1, max_steps_limit + 1):
            if done:
                break
                
            action = await get_llm_action(step_num, obs)
            
            # Perform step
            obs, reward, done, info = env.step(action)
            
            rewards.append(float(reward))
            steps_taken = step_num
            
            log_step(step=step_num, action=action, reward=float(reward), done=done, error=None)
            
            # OpenEnv Graders often look at the trajectory (list of step info)
            history.append({"reward": float(reward), "done": done})
            
            if done:
                break
        
        # Calculate score using the local graders logic to match what the platform will do
        if not rewards:
            score = 0.0
        else:
            score = sum(rewards) / len(rewards)
        
        score = max(0.0, min(1.0, float(score)))
        success = score > 0.1
        
    except Exception as e:
        print(f"[DEBUG] Execution error: {e}")
        score = 0.0
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())