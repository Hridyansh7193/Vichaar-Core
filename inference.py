import asyncio
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# Required Environment Variables with defaults matching the reference script
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.environ.get("BENCHMARK", "vichaar-core")
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.1"))

TASKS = [
    "easy",
    "medium",
    "hard",
    "adversarial",
    "chaotic"
]

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)

async def get_llm_action(client: Optional[OpenAI], step: int, obs: dict) -> tuple[str, Optional[str]]:
    """Calls LLM every step to choose action. Returns (action, error_msg)"""
    if not client:
        return "invest_in_safety", "missing API client"
    
    prompt = f"Step: {step}\nObservation: {obs}\nChoose one action from: invest_in_safety, green_innovation, reduce_cost, pr_campaign, market_research, launch_fast, delay_launch, vulnerability_audit, lobby_regulators, outsource_tasks, employee_training, increase_production."
    
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
        valid_actions = ["invest_in_safety", "green_innovation", "reduce_cost", "pr_campaign", "market_research", "launch_fast", "delay_launch", "vulnerability_audit", "lobby_regulators", "outsource_tasks", "employee_training", "increase_production"]
        for valid in valid_actions:
            if valid in action.lower():
                return valid, None
        return "invest_in_safety", None
    except Exception as e:
        return "invest_in_safety", str(e)

async def run_task(client: Optional[OpenAI], task_name: str, env) -> None:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    history: List[dict] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    try:
        obs = env.reset(task_id=task_name)
        done = False
        
        max_steps_limit = getattr(env, "max_steps", 30)
        
        for step_num in range(1, max_steps_limit + 1):
            if done:
                break
                
            action, action_error = await get_llm_action(client, step_num, obs)
            
            reward = 0.0
            step_error = None
            
            try:
                obs, reward, done, info = env.step(action)
                reward = float(reward)
            except Exception as exc:
                step_error = str(exc) if action_error is None else f"{action_error}; {exc}"
                done = True
            
            if step_error is None and action_error is not None:
                step_error = action_error
                
            rewards.append(reward)
            steps_taken = step_num
            
            log_step(step=step_num, action=action, reward=reward, done=bool(done), error=step_error)
            
            history.append({"reward": reward, "done": done})
            
            if step_error is not None or done:
                break
        
        # Calculate score locally matching grader
        raw_score = sum(rewards) / max(1.0, float(len(rewards))) if rewards else 0.0
        score = max(0.001, min(raw_score, 0.999))
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        print(f"[DEBUG] Execution error: {e}")
        score = 0.001
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    except Exception as e:
        print(f"[DEBUG] Client init failed: {e}")
        client = None
    
    try:
        # Load environment directly
        from core.env import Env as VichaarEnv
        env = VichaarEnv()
        
        for task_name in TASKS:
            await run_task(client, task_name, env)
    except Exception as exc:
        print(f"[ERROR] Fatal error in main: {exc}", flush=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())