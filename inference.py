import os
import asyncio
from typing import List, Optional
from openai import OpenAI

from core.env import Env as VichaarEnv
from decision.policy import Policy
from agents import make_agents

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Standardizing Variable Names exactly to demo structure
TASK_NAME = os.getenv("VICHAAR_CORE_TASK") or os.getenv("TASK_NAME") or os.getenv("TASK_ID") or "easy"
BENCHMARK = os.getenv("VICHAAR_CORE_BENCHMARK") or os.getenv("BENCHMARK") or "vichaar-core"
MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.1

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def main() -> None:
    # Initialize Client safely if you intend to execute real LLMs
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:
        client = None

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = VichaarEnv()
        agents = make_agents()
        policy = Policy(agents)
        
        state = env.reset(task_id=TASK_NAME)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            try:
                # Run multi-agent policy
                act_str, board, votes = await policy.run_step(state)
                
                if not act_str:
                    act_str = "invest_in_safety"

                obs, reward, done_flag, info = env.step(act_str, messages=board, agent_votes=votes)
                state = env.state()
                done = done_flag
                error = None

            except Exception as e:
                reward = 0.0
                done = True
                act_str = "error"
                error = str(e).replace(' ', '_')

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=act_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {act_str!r} -> reward {reward:+.2f}")

        # Final Score Calculation based on episodes
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        score = 0.0
        success = False
        print(f"[DEBUG] Execution error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())