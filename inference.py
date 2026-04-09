import os
import asyncio
from openai import OpenAI

# Required initialization for Phase 2 strict checks - exactly matching demo logic
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Initialize Client safely
try:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception:
    client = None

from core.env import Env as VichaarEnv
from decision.policy import Policy
from agents import make_agents

async def run_inference_async():
    # Detect Task & Benchmark
    task_id = os.getenv("TASK_ID", "easy")
    benchmark = "openenv"

    rewards = []
    steps = 0
    success = False

    # EXACT [START] format from demo
    print(f"[START] task={task_id} env={benchmark} model={MODEL_NAME}", flush=True)

    try:
        env = VichaarEnv()
        agents = make_agents()
        policy = Policy(agents)
        
        state = env.reset(task_id=task_id)

        done = False
        step_num = 1

        while not done and step_num <= 50:
            try:
                # Run multi-agent policy
                act_str, board, votes = await policy.run_step(state)
                
                if not act_str:
                    act_str = "invest_in_safety"

                obs, reward, done, info = env.step(act_str, messages=board, agent_votes=votes)
                state = env.state()
                error = "null"

            except Exception as e:
                reward = 0.0
                done = True
                act_str = "error"
                error = str(e).replace(' ', '_')

            rewards.append(reward)
            steps = step_num

            # EXACT [STEP] format from demo
            print(f"[STEP] step={step_num} action={act_str} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)

            if done:
                break
            step_num += 1

        # Final Score Calculation
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score > 0.1

    except Exception:
        score = 0.0
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # EXACT [END] format from demo (Removed task= field which was causing regex fail)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def run_inference():
    asyncio.run(run_inference_async())

if __name__ == "__main__":
    run_inference()