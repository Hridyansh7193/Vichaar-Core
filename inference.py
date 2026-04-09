import os
import asyncio
from openai import OpenAI

# Required initialization for Phase 2 strict checks
try:
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )
except Exception:
    client = None

from core.env import Env as VichaarEnv
from decision.policy import Policy
from agents import make_agents

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

async def run_inference_async():
    task_name = "vichaar-core"
    benchmark = "openenv"

    rewards = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}", flush=True)

    try:
        # Initialize internal Vichaar-Core natively
        env = VichaarEnv()
        agents = make_agents()
        policy = Policy(agents)
        
        state = env.reset()

        done = False
        step_num = 1

        while not done and step_num <= 50:
            try:
                # Agent policy decides action (this automatically makes API proxy requests internally via agents/base.py)
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

            print(
                f"[STEP] step={step_num} action={act_str} reward={reward:.2f} done={str(done).lower()} error={error}",
                flush=True
            )

            if done:
                break

            step_num += 1

        # SCORE
        if rewards:
            score = sum(rewards) / len(rewards)
        else:
            score = 0.0

        score = max(0.0, min(1.0, score))
        success = score > 0.1

    except Exception as e:
        score = 0.0
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True
    )

def run_inference():
    asyncio.run(run_inference_async())

if __name__ == "__main__":
    run_inference()