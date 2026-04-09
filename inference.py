import os
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy"
)

def run_inference():
    task_name = "vichaar-core"
    benchmark = "openenv"

    rewards = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}", flush=True)

    try:
        # RESET (SAFE)
        try:
            res = requests.post(f"{API_BASE_URL}/reset", timeout=15)
            res.raise_for_status()
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
            return

        done = False
        step_num = 1

        while not done and step_num <= 50:
            try:
                step_res = requests.post(
                    f"{API_BASE_URL}/step",
                    json={"action": ""},
                    timeout=30
                )
                step_res.raise_for_status()
                data = step_res.json()

                reward = float(data.get("reward", 0.0))
                done = bool(data.get("done", False))
                action = data.get("info", {}).get("action", "invest_in_safety")
                error = "null"

            except Exception as e:
                reward = 0.0
                done = True
                action = "error"
                error = "request_failed"

            rewards.append(reward)
            steps = step_num

            print(
                f"[STEP] step={step_num} action={action} reward={reward:.2f} done={str(done).lower()} error={error}",
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

    except Exception:
        score = 0.0
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True
    )

if __name__ == "__main__":
    run_inference()