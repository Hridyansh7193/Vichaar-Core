"""
Inference Script — Vichaar-Core Evaluator
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    
- The inference script must be named `inference.py` and placed in the root directory.
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from core.env import Env as VichaarEnv
from configs.env_config import ACTIONS

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Check standard Vichaar-Core validation vars
TASK_NAME = os.getenv("VICHAAR_CORE_TASK", os.getenv("TASK_ID", "easy"))
BENCHMARK = os.getenv("VICHAAR_CORE_BENCHMARK", os.getenv("BENCHMARK", "vichaar-core"))
MAX_STEPS = 50
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.3  # As per Vichaar-Core benchmark

SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You are the CEO of a corporation facing high-stakes crises.
    You interact with a corporate boardroom of executive agents (Profit, Ethics, PR, Legal, Risk).
    Each turn you must evaluate the metrics and decide on an executive action.
    Valid actions are exactly one of the following:
    {', '.join(ACTIONS)}
    
    Reply with EXACTLY ONE action string from the list above. No quotes, no prefixes, no explanation.
    """
).strip()


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


def build_user_prompt(step: int, metrics: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    
    # Format metrics cleanly
    metrics_str = ", ".join(
        f"{k}={v:.2f}" if isinstance(v, (int, float)) 
        else f"{k}={v.get('value', 0.0):.2f}" if isinstance(v, dict) 
        else f"{k}={v}" 
        for k, v in metrics.items()
    )
    
    return textwrap.dedent(
        f"""
        Step: {step}
        Current Board Metrics: {metrics_str}
        Last reward: {last_reward:.2f}
        
        Recent actions and rewards:
        {history_block}
        
        Send your next corporate action.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, metrics: dict, last_reward: float, history: List[str]) -> str:
    if client is None:
        return "invest_in_safety"
        
    user_prompt = build_user_prompt(step, metrics, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Simple extraction heuristic to ensure valid action if model included formatting
        for act in ACTIONS:
            if act in text.lower():
                return act
                
        return text if text else "invest_in_safety"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "invest_in_safety"


async def main() -> None:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[DEBUG] OpenAI client init failed (fallback to random/default actions): {e}", flush=True)
        client = None

    env = VichaarEnv()
    
    # In Vichaar-Core, grader mapping is used from tasks
    from tasks.grader import (
        grade_easy, grade_medium, grade_hard, grade_adversarial, grade_chaotic
    )
    grader_map = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
        "adversarial": grade_adversarial,
        "chaotic": grade_chaotic
    }
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        state = env.reset(task_id=TASK_NAME)
        metrics = state.get("metrics", {})
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            
            # Use OpenAI model to generate next valid action
            message = get_model_message(client, step, metrics, last_reward, history)

            obs, reward, done, info = env.step(message)
            
            error = None
            rewards.append(float(reward))
            steps_taken = step
            metrics = env.state().get("metrics", {})
            last_action_executed = info.get("action", message)
            last_reward = float(reward)

            log_step(step=step, action=last_action_executed, reward=last_reward, done=bool(done), error=error)

            history.append(f"Step {step}: {last_action_executed!r} -> reward {last_reward:+.2f}")

            if done:
                break

        # Compute final grade with the respective grader
        grader_func = grader_map.get(TASK_NAME, grade_easy)
        score = grader_func(env.state(), task_id=TASK_NAME)
        
        score = min(max(float(score), 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        success = False
        score = 0.0
        print(f"[DEBUG] Execution error: {e}", flush=True)
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())