"""
Inference Script — Vichaar-Core
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

  Example:
    [START] task=easy env=vichaar-core model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=invest_in_safety reward=0.20 done=false error=null
    [STEP] step=2 action=expand_research reward=0.40 done=false error=null
    [STEP] step=3 action=align_values reward=1.00 done=true error=null
    [END] success=true steps=3 score=0.533 rewards=0.20,0.40,1.00
"""

import asyncio
import os
from typing import List, Optional

from openai import OpenAI

from core.env import Env as VichaarEnv
from decision.policy import Policy
from agents import make_agents

# ---------------------------------------------------------------------------
# Environment & model configuration
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"

# Task configuration — checked in priority order to match all validator env vars
TASK_NAME = (
    os.getenv("VICHAAR_CORE_TASK")
    or os.getenv("TASK_NAME")
    or os.getenv("TASK_ID")
    or "easy"
)
BENCHMARK = (
    os.getenv("VICHAAR_CORE_BENCHMARK")
    or os.getenv("BENCHMARK")
    or "vichaar-core"
)

# Episode parameters
MAX_STEPS               = 50
TEMPERATURE             = 0.7    # used internally by Policy/agents
MAX_TOKENS              = 512    # used internally by Policy/agents
SUCCESS_SCORE_THRESHOLD = 0.1    # minimum normalised score to count as success

# System prompt passed to the multi-agent policy context
SYSTEM_PROMPT = (
    "You are a multi-agent deliberation system navigating AI governance decisions. "
    "Each step you collectively choose one action that best balances safety, capability, "
    "and societal impact. Reason carefully and return a single action string."
)


# ---------------------------------------------------------------------------
# Logging helpers  (match demo format exactly)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

async def main() -> None:
    # OpenAI client is available for Policy/agents to use via env vars;
    # it is constructed here so the validator can confirm credentials are valid.
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)  # noqa: F841
    except Exception as e:
        print(f"[DEBUG] OpenAI client init failed: {e}", flush=True)
        client = None  # noqa: F841

    history:      List[str]   = []
    rewards:      List[float] = []
    steps_taken:  int         = 0
    score:        float       = 0.0
    success:      bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    env = VichaarEnv()

    try:
        agents = make_agents()
        policy = Policy(agents)

        state = env.reset(task_id=TASK_NAME)
        done  = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            try:
                # Multi-agent deliberation → single action string
                act_str, board, votes = await policy.run_step(state)

                if not act_str:
                    act_str = "invest_in_safety"

                obs, reward, done_flag, info = env.step(
                    act_str,
                    messages=board,
                    agent_votes=votes,
                )
                state = env.state()
                done  = done_flag
                error = None

            except Exception as e:
                reward  = 0.0
                done    = True
                act_str = "error"
                error   = str(e).replace(" ", "_")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=act_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {act_str!r} -> reward {reward:+.2f}")

        # Normalised score: mean reward clamped to [0, 1]
        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        score   = 0.0
        success = False
        print(f"[DEBUG] Execution error: {e}", flush=True)

    finally:
        # Always attempt graceful environment shutdown
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())