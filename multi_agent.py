import json
import asyncio
import logging
from typing import Dict, Any, List
from openai import AsyncOpenAI
import config

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Initialize client if API is given
try:
    client = AsyncOpenAI(
        api_key=config.OPENAI_API_KEY or "dummy",
        base_url=config.OPENAI_API_BASE
    )
except Exception:
    client = None

# Step 3 action space
ACTIONS = ["increase_production", "delay_launch", "invest_in_safety", "launch_fast", "reduce_cost"]

def build_prompt(role: str, state: Dict[str, Any]) -> str:
    return f"""You are the {role}.
Goal: Analyze the state and choose EXACTLY ONE action that aligns with your specific domain.
Return your answer ONLY as a JSON object: {{"action": "<exact_choice>"}}

Valid Actions: {", ".join(ACTIONS)}

Environment State:
{json.dumps(state, indent=2)}
"""

async def call_llm_agent(role: str, state: Dict[str, Any], fallback_action: str) -> str:
    if not client or config.OPENAI_API_KEY is None:
        # Fallback to local heuristic simulation if no API key is provided
        return fallback_action
        
    prompt = build_prompt(role, state)
    try:
        response = await client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        action = data.get("action", "")
        if action in ACTIONS:
            return action
    except Exception as e:
        logger.warning(f"Agent {role} failed to parse or reach LLM. Error: {e}")
        
    return fallback_action

# Step 4: Agents taking state -> one action
async def profit_agent(state: Dict[str, Any]) -> str:
    m = state.get("metrics", {})
    fallback = "increase_production" if m.get("expected_profit", 0) < 0.8 else "launch_fast"
    return await call_llm_agent("Profit Agent (Maximize revenue/expected_profit, risk-tolerant)", state, fallback)

async def pr_agent(state: Dict[str, Any]) -> str:
    m = state.get("metrics", {})
    fallback = "delay_launch" if m.get("public_sentiment", 0) < 0.6 else "invest_in_safety"
    return await call_llm_agent("PR Agent (Maximize public_sentiment, avoid bad press)", state, fallback)

async def ethics_agent(state: Dict[str, Any]) -> str:
    m = state.get("metrics", {})
    fallback = "invest_in_safety" if m.get("env_impact", 0) > 0.4 else "reduce_cost"
    return await call_llm_agent("Ethics Agent (Minimize env_impact and seek fairness)", state, fallback)

async def legal_agent(state: Dict[str, Any]) -> str:
    m = state.get("metrics", {})
    fallback = "invest_in_safety" if m.get("legal_risk", 0) > 0.3 else "reduce_cost"
    return await call_llm_agent("Legal Agent (Minimize legal_risk securely)", state, fallback)

async def risk_agent(state: Dict[str, Any]) -> str:
    m = state.get("metrics", {})
    if m.get("legal_risk", 0) > 0.6 or m.get("env_impact", 0) > 0.6:
        fallback = "delay_launch"
    elif m.get("cost", 0) > 0.7:
        fallback = "reduce_cost"
    else:
        fallback = "invest_in_safety"
    return await call_llm_agent("Risk Agent (Balance all downsides globally)", state, fallback)

# Step 5: Aggregation combining via voting
def aggregate_actions(actions: List[str]) -> str:
    """Combine agent actions using voting. Returns ONE final action."""
    counts = {}
    for a in actions:
        counts[a] = counts.get(a, 0) + 1
        
    # Sort by frequency descending. Break ties deterministically via alphabetical sorting
    best_action = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return best_action

async def get_multi_agent_action(state: Dict[str, Any]) -> str:
    # Triggering the 5 agents simultaneously
    results = await asyncio.gather(
        profit_agent(state),
        pr_agent(state),
        ethics_agent(state),
        legal_agent(state),
        risk_agent(state)
    )
    final_action = aggregate_actions(list(results))
    return final_action
