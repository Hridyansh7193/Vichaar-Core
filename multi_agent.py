import os
import json
import asyncio
import logging
from typing import Dict, Any, Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# PYDANTIC VALIDATION MODELS
# ==========================================

class RiskAssessment(BaseModel):
    legal: Literal["low", "medium", "high"]
    reputation: Literal["low", "medium", "high"]
    environment: Literal["low", "medium", "high"]

class AgentDecision(BaseModel):
    agent: Literal["profit", "pr", "ethics", "fallback"]
    decision: Literal["launch", "modify", "reject"]
    justification: str
    risk_assessment: RiskAssessment
    confidence: float

class FinalTradeoffs(BaseModel):
    profit: str
    reputation: str
    ethics: str

class FinalDecision(BaseModel):
    final_decision: Literal["launch", "modify", "reject"]
    reasoning: str
    tradeoffs: FinalTradeoffs
    confidence: float

# ==========================================
# SYSTEM PROMPTS
# ==========================================

PROFIT_PROMPT = """You are the Profit Agent.
Goal: maximize profit, ignore ethics, pr, and long term implications. Be aggressive.
You must ONLY output valid JSON.
{
  "agent": "profit",
  "decision": "launch | modify | reject",
  "justification": "string",
  "risk_assessment": { "legal": "...", "reputation": "...", "environment": "..." },
  "confidence": float
}"""

PR_PROMPT = """You are the PR Agent.
Goal: protect brand, avoid backlash, ignore profit if needed.
You must ONLY output valid JSON.
{
  "agent": "pr",
  "decision": "launch | modify | reject",
  "justification": "string",
  "risk_assessment": { "legal": "...", "reputation": "...", "environment": "..." },
  "confidence": float
}"""

ETHICS_PROMPT = """You are the Ethics Agent.
Goal: focus on sustainability and fairness. Reject harmful actions.
You must ONLY output valid JSON.
{
  "agent": "ethics",
  "decision": "launch | modify | reject",
  "justification": "string",
  "risk_assessment": { "legal": "...", "reputation": "...", "environment": "..." },
  "confidence": float
}"""

FINAL_PROMPT = """You are the Final Decision-Making Agent.
Goal: balance all agents, ensure long term survival, resolve conflicts, prefer modify when possible.
You must ONLY output valid JSON.
{
  "final_decision": "launch | modify | reject",
  "reasoning": "string",
  "tradeoffs": { "profit": "...", "reputation": "...", "ethics": "..." },
  "confidence": float
}"""

# ==========================================
# CORE LLM INTEGRATION (ASYNC)
# ==========================================

# Initialize the Async OpenAI Client
# Note: Ensure you have OPENAI_API_KEY set in your environment!
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-sk-key-here"))

async def call_llm(prompt: str) -> str:
    """Async Wrapper for the LLM API using OpenAI's JSON mode."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",  # Using a fast model for agentic workflow
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,       # Low temp for structured adherence
        max_tokens=350,
        response_format={"type": "json_object"} # Guarantees valid JSON layout
    )
    return response.choices[0].message.content

def build_prompt(base_prompt: str, state: Dict[str, Any]) -> str:
    """Safely builds the final prompt payload."""
    return f"""{base_prompt}

Scenario:
{state.get('scenario', '')}

Metrics:
{json.dumps(state.get('metrics', {}), indent=2)}

Uncertainty:
{json.dumps(state.get('uncertainty', {}), indent=2)}

Previous Opinions:
{json.dumps(state.get('history', []), indent=2)}
"""

async def run_agent_workflow(base_prompt: str, state: Dict[str, Any], is_final: bool = False) -> Dict[str, Any]:
    """Shared function to format prompt, execute async call, and strictly validate data."""
    prompt = build_prompt(base_prompt, state)
    
    for attempt in range(2):
        try:
            response_text = await call_llm(prompt)
            data = json.loads(response_text)
            
            # Pydantic Enforcement
            if is_final:
                validated = FinalDecision(**data)
                return validated.model_dump()
            else:
                validated = AgentDecision(**data)
                return validated.model_dump()
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed extracting data: {str(e)}")
            if attempt == 1:
                logger.error("All attempts failed. Engaging strict fallback defaults.")
                break
                
    # Deterministic Schema Fallbacks
    if is_final:
        return FinalDecision(
            final_decision="modify",
            reasoning="Fallback safe decision triggered due to structural system failure or API outage.",
            tradeoffs=FinalTradeoffs(profit="neutral", reputation="neutral", ethics="neutral"),
            confidence=0.5
        ).model_dump()
    else:
        # Determine agent name for fallback object
        agent_name = "fallback"
        if "Profit" in base_prompt: agent_name = "profit"
        elif "PR" in base_prompt: agent_name = "pr"
        elif "Ethics" in base_prompt: agent_name = "ethics"
        
        return AgentDecision(
            agent=agent_name,
            decision="modify",
            justification="Fallback safe decision triggered due to API error.",
            risk_assessment=RiskAssessment(legal="medium", reputation="medium", environment="medium"),
            confidence=0.5
        ).model_dump()

# ==========================================
# BINDING FUNCTIONS
# ==========================================

async def profit_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    return await run_agent_workflow(PROFIT_PROMPT, state)

async def pr_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    return await run_agent_workflow(PR_PROMPT, state)

async def ethics_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    return await run_agent_workflow(ETHICS_PROMPT, state)

async def final_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    return await run_agent_workflow(FINAL_PROMPT, state, is_final=True)

# ==========================================
# PARALLEL EXECUTION PIPELINE
# ==========================================

async def orchestrate_decision(state: Dict[str, Any]):
    print("🚀 Triggering biased sub-agents in parallel...")
    
    # 🌟 Asyncio trick: We command all three separate agents to hit the LLM API 
    # simultaneously instead of waiting on each one in a row!
    results = await asyncio.gather(
        profit_agent(state),
        pr_agent(state),
        ethics_agent(state)
    )
    print("✅ Sub-agents complete. Responses collected.\n")
    
    # Pack their responses into history so the final agent can read them
    final_state = state.copy()
    final_state["history"] = results
    
    print("⚖️ Initiating Final Decision Engine Analysis...")
    final_result = await final_agent(final_state)
    
    print("\n[🎯 FINAL DECISION RECORD]")
    print(json.dumps(final_result, indent=2))
    return final_result


# ==========================================
# EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    dummy_state = {
        "scenario": "Launch a social media timeline that prioritizes enragement to boost engagement 45%, but carries unresolved European GDPR risks.",
        "metrics": {
            "cost": 50000,
            "expected_profit": 1500000,
            "legal_risk": 0.85,
            "env_impact": 0.05,
            "public_sentiment": -0.6
        },
        "uncertainty": {
            "legal_conf": 0.5,
            "env_conf": 0.95
        },
        "history": []
    }
    
    # Warning if testing without key attached
    if not os.getenv("OPENAI_API_KEY"):
        print("\n\n⚠️ WARNING: You must set your OPENAI_API_KEY environment variable to test this successfully! ⚠️\n")
        
    asyncio.run(orchestrate_decision(dummy_state))
