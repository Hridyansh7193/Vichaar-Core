"""Risk Agent -- balances all metrics for long-term stability."""
from agents.base import Agent
from configs.agent_config import AGENT_DEFS

def create() -> Agent:
    return Agent("risk", AGENT_DEFS["risk"])
