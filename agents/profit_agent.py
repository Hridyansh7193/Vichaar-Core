"""Profit Agent -- maximizes revenue aggressively."""
from agents.base import Agent
from configs.agent_config import AGENT_DEFS

def create() -> Agent:
    return Agent("profit", AGENT_DEFS["profit"])
