"""Agent factory — creates all 5 agents from config."""
from typing import Dict
from agents.base import Agent
from configs.agent_config import AGENT_DEFS


def make_agents() -> Dict[str, Agent]:
    return {role: Agent(role, defn) for role, defn in AGENT_DEFS.items()}
