"""Ethics Agent -- minimizes environmental impact and social harm."""
from agents.base import Agent
from configs.agent_config import AGENT_DEFS

def create() -> Agent:
    return Agent("ethics", AGENT_DEFS["ethics"])
