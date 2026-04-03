"""Legal Agent -- minimizes legal risk and ensures compliance."""
from agents.base import Agent
from configs.agent_config import AGENT_DEFS

def create() -> Agent:
    return Agent("legal", AGENT_DEFS["legal"])
