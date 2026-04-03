"""PR Agent -- maximizes public sentiment and brand image."""
from agents.base import Agent
from configs.agent_config import AGENT_DEFS

def create() -> Agent:
    return Agent("pr", AGENT_DEFS["pr"])
