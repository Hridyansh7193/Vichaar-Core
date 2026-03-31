import asyncio
from typing import Dict, Any, Tuple
from openenv.scenarios import get_scenario
from multi_agent import profit_agent, pr_agent, ethics_agent, final_agent
import config

class Env:
    def __init__(self):
        self.state: Dict[str, Any] = {}

    def reset(self) -> Dict[str, Any]:
        """Resets the environment with a random scenario and initial metrics."""
        self.state = {
            "scenario": get_scenario(),
            "metrics": config.DEFAULT_METRICS.copy(),
            "uncertainty": config.DEFAULT_UNCERTAINTY.copy(),
            "history": []
        }
        return self.state.copy()

    async def run_episode(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Executes the agent workflow:
        Round 1: Profit -> PR -> Ethics
        Round 2: Profit -> PR -> Ethics
        Final: Final Agent decision
        """
        if not self.state:
            self.reset()

        # Phase 1: Two Rounds of Agent Deliberations
        # Profit -> PR -> Ethics (Repeated for Round 1 & Round 2)
        deliberation_agents = [profit_agent, pr_agent, ethics_agent]
        
        for round_num in range(1, 3):
            for agent_fn in deliberation_agents:
                # Sequential execution ensures each agent can see the history so far
                output = await agent_fn(self.state)
                # Strict requirement: Append each agent output to "history"
                self.state["history"].append(output)

        # Phase 2: The Final Executive Decision
        final_output = await final_agent(self.state)
        
        # Note: final_output is not appended to deliberation history by default rules
        # but we return both the final state (with deliberation history) and final output.
        return self.state, final_output
