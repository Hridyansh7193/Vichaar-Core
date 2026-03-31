from typing import Tuple, Optional
from .models import State, AgentAction, Metrics, Uncertainty
from .scenarios import get_scenario

class OpenEnv:
    AGENTS = ["Profit", "PR", "Ethics"]
    FINAL_AGENT = "Final Decision Agent"

    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds
        self.state: State = State()
        self.done: bool = False
        self.current_round: int = 0
        self.current_agent_index: int = 0

    def reset(self, scenario_id: Optional[str] = None) -> Tuple[State, dict]:
        scenario_text = get_scenario(scenario_id)
        self.state = State(scenario=scenario_text)
        self.done = False
        self.current_round = 0
        self.current_agent_index = 0
        info = {
            "message": "Environment reset. Ready for the first agent.",
            "next_expected_agent": self.get_expected_agent()
        }
        return self.state.model_copy(deep=True), info

    def get_expected_agent(self) -> str:
        if self.done:
            return "None (Episode Finished)"
        if self.current_round < self.max_rounds:
            return self.AGENTS[self.current_agent_index]
        return self.FINAL_AGENT

    def step(self, action: AgentAction) -> Tuple[State, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode has finished. Call reset() to start a new episode.")

        expected_agent = self.get_expected_agent()
        
        if action.agent_name != expected_agent:
            raise ValueError(
                f"Turn order violation! Expected action from '{expected_agent}', "
                f"but got action from '{action.agent_name}'."
            )

        self.state.history.append(action)

        if action.metrics_update:
            self.state.metrics.cost += action.metrics_update.cost
            self.state.metrics.expected_profit += action.metrics_update.expected_profit
            self.state.metrics.legal_risk += action.metrics_update.legal_risk
            self.state.metrics.env_impact += action.metrics_update.env_impact
            self.state.metrics.public_sentiment += action.metrics_update.public_sentiment

        if action.uncertainty_update:
            self.state.uncertainty.confidence_scores.update(
                action.uncertainty_update.confidence_scores
            )

        if expected_agent == self.FINAL_AGENT:
            self.done = True
        else:
            self.current_agent_index += 1
            if self.current_agent_index >= len(self.AGENTS):
                self.current_agent_index = 0
                self.current_round += 1

        reward = self._calculate_reward() if self.done else 0.0

        info = {
            "current_round": self.current_round,
            "next_expected_agent": self.get_expected_agent()
        }

        return self.state.model_copy(deep=True), reward, self.done, info

    def _calculate_reward(self) -> float:
        m = self.state.metrics
        return m.expected_profit - m.cost - (m.legal_risk * 2.0) - (m.env_impact * 2.0) + (m.public_sentiment * 1.5)
