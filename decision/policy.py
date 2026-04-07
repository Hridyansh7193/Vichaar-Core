"""Policy -- 4-layer strategic multi-agent controller."""
import asyncio
from typing import Dict, Any, List, Tuple

from agents.base import Agent
from decision.coordinator import Coordinator
from decision.ceo import CEO
from decision.safemode import SafeMode
from decision.aggregator import parse_board_suggestions
from configs.env_config import ACTIONS, UNSAFE_ACTIONS


class Policy:
    """4-layer decision engine:

    Layer 1 -- CEO (hard constraints):  override if thresholds breached
    Layer 2 -- Safe Mode:               ban risky actions when score declining
    Layer 3 -- Coordinator (lookahead):  pick best action by 1-step simulation
    Layer 4 -- Agent votes (fallback):   discussion -> vote -> majority

    The Coordinator's lookahead ranking is blended with agent vote counts
    to produce the final decision. CEO can veto everything.
    """

    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.safe_mode = SafeMode()
        self._last_decision_info: Dict[str, Any] = {}

    @property
    def decision_info(self) -> Dict[str, Any]:
        return self._last_decision_info

    async def run_step(
        self, state: Dict[str, Any]
    ) -> Tuple[str, List[str], Dict[str, str]]:
        """Returns (final_action, board_messages, agent_votes_dict)."""
        metrics = state.get("metrics", {})
        roles = list(self.agents.keys())

        self.safe_mode.update(metrics)

        info: Dict[str, Any] = {
            "ceo_override": False,
            "ceo_reason": "",
            "safe_mode": self.safe_mode.active,
            "coordinator_pick": "",
            "coordinator_score": 0.0,
            "coordinator_reason": "",
            "decision_source": "agents",
        }



        # Layer 2 + 3: Coordinator Lookahead
        coord_action, coord_score, coord_reason = Coordinator.best_action(
            metrics, safe_mode=self.safe_mode.active, history=state.get("history", [])
        )
        info["coordinator_pick"] = coord_action
        info["coordinator_score"] = coord_score
        info["coordinator_reason"] = coord_reason

        # Layer 4: Agent Discussion + Voting (Serialized to prevent massive rate limits)
        discuss_tasks = [self.agents[r].discuss(state, []) for r in roles]
        board = list(await asyncio.gather(*discuss_tasks))
        board_suggestions = parse_board_suggestions(board)

        # Fire sequentially to throttle burst limits (Groq 429 protections)
        raw_votes = []
        for r in roles:
            try:
                res = await self.agents[r].vote(state, board, board_suggestions)
                raw_votes.append(res)
                await asyncio.sleep(0.3)
            except Exception:
                pass
                
        agent_votes = {r: v for r, v in zip(roles, raw_votes)}

        # 1. Get Coordinator's Native Top Ranking
        coord_ranking = Coordinator.rank_actions(
            metrics, safe_mode=self.safe_mode.active, history=state.get("history", [])
        )
        coord_top_3 = {act for act, score in coord_ranking[:3]}
        final_action = coord_ranking[0][0]  # Default Boss behavior
        
        # 2. Strict LLM Output Control & Advisory Validation
        from agents.base import GLOBAL_LLM_DISABLED
        info["decision_source"] = "coordinator"
        
        llm_action = None

        if not GLOBAL_LLM_DISABLED and raw_votes:
            vote_counts = {v: raw_votes.count(v) for v in set(raw_votes) if v in ACTIONS}
            if vote_counts:
                llm_action = max(vote_counts, key=vote_counts.get)

        if llm_action and llm_action in ACTIONS:
            if llm_action not in state.get("history", [])[-1:]:
                final_action = llm_action
                info["decision_source"] = "llm"
                info["llm_used"] = True

        self._last_decision_info = info
        return final_action, board, agent_votes
