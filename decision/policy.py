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

        # Layer 1: CEO Override
        ceo_action, ceo_reason = CEO.check(metrics)
        if ceo_action:
            info["ceo_override"] = True
            info["ceo_reason"] = ceo_reason
            info["decision_source"] = "CEO"
            discuss_tasks = [self.agents[r].discuss(state, []) for r in roles]
            board = list(await asyncio.gather(*discuss_tasks))
            agent_votes = {r: ceo_action for r in roles}
            self._last_decision_info = info
            return ceo_action, board, agent_votes

        # Layer 2 + 3: Coordinator Lookahead
        coord_action, coord_score, coord_reason = Coordinator.best_action(
            metrics, safe_mode=self.safe_mode.active
        )
        info["coordinator_pick"] = coord_action
        info["coordinator_score"] = coord_score
        info["coordinator_reason"] = coord_reason

        # Layer 4: Agent Discussion + Voting
        discuss_tasks = [self.agents[r].discuss(state, []) for r in roles]
        board = list(await asyncio.gather(*discuss_tasks))
        board_suggestions = parse_board_suggestions(board)

        vote_tasks = [
            self.agents[r].vote(state, board, board_suggestions) for r in roles
        ]
        raw_votes = list(await asyncio.gather(*vote_tasks))
        agent_votes = {r: v for r, v in zip(roles, raw_votes)}

        # Blend: Coordinator + Agent Votes
        vote_counts: Dict[str, int] = {}
        for v in raw_votes:
            if v in ACTIONS:
                vote_counts[v] = vote_counts.get(v, 0) + 1

        coord_ranking = Coordinator.rank_actions(metrics, safe_mode=self.safe_mode.active)
        coord_top3 = {a for a, _ in coord_ranking[:3]}

        blend_scores: Dict[str, float] = {}
        for act in ACTIONS:
            s = float(vote_counts.get(act, 0))
            if act == coord_action:
                s += 2.0
            elif act in coord_top3:
                s += 1.0
            if self.safe_mode.active and act in UNSAFE_ACTIONS:
                s = -999.0
            blend_scores[act] = s

        final_action = max(blend_scores, key=blend_scores.get)

        if final_action == coord_action and vote_counts.get(coord_action, 0) == 0:
            info["decision_source"] = "coordinator"
        elif final_action == coord_action:
            info["decision_source"] = "coordinator+agents"
        else:
            info["decision_source"] = "agents"

        if self.safe_mode.active:
            info["decision_source"] += " [SAFE MODE]"

        self._last_decision_info = info
        return final_action, board, agent_votes
