"""Coordinator -- Global Planner with 1-step lookahead simulation."""
from typing import Dict, List, Tuple
from configs.env_config import ACTIONS, ACTION_EFFECTS, GLOBAL_SCORE_WEIGHTS, UNSAFE_ACTIONS
from configs.agent_config import METRIC_LABELS


def compute_global_score(metrics: Dict[str, float]) -> float:
    """Compute a single global quality score from current metrics."""
    return sum(
        GLOBAL_SCORE_WEIGHTS.get(k, 0.0) * metrics.get(k, 0.0)
        for k in GLOBAL_SCORE_WEIGHTS
    )


class Coordinator:
    """Ranks every action by simulating its effect on GLOBAL_SCORE."""

    @staticmethod
    def rank_actions(
        metrics: Dict[str, float],
        safe_mode: bool = False,
    ) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        for act in ACTIONS:
            if safe_mode and act in UNSAFE_ACTIONS:
                continue
            simulated = dict(metrics)
            for m, delta in ACTION_EFFECTS.get(act, {}).items():
                simulated[m] = max(0.0, min(1.0, simulated.get(m, 0.0) + delta))
            score = compute_global_score(simulated)
            results.append((act, round(score, 4)))
        results.sort(key=lambda x: -x[1])
        return results

    @staticmethod
    def best_action(
        metrics: Dict[str, float],
        safe_mode: bool = False,
    ) -> Tuple[str, float, str]:
        """Return (action, projected_score, explanation)."""
        ranked = Coordinator.rank_actions(metrics, safe_mode=safe_mode)
        best_act, best_score = ranked[0]

        effects = ACTION_EFFECTS.get(best_act, {})
        impact = {}
        for m, delta in effects.items():
            w = GLOBAL_SCORE_WEIGHTS.get(m, 0.0)
            impact[m] = delta * w
        if impact:
            top_metric = max(impact, key=lambda k: abs(impact[k]))
            label = METRIC_LABELS.get(top_metric, top_metric)
            reason = f"Lookahead: {best_act} best for global score ({label} impact {impact[top_metric]:+.3f})"
        else:
            reason = f"Lookahead: {best_act} optimal"

        return best_act, best_score, reason
