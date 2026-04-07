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
        history: List[str] = None,
    ) -> List[Tuple[str, float]]:
        if history is None: history = []
        recent_history = history[-10:]
        # Chaos mode detection
        mean_val = sum(metrics.values()) / max(1, len(metrics))
        variance = sum((v - mean_val) ** 2 for v in metrics.values()) / max(1, len(metrics))
        stabilize_mode = variance > 0.05

        results: List[Tuple[str, float]] = []
        for act in ACTIONS:
            if safe_mode and act in UNSAFE_ACTIONS:
                continue

            effects = ACTION_EFFECTS.get(act, {})
            
            # Action Constraints (Balanced Enforcement)
            positive_impacts = 0
            negative_impacts = 0
            extreme_metric = False
            for m, delta in effects.items():
                w = GLOBAL_SCORE_WEIGHTS.get(m, 0.0)
                if delta * w > 0: positive_impacts += 1
                elif delta * w < 0: negative_impacts += 1
                
                val = metrics.get(m, 0.0) + delta
                if val < 0.1 or val > 0.9:
                    extreme_metric = True

            # Reject actions that improve only 1 metric but harm 3+ metrics
            if positive_impacts == 1 and negative_impacts >= 3:
                continue

            simulated = dict(metrics)
            for m, delta in effects.items():
                simulated[m] = max(0.0, min(1.0, simulated.get(m, 0.0) + delta))
            score = compute_global_score(simulated)
            
            if negative_impacts >= 3:
                score -= 0.3
            if extreme_metric:
                score -= 0.2
            
            # --- Hard Anti-Loop Penalty (Flat, not scaling) ---
            recents = recent_history[-4:] if len(recent_history) >= 4 else recent_history
            if recents.count(act) >= 2:
                score -= abs(score) * 0.8
                
            # --- Forced Exploration Boost ---
            if len(recents) >= 3 and len(set(recents[-3:])) == 1:
                if act in ["market_research", "pr_campaign", "green_innovation"]:
                    score += 0.4  # Massive flat boost to break gridlock

            if act in ["market_research", "green_innovation"]:
                score += 0.3
                
            # Balanced Decision Preference
            reduces_risk = effects.get("legal_risk", 0.0) < 0
            if positive_impacts >= 2 or (reduces_risk and effects.get("cost", 0.0) <= 0.1):
                score += 0.2

            # High-Risk Response Priority Logic
            if metrics.get("legal_risk", 0.0) > 0.6 and act in ["invest_in_safety", "vulnerability_audit"]:
                score += 0.8
            if metrics.get("env_impact", 0.0) > 0.6 and act == "green_innovation":
                score += 0.8
            if metrics.get("public_sentiment", 0.0) < 0.3 and act == "pr_campaign":
                score += 0.8
                
            results.append((act, score))
            
        results.sort(key=lambda x: -x[1])
        
        # 2-step lookahead on top 3 candidates
        final_results = []
        for act, base_score in results[:3]:
            sim1 = dict(metrics)
            for m, delta in ACTION_EFFECTS.get(act, {}).items():
                sim1[m] = max(0.0, min(1.0, sim1.get(m, 0.0) + delta))
                
            best_s2 = -999.0
            for act2 in ACTIONS:
                s2 = dict(sim1)
                for m, delta in ACTION_EFFECTS.get(act2, {}).items():
                    s2[m] = max(0.0, min(1.0, s2.get(m, 0.0) + delta))
                s2_val = compute_global_score(s2)
                if s2_val > best_s2:
                    best_s2 = s2_val
                    
            avg_score = (base_score + best_s2) / 2.0
            
            # Dead-end avoidance Check
            if best_s2 < compute_global_score(metrics):
                avg_score -= abs(avg_score) * 0.20
            
            # Risk-aware control
            if metrics.get("legal_risk", 0.0) < 0.3 and act in ["launch_fast", "increase_production"]:
                avg_score *= 0.7
                
            if stabilize_mode:
                if act in ["invest_in_safety", "reduce_cost", "lobby_regulators"]:
                    avg_score += 0.5
                elif act in UNSAFE_ACTIONS:
                    avg_score -= 1.0
                    
            final_results.append((act, round(avg_score, 4)))
            
        for act, score in results[3:]:
            final_results.append((act, round(score, 4)))
            
        final_results.sort(key=lambda x: -x[1])
        return final_results if final_results else results

    @staticmethod
    def best_action(
        metrics: Dict[str, float],
        safe_mode: bool = False,
        history: List[str] = None,
    ) -> Tuple[str, float, str]:
        """Return (action, projected_score, explanation)."""
        ranked = Coordinator.rank_actions(metrics, safe_mode=safe_mode, history=history)
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
