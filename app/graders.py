"""
app/graders.py
==============
Terminal reward grader for the Disaster Triage RL Environment.

Called once per episode when the agent invokes finalize().
Scores performance across three weighted axes:

  Prioritization Accuracy  (50%) — did critical zones get priority?
  Allocation Efficiency    (30%) — how close to actual demand?
  Resource Utilization     (20%) — were resources used without waste?
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from app.models import ResourceBundle, ResourceType, Zone


# ---------------------------------------------------------------------------
# Grader constants
# ---------------------------------------------------------------------------

W_PRIORITIZATION: float = 0.50
W_EFFICIENCY:     float = 0.30
W_UTILIZATION:    float = 0.20

# Severity thresholds
HIGH_SEVERITY_THRESHOLD: int   = 4
LOW_SEVERITY_THRESHOLD:  int   = 2

# Critical zones must have received this fraction of their demand
HIGH_SEVERITY_MIN_FRACTION: float = 0.40
# Low-priority zones — allocating up to this fraction is acceptable
LOW_SEVERITY_MAX_FRACTION:  float = 0.40


# ---------------------------------------------------------------------------
# Safe score guard — guarantees score is NEVER exactly 0.0 or 1.0
# ---------------------------------------------------------------------------

def safe_score(score: float) -> float:
    """Ensure score is strictly within (0, 1) — never 0.0 or 1.0."""
    # Round first to avoid floating point noise, then clip to keep it strictly in (0, 1)
    # Using 0.0001 to 0.9999 range as per hackathon requirements
    clamped = float(np.clip(round(float(score), 4), 0.0001, 0.9999))
    return clamped


# ---------------------------------------------------------------------------
# DisasterTriageGrader
# ---------------------------------------------------------------------------

class DisasterTriageGrader:
    """
    Stateless grader — instantiate once and call compute_final_score()
    at the end of every episode.
    """

    def compute_final_score(
        self,
        zones:               List[Zone],
        initial_resources:   ResourceBundle,
        available_resources: ResourceBundle,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the terminal reward for an episode.

        Parameters
        ----------
        zones               : all zones at episode end (with true values intact)
        initial_resources   : resources at episode start
        available_resources : resources remaining at episode end

        Returns
        -------
        final_score : float in [0, 1]
        breakdown   : dict with per-axis scores and the weighted total
        """
        prioritization = self._prioritization_score(zones)
        efficiency     = self._allocation_efficiency(zones)
        utilization    = self._resource_utilization(
            initial_resources, available_resources, zones
        )

        final_score = (
            W_PRIORITIZATION * prioritization
            + W_EFFICIENCY   * efficiency
            + W_UTILIZATION  * utilization
        )

        # STRICT COMPLIANCE: pass through safe_score to guarantee
        # the result is NEVER exactly 0.0 or 1.0.
        clamped_score = safe_score(final_score)

        breakdown: Dict[str, float] = {
            "prioritization": round(prioritization, 4),
            "efficiency":     round(efficiency, 4),
            "utilization":    round(utilization, 4),
            "final_score":    clamped_score,
        }

        return clamped_score, breakdown

    def _prioritization_score(self, zones: List[Zone]) -> float:
        if not zones: return 0.0
        scores: List[float] = []; weights: List[float] = []
        for zone in zones:
            demand_total = zone.true_demand.total(); alloc_total = zone.allocated.total()
            weight = 2.0 if zone.true_severity >= HIGH_SEVERITY_THRESHOLD else 1.0
            weights.append(weight)
            if demand_total <= 0.0:
                scores.append(1.0); continue
            fraction = alloc_total / demand_total
            if zone.true_severity >= HIGH_SEVERITY_THRESHOLD:
                score = 1.0 if fraction >= HIGH_SEVERITY_MIN_FRACTION else (fraction / HIGH_SEVERITY_MIN_FRACTION) * 0.8
            elif zone.true_severity <= LOW_SEVERITY_THRESHOLD:
                score = 1.0 if fraction <= LOW_SEVERITY_MAX_FRACTION else max(0.0, 1.0 - (fraction - LOW_SEVERITY_MAX_FRACTION) * 1.0)
            else:
                score = max(0.0, 1.0 - abs(fraction - 0.50) * 1.0)
            is_highest = (zone.true_severity == max(z.true_severity for z in zones))
            if is_highest and zone.revealed:
                score = min(1.0, score + 0.05)
            scores.append(float(np.clip(score, 0.0, 1.0)))
        return float(np.average(scores, weights=weights))

    def _allocation_efficiency(self, zones: List[Zone]) -> float:
        if not zones: return 0.0
        errors: List[float] = []
        for zone in zones:
            for resource in ResourceType:
                demand = getattr(zone.true_demand, resource.value); alloc = getattr(zone.allocated, resource.value)
                if demand > 0.0: errors.append(min(abs(alloc - demand) / demand, 1.0))
                elif alloc > 0.0: errors.append(1.0)
        if not errors: return 1.0
        mae = float(np.mean(errors))
        efficiency = 1.0 - (mae ** 0.7)
        over_allocation_penalty = 0.0
        for zone in zones:
            if zone.true_demand.total() > 0:
                if (zone.allocated.total() / zone.true_demand.total()) > 1.10:
                    over_allocation_penalty += 0.05
        return float(np.clip(efficiency - over_allocation_penalty, 0.0, 1.0))

    def _resource_utilization(self, initial, remaining, zones) -> float:
        initial_total = initial.total()
        if initial_total <= 0.0: return 1.0
        productive = 0.0; total_demand = 0.0
        for zone in zones:
            for resource in ResourceType:
                demand = getattr(zone.true_demand, resource.value); alloc = getattr(zone.allocated, resource.value)
                productive += min(alloc, max(demand, 0.0)); total_demand += max(demand, 0.0)
        if total_demand <= 0.0:
            deployed = initial_total - remaining.clamp_positive().total()
            return 1.0 if deployed == 0.0 else 0.0
        denominator = min(initial_total, total_demand)
        return float(np.clip(productive / denominator, 0.0, 1.0))

    def explain(self, zones, initial, available) -> Dict[str, object]:
        _, breakdown = self.compute_final_score(zones, initial, available)
        zone_details = []
        for zone in zones:
            demand_total = zone.true_demand.total(); alloc_total = zone.allocated.total()
            zone_details.append({
                "zone_id": zone.zone_id, "severity": zone.true_severity, "revealed": zone.revealed,
                "urgency_signal": round(zone.urgency_signal, 4), "true_demand": zone.true_demand.to_dict(),
                "allocated": zone.allocated.to_dict(), "alloc_fraction": round(alloc_total / demand_total if demand_total > 0 else 0.0, 4),
            })
        return {"score_breakdown": breakdown, "zone_details": zone_details,
                "weights": {"prioritization": W_PRIORITIZATION, "efficiency": W_EFFICIENCY, "utilization": W_UTILIZATION}}
