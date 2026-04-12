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

W_PRIORITIZATION: float = 0.35
W_EFFICIENCY:     float = 0.40
W_UTILIZATION:    float = 0.25

# Severity thresholds
HIGH_SEVERITY_THRESHOLD: int   = 4
LOW_SEVERITY_THRESHOLD:  int   = 2

# Critical zones must have received this fraction of their demand
HIGH_SEVERITY_MIN_FRACTION: float = 0.80
# Low-priority zones — allocating up to this fraction is acceptable
LOW_SEVERITY_MAX_FRACTION:  float = 0.15


# ---------------------------------------------------------------------------
# Safe score guard — guarantees score is NEVER exactly 0.0 or 1.0
# ---------------------------------------------------------------------------

def safe_score(score: float) -> float:
    """Ensure score is strictly within (0, 1) — never 0.0 or 1.0."""
    if score >= 1.0:
        score = 0.999
    elif score <= 0.0:
        score = 0.001
    return round(score, 4)   # 4dp prevents rounding to 1.0000


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

    # -----------------------------------------------------------------------
    # Axis 1 – Prioritization Accuracy (50%)
    # -----------------------------------------------------------------------

    def _prioritization_score(self, zones: List[Zone]) -> float:
        """
        Checks whether the agent respected severity-based allocation rules:

          High severity (≥ 4) → must receive ≥ 60 % of true demand
          Low  severity (≤ 2) → must receive ≤ 20 % of true demand
          Medium severity     → smooth proportional score

        Returns: float in [0, 1]
        """
        if not zones:
            return 0.0

        scores:  List[float] = []
        weights: List[float] = []

        for zone in zones:
            demand_total = zone.true_demand.total()
            alloc_total  = zone.allocated.total()

        for zone in zones:
            demand_total = zone.true_demand.total()
            alloc_total  = zone.allocated.total()

            # Mathematical Definition: Zone Weight = 2 ^ severity
            weight = 2.0 ** zone.true_severity
            weights.append(weight)

            if demand_total <= 0.0:
                scores.append(1.0)
                continue

            # Fractional fulfillment score
            fraction = alloc_total / demand_total
            if zone.true_severity >= HIGH_SEVERITY_THRESHOLD:
                score = 1.0 if fraction >= HIGH_SEVERITY_MIN_FRACTION else (fraction / HIGH_SEVERITY_MIN_FRACTION)
            elif zone.true_severity <= LOW_SEVERITY_THRESHOLD:
                score = 1.0 if fraction <= LOW_SEVERITY_MAX_FRACTION else max(0.0, 1.0 - (fraction - LOW_SEVERITY_MAX_FRACTION))
            else:
                score = 1.0 - abs(fraction - 0.5)

            # Information Bonus
            if zone.revealed and zone.true_severity == max(z.true_severity for z in zones):
                score = min(1.0, score + 0.02)

            scores.append(float(np.clip(score, 0.0, 1.0)))

        return float(np.average(scores, weights=weights))

    # -----------------------------------------------------------------------
    # Axis 2 – Allocation Efficiency (30%)
    # -----------------------------------------------------------------------

    def _allocation_efficiency(self, zones: List[Zone]) -> float:
        """
        Measures how accurately allocations matched true demand.

        Uses inverse Mean Absolute Error (MAE) normalized per resource per zone.
        A perfect match → 0 MAE → efficiency = 1.0.

        Returns: float in [0, 1]
        """
        if not zones:
            return 0.0

        errors: List[float] = []

        total_useful_alloc = 0.0
        total_demand       = 0.0

        for zone in zones:
            for resource in ResourceType:
                demand = getattr(zone.true_demand, resource.value)
                alloc  = getattr(zone.allocated,   resource.value)
                total_useful_alloc += min(alloc, demand)
                total_demand       += demand

        if total_demand <= 0.0:
            return 1.0

        # Efficiency = Σ(min(allocated, demand)) / Σ(demand)
        efficiency = total_useful_alloc / total_demand

        # Efficiency Penalty: Subtract 0.05 for every zone that is significantly over-allocated (>110%)
        over_allocation_penalty = 0.0
        for zone in zones:
            if zone.true_demand.total() > 0:
                if (zone.allocated.total() / zone.true_demand.total()) > 1.10:
                    over_allocation_penalty += 0.10

        efficiency = max(0.0, efficiency - over_allocation_penalty)
        return float(np.clip(efficiency, 0.0, 1.0))

    # -----------------------------------------------------------------------
    # Axis 3 – Resource Utilization (20%)
    def _resource_utilization(
        self,
        initial_resources:   ResourceBundle,
        available_resources: ResourceBundle,
        zones:               List[Zone],
    ) -> float:
        total_allocated = sum(z.allocated.total() for z in zones)
        if total_allocated <= 0.0:
            return 0.0

        total_waste = 0.0
        for zone in zones:
            for resource in ResourceType:
                demand = getattr(zone.true_demand, resource.value)
                alloc  = getattr(zone.allocated,   resource.value)
                # Waste = Σ(max(allocated - demand, 0))
                total_waste += max(alloc - demand, 0.0)

        # Utilization = 1 - (Waste / Total Allocated)
        utilization = 1.0 - (total_waste / total_allocated)
        return float(np.clip(utilization, 0.0, 1.0))

    # -----------------------------------------------------------------------
    # Diagnostic helpers
    # -----------------------------------------------------------------------

    def explain(
        self,
        zones:               List[Zone],
        initial_resources:   ResourceBundle,
        available_resources: ResourceBundle,
    ) -> Dict[str, object]:
        """
        Return a human-readable breakdown of every zone's contribution
        to the final score, useful for debugging agents.
        """
        _, breakdown = self.compute_final_score(
            zones, initial_resources, available_resources
        )

        zone_details = []
        for zone in zones:
            demand_total = zone.true_demand.total()
            alloc_total  = zone.allocated.total()
            fraction     = alloc_total / demand_total if demand_total > 0 else 0.0

            zone_details.append({
                "zone_id":        zone.zone_id,          # e.g. "Z0", "Z1", ...
                "severity":       zone.true_severity,
                "revealed":       zone.revealed,
                "urgency_signal": round(zone.urgency_signal, 4),  # float 0–1
                "true_demand":    zone.true_demand.to_dict(),
                "allocated":      zone.allocated.to_dict(),
                "alloc_fraction": round(fraction, 4),
            })

        return {
            "score_breakdown": breakdown,
            "zone_details":    zone_details,
            "weights": {
                "prioritization": W_PRIORITIZATION,
                "efficiency":     W_EFFICIENCY,
                "utilization":    W_UTILIZATION,
            },
        }
