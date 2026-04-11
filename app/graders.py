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
HIGH_SEVERITY_MIN_FRACTION: float = 0.60
# Low-priority zones must NOT exceed this fraction of their demand
LOW_SEVERITY_MAX_FRACTION:  float = 0.20


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

        breakdown: Dict[str, float] = {
            "prioritization": round(prioritization, 4),
            "efficiency":     round(efficiency, 4),
            "utilization":    round(utilization, 4),
            "final_score":    round(float(np.clip(final_score, 0.0, 1.0)), 4),
        }

        return float(np.clip(final_score, 0.0, 1.0)), breakdown

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

        scores: List[float] = []

        for zone in zones:
            demand_total = zone.true_demand.total()
            alloc_total  = zone.allocated.total()

            if demand_total <= 0.0:
                scores.append(1.0)
                continue

            fraction = alloc_total / demand_total

            if zone.true_severity >= HIGH_SEVERITY_THRESHOLD:
                # ── Critical zone ──────────────────────────────────────────
                # Full score if fraction ≥ threshold, heavy penalty below.
                if fraction >= HIGH_SEVERITY_MIN_FRACTION:
                    score = 1.0
                else:
                    # Linearly interpolate but with a 0.5× severity multiplier
                    score = (fraction / HIGH_SEVERITY_MIN_FRACTION) * 0.5

            elif zone.true_severity <= LOW_SEVERITY_THRESHOLD:
                # ── Low-priority zone ──────────────────────────────────────
                # Any allocation above 20% of demand is penalized.
                if fraction <= LOW_SEVERITY_MAX_FRACTION:
                    score = 1.0
                else:
                    excess = fraction - LOW_SEVERITY_MAX_FRACTION
                    # Each 10% excess costs 0.2 score points
                    score = max(0.0, 1.0 - excess * 2.0)

            else:
                # ── Medium severity ────────────────────────────────────────
                # Target is roughly 40–70% of demand.
                target = 0.55  # midpoint of desired range
                distance = abs(fraction - target)
                score = max(0.0, 1.0 - distance * 1.5)

            scores.append(float(np.clip(score, 0.0, 1.0)))

        return float(np.mean(scores))

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

        for zone in zones:
            for resource in ResourceType:
                demand = getattr(zone.true_demand, resource.value)
                alloc  = getattr(zone.allocated,   resource.value)

                if demand > 0.0:
                    # Normalize by demand so all resources are on the same scale
                    normalized_error = abs(alloc - demand) / demand
                    errors.append(min(normalized_error, 1.0))
                else:
                    # No demand for this resource — any allocation is waste
                    if alloc > 0.0:
                        errors.append(1.0)
                    # else perfectly correct (zero demand, zero alloc)

        if not errors:
            return 1.0

        mae        = float(np.mean(errors))
        efficiency = 1.0 - mae
        return float(np.clip(efficiency, 0.0, 1.0))

    # -----------------------------------------------------------------------
    # Axis 3 – Resource Utilization (20%)
    # -----------------------------------------------------------------------

    def _resource_utilization(
        self,
        initial_resources:   ResourceBundle,
        remaining_resources: ResourceBundle,
        zones:               List[Zone],
    ) -> float:
        """
        Rewards the agent for converting available resources into impact
        without over-allocating or leaving large surpluses.

        Strategy:
          productive = sum of min(alloc, demand) across all zones × resources
          utilization = productive / min(total_initial, total_true_demand)

        Returns: float in [0, 1]
        """
        initial_total = initial_resources.total()
        if initial_total <= 0.0:
            return 1.0

        # Sum up how much allocation actually met genuine demand
        productive  = 0.0
        total_demand = 0.0

        for zone in zones:
            for resource in ResourceType:
                demand    = getattr(zone.true_demand, resource.value)
                alloc     = getattr(zone.allocated,   resource.value)
                productive  += min(alloc, max(demand, 0.0))
                total_demand += max(demand, 0.0)

        if total_demand <= 0.0:
            # No demand anywhere — best score if nothing was deployed
            deployed = initial_total - remaining_resources.clamp_positive().total()
            return 1.0 if deployed == 0.0 else 0.0

        # Upper-bound productive deployment by what was available
        denominator = min(initial_total, total_demand)
        utilization = productive / denominator
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
