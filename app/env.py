"""
app/env.py
==========
Core OpenEnv-compliant Disaster Triage RL Environment.

Implements a POMDP where an agent dispatches limited resources across
N partially observable disaster zones, under a step budget.

Interface (gym-style):
  env = DisasterTriageEnv(difficulty=Difficulty.MEDIUM, seed=42)
  obs = env.reset()
  obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.graders import DisasterTriageGrader
from app.models import (
    Action,
    ActionType,
    Difficulty,
    DifficultyConfig,
    DIFFICULTY_CONFIGS,
    EpisodeResult,
    Observation,
    ResourceBundle,
    ResourceType,
    Zone,
)


# ---------------------------------------------------------------------------
# Step-wise cost constants
# ---------------------------------------------------------------------------

INFO_COST:               float = 0.02    # internal cost per request_info
ALLOCATION_COST:         float = 0.03    # internal cost per allocate_resource
TEMPORAL_DECAY:          float = 0.01    # internal cost every step
EFFICIENCY_BONUS_RATE:   float = 0.02    # per unused step at finalize
INVALID_ACTION_PENALTY:  float = 0.10    # internal penalty for bad actions

# Dense step-level reward tiers
# Tier 1: Weak but valid          →  0.01 – 0.02
# Tier 2: Good context-aware      →  0.03 – 0.05
# Tier 3: Optimal severity match  →  0.06 – 0.10
STEP_REWARD_REPEAT_INFO:      float = 0.005  # repeat request_info on revealed zone
STEP_REWARD_NEW_INFO:         float = 0.02   # new zone revealed
STEP_REWARD_NEW_INFO_HIGH:    float = 0.04   # revealed a high-severity zone (4-5)
STEP_REWARD_WEAK_ALLOC:       float = 0.01   # valid alloc to low-severity zone
STEP_REWARD_GOOD_ALLOC:       float = 0.03   # valid alloc to medium-severity zone
STEP_REWARD_OPTIMAL_ALLOC:    float = 0.06   # alloc to high-severity with correct resource
STEP_REWARD_PERFECT_ALLOC:    float = 0.10   # alloc to highest-severity with best resource match
STEP_REWARD_WASTE_PENALTY:    float = 0.005  # reduced reward for over-allocation
STEP_REWARD_DIVERSITY_BONUS:  float = 0.02   # bonus for allocating to a new zone
MAX_STEP_REWARD:              float = 0.10   # cap per step

# Difficulty scaling factors (subtle, prevents Hard > Easy)
DIFFICULTY_SCALE = {
    "easy":   1.00,   # no penalty
    "medium": 0.97,   # ~3% scaling
    "hard":   0.92,   # ~8% scaling
}


# ---------------------------------------------------------------------------
# DisasterTriageEnv
# ---------------------------------------------------------------------------

class DisasterTriageEnv:
    """
    OpenEnv-compliant Disaster Triage environment.

    Parameters
    ----------
    difficulty : Difficulty
        EASY | MEDIUM | HARD — controls num_zones, observability, and budget.
    seed : int, optional
        Random seed for reproducible episodes.
    """

    metadata = {
        "render_modes": ["human", "json"],
        "name": "DisasterTriage-v1",
        "version": "1.0.0",
    }

    def __init__(
        self,
        difficulty: Difficulty = Difficulty.MEDIUM,
        seed: Optional[int] = None,
    ) -> None:
        self.difficulty: Difficulty         = difficulty
        self.config:     DifficultyConfig   = DIFFICULTY_CONFIGS[difficulty]
        self.grader:     DisasterTriageGrader = DisasterTriageGrader()
        self.rng:        np.random.Generator = np.random.default_rng(seed)

        # Episode state — all populated on reset()
        self.zones:                List[Zone]          = []
        self.available_resources:  ResourceBundle      = ResourceBundle()
        self._initial_resources:   ResourceBundle      = ResourceBundle()
        self.step_count:           int                 = 0
        self.accumulated_costs:    float               = 0.0
        self.step_quality_sum:     float               = 0.0   # sum of step rewards
        self.step_quality_count:   int                 = 0     # number of scored steps
        self._zones_allocated_to:  set                 = set() # track zone diversity
        self.done:                 bool                = False
        self.episode_result:       Optional[EpisodeResult] = None

    # ==========================================================================
    # Public OpenEnv interface
    # ==========================================================================

    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Reset the environment to a fresh episode.

        Parameters
        ----------
        seed : int, optional — overrides the seed set at construction.

        Returns
        -------
        obs : Observation
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.zones               = self._generate_zones()
        self.available_resources = ResourceBundle(
            food=self.config.initial_resources.food,
            water=self.config.initial_resources.water,
            medicine=self.config.initial_resources.medicine,
        )
        self._initial_resources  = ResourceBundle(
            food=self.config.initial_resources.food,
            water=self.config.initial_resources.water,
            medicine=self.config.initial_resources.medicine,
        )
        self.step_count         = 0
        self.accumulated_costs  = 0.0
        self.step_quality_sum   = 0.0
        self.step_quality_count = 0
        self._zones_allocated_to = set()
        self.done               = False
        self.episode_result     = None

        self._apply_initial_observability()
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Execute one action.

        Every step incurs temporal decay (-0.01).
        Positive reward only comes from finalize().

        Returns
        -------
        obs    : Observation
        reward : float  (negative on intermediate steps, ≥ 0 at finalize)
        done   : bool
        info   : dict
        """
        if self.done:
            raise RuntimeError(
                "Episode is over. Call reset() before calling step() again."
            )

        # --- validate --------------------------------------------------------
        valid, err = action.validate()
        if not valid:
            self.accumulated_costs += INVALID_ACTION_PENALTY
            info = {"valid": False, "error": err, "action": action.to_dict(), "step": self.step_count}
            return (
                self._build_observation(),
                0.0,
                False,
                info,
            )

        # --- dispatch --------------------------------------------------------
        base_cost = TEMPORAL_DECAY
        info: dict = {"valid": True, "action": action.to_dict()}

        if action.action_type == ActionType.FINALIZE:
            return self._handle_finalize(base_cost)

        elif action.action_type == ActionType.REQUEST_INFO:
            reward, info = self._handle_request_info(action, base_cost, info)

        elif action.action_type == ActionType.ALLOCATE_RESOURCE:
            reward, info = self._handle_allocate(action, base_cost, info)

        else:
            # Defensive: unreachable given validate()
            return (
                self._build_observation(),
                INVALID_ACTION_PENALTY,
                False,
                {"error": f"Unknown action type: {action.action_type}"},
            )

        # --- advance step counter -------------------------------------------
        self.step_count        += 1
        self.accumulated_costs += reward  # reward here is the positive cost component
        info["step"]            = self.step_count

        # --- compute dense step-level reward signal --------------------------
        step_reward = self._compute_step_reward(action, info)

        self.step_quality_sum   += step_reward
        self.step_quality_count += 1
        info["step_reward"] = round(step_reward, 2)

        # --- check step budget -----------------------------------------------
        if self.step_count >= self.config.max_steps:
            # Force finalize when the budget is exhausted
            return self._handle_finalize(step_cost=0.0)

        # Return the step_reward for intermediate steps
        return self._build_observation(), round(step_reward, 2), self.done, info

    def render(self, mode: str = "human") -> Optional[dict]:
        """Pretty-print or return the current observation as a dict."""
        obs_dict = self._build_observation().to_dict()
        if mode == "human":
            print(json.dumps(obs_dict, indent=2))
        return obs_dict

    def close(self) -> None:
        """No-op; included for OpenEnv compliance."""
        pass

    # ==========================================================================
    # Action handlers
    # ==========================================================================

    def _handle_request_info(
        self, action: Action, base_cost: float, info: dict
    ) -> Tuple[float, dict]:
        zone = self._get_zone(action.zone_id)

        if zone is None:
            info["error"] = f"Zone {action.zone_id} does not exist."
            return base_cost + INVALID_ACTION_PENALTY, info

        if zone.revealed:
            info["warning"] = (
                f"Zone {action.zone_id} was already revealed — cost wasted."
            )
        else:
            zone.reveal()
            info["revealed"] = {
                "zone_id":  zone.zone_id,
                "severity": zone.true_severity,
                "demand":   zone.true_demand.to_dict(),
            }

        reward         = TEMPORAL_DECAY + INFO_COST
        info["reward"] = 0.0
        return reward, info

    def _handle_allocate(
        self, action: Action, base_cost: float, info: dict
    ) -> Tuple[float, dict]:
        zone = self._get_zone(action.zone_id)

        if zone is None:
            info["error"] = f"Zone {action.zone_id} does not exist."
            return base_cost + INVALID_ACTION_PENALTY, info

        resource = action.resource_type
        requested = action.amount
        available  = getattr(self.available_resources, resource.value)

        # Clamp to what we actually have
        if requested > available:
            info["warning"] = (
                f"Requested {requested:.1f} {resource.value} but only "
                f"{available:.1f} available — clamped."
            )
            requested = available

        if requested <= 0.0:
            info["warning"] = f"No {resource.value} available to deploy."
            reward           = base_cost
            info["reward"]   = round(reward, 4)
            return reward, info

        # Apply allocation
        setattr(
            self.available_resources,
            resource.value,
            available - requested,
        )
        current_alloc = getattr(zone.allocated, resource.value)
        setattr(zone.allocated, resource.value, current_alloc + requested)

        info["allocated"] = {
            "zone_id":  zone.zone_id,
            "resource": resource.value,
            "amount":   round(requested, 4),
        }
        reward         = TEMPORAL_DECAY + ALLOCATION_COST
        info["reward"] = 0.0
        return reward, info

    def _handle_finalize(
        self, step_cost: float = 0.0
    ) -> Tuple[Observation, float, bool, dict]:
        """Compute terminal reward via the grader and close the episode."""
        self.done = True

        final_score, score_breakdown = self.grader.compute_final_score(
            zones=self.zones,
            initial_resources=self._initial_resources,
            available_resources=self.available_resources,
        )

        # --- Zero-allocation penalty ------------------------------------
        total_allocated = sum(
            z.allocated.total() for z in self.zones
        )
        if total_allocated == 0:
            final_score *= 0.3
            score_breakdown["zero_allocation_penalty"] = True
        else:
            score_breakdown["zero_allocation_penalty"] = False

        # --- Step quality score (0-1) -----------------------------------
        # Normalize against realistic expected reward per step (0.04),
        # not the theoretical max (0.10), to avoid crushing the score.
        EXPECTED_AVG_STEP_REWARD = 0.04
        if self.step_quality_count > 0:
            max_possible = self.step_quality_count * EXPECTED_AVG_STEP_REWARD
            step_quality = self.step_quality_sum / max_possible if max_possible > 0 else 0.0
        else:
            step_quality = 0.0
        step_quality = float(np.clip(step_quality, 0.0, 1.0))

        # --- Blended reward: 75% grader + 25% step quality ---------------
        blended_score = 0.75 * final_score + 0.25 * step_quality

        # --- Difficulty scaling ------------------------------------------
        diff_scale = DIFFICULTY_SCALE.get(self.difficulty.value, 1.0)
        scaled_score = blended_score * diff_scale

        # --- Final reward ------------------------------------------------
        total_reward = round(float(np.clip(scaled_score, 0.01, 0.99)), 2)

        score_breakdown["step_quality"] = round(step_quality, 4)
        score_breakdown["difficulty_scale"] = diff_scale
        score_breakdown["blended_score"] = round(blended_score, 4)

        self.episode_result = EpisodeResult(
            total_reward=total_reward,
            step_costs=self.accumulated_costs,
            final_score=final_score,
            efficiency_bonus=0.0,
            prioritization_score=score_breakdown["prioritization"],
            allocation_efficiency=score_breakdown["efficiency"],
            resource_utilization=score_breakdown["utilization"],
            steps_taken=self.step_count,
            data_completeness=self._data_completeness(),
        )

        info = {
            "final":          True,
            "episode_result": self.episode_result.to_dict(),
            "score_breakdown": score_breakdown,
        }

        return (
            self._build_observation(done=True),
            total_reward,
            True,
            info,
        )

    # ==========================================================================
    # Dense step reward computation
    # ==========================================================================

    def _compute_step_reward(self, action: Action, info: dict) -> float:
        """
        Compute a dense, context-aware step reward signal.

        Reward tiers:
          Tier 1 (Weak):    0.01 – 0.02  → valid but low-impact action
          Tier 2 (Good):    0.03 – 0.05  → context-aware allocation
          Tier 3 (Optimal): 0.06 – 0.10  → perfect severity-resource alignment

        Bonuses:
          +0.02 diversity bonus for first allocation to a new zone

        Penalties:
          -0.005 waste penalty for over-allocation (>100% of demand)
        """
        step_reward = 0.0

        # ── REQUEST_INFO ─────────────────────────────────────────────────
        if action.action_type == ActionType.REQUEST_INFO:
            if info.get("revealed"):
                zone = self._get_zone(action.zone_id)
                if zone and zone.true_severity >= 4:
                    step_reward = STEP_REWARD_NEW_INFO_HIGH   # 0.04 — discovered critical zone
                else:
                    step_reward = STEP_REWARD_NEW_INFO        # 0.02 — useful discovery
            else:
                step_reward = STEP_REWARD_REPEAT_INFO         # 0.005 — repeat, minimal signal

        # ── ALLOCATE_RESOURCE ────────────────────────────────────────────
        elif action.action_type == ActionType.ALLOCATE_RESOURCE:
            if info.get("allocated"):
                zone = self._get_zone(action.zone_id)
                if zone is None:
                    return 0.0

                severity = zone.true_severity
                resource = action.resource_type
                demand_val = getattr(zone.true_demand, resource.value, 0)
                alloc_val  = getattr(zone.allocated, resource.value, 0)

                # --- Base tier by severity ---
                if severity >= 5:
                    # Tier 3: highest severity → check resource alignment
                    if resource == ResourceType.MEDICINE:
                        step_reward = STEP_REWARD_PERFECT_ALLOC   # 0.10 — perfect match
                    elif resource == ResourceType.WATER:
                        step_reward = STEP_REWARD_OPTIMAL_ALLOC   # 0.06 — strong match
                    else:
                        step_reward = STEP_REWARD_GOOD_ALLOC      # 0.03 — acceptable
                elif severity >= 4:
                    # Tier 2-3: high severity
                    if resource == ResourceType.MEDICINE:
                        step_reward = STEP_REWARD_OPTIMAL_ALLOC   # 0.06
                    else:
                        step_reward = STEP_REWARD_GOOD_ALLOC      # 0.03
                elif severity >= 3:
                    # Tier 2: medium severity
                    step_reward = STEP_REWARD_GOOD_ALLOC          # 0.03
                else:
                    # Tier 1: low severity
                    step_reward = STEP_REWARD_WEAK_ALLOC          # 0.01

                # --- Diversity bonus: first time allocating to this zone ---
                if zone.zone_id not in self._zones_allocated_to:
                    step_reward += STEP_REWARD_DIVERSITY_BONUS    # +0.02
                    self._zones_allocated_to.add(zone.zone_id)

                # --- Waste penalty: over-allocated beyond demand ---
                if demand_val > 0 and alloc_val > demand_val * 1.1:
                    step_reward = max(0.0, step_reward - STEP_REWARD_WASTE_PENALTY)  # -0.005

        # Clamp to [0, MAX_STEP_REWARD]
        return round(float(np.clip(step_reward, 0.0, MAX_STEP_REWARD)), 2)


    # ==========================================================================
    # Zone generation
    # ==========================================================================

    def _generate_zones(self) -> List[Zone]:
        zones: List[Zone] = []
        for i in range(self.config.num_zones):
            severity = int(self.rng.integers(1, 6))  # 1–5 inclusive

            # Demand scales with severity; add stochasticity
            base = float(severity) * 10.0
            demand = ResourceBundle(
                food=float(self.rng.uniform(base * 0.5, base * 1.5)),
                water=float(self.rng.uniform(base * 0.5, base * 1.5)),
                medicine=float(self.rng.uniform(base * 0.3, base * 1.2)),
            )

            # Urgency signal: noisy proxy for true severity (clamped to [0, 1])
            noise    = float(self.rng.normal(0.0, self.config.noise_level))
            urgency  = float(np.clip((severity / 5.0) + noise, 0.0, 1.0))

            zones.append(
                Zone(
                    zone_id=f"Z{i}",          # string format: "Z0", "Z1", ...
                    true_severity=severity,
                    true_demand=demand,
                    urgency_signal=urgency,
                )
            )

        return zones

    def _apply_initial_observability(self) -> None:
        """Pre-reveal zones according to the difficulty tier."""
        obs_mode = self.config.observability

        if obs_mode == "full":
            for zone in self.zones:
                zone.reveal()

        elif obs_mode == "partial":
            # Reveal roughly half, chosen at random
            n_reveal = max(1, len(self.zones) // 2)
            indices  = self.rng.choice(len(self.zones), size=n_reveal, replace=False)
            for idx in indices:
                self.zones[int(idx)].reveal()

        elif obs_mode == "hidden":
            pass  # nothing revealed — agent must pay info costs

    # ==========================================================================
    # Observation builders & helpers
    # ==========================================================================

    def _build_observation(self, done: bool = False) -> Observation:
        return Observation(
            zones=self.zones,
            available_resources=ResourceBundle(
                food=self.available_resources.food,
                water=self.available_resources.water,
                medicine=self.available_resources.medicine,
            ),
            step_count=self.step_count,
            max_steps=self.config.max_steps,
            data_completeness=self._data_completeness(),
            done=done or self.done,
        )

    def _data_completeness(self) -> float:
        if not self.zones:
            return 0.0
        return sum(1.0 for z in self.zones if z.revealed) / len(self.zones)

    def _get_zone(self, zone_id: Optional[str]) -> Optional[Zone]:
        if zone_id is None:
            return None
        for z in self.zones:
            if z.zone_id == zone_id:
                return z
        return None

    # ==========================================================================
    # Space descriptors (for RL libraries that inspect env.observation_space)
    # ==========================================================================

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Flat tensor shape: 7 features per zone + 6 meta features."""
        return (self.config.num_zones * 7 + 6,)

    @property
    def num_actions(self) -> int:
        """
        Total discrete actions (upper bound):
          request_info  × num_zones
          allocate      × num_zones × 3 resources
          finalize      × 1
        """
        n = self.config.num_zones
        return n + n * len(ResourceType) + 1

    # ==========================================================================
    # Convenience factory
    # ==========================================================================

    @classmethod
    def make(cls, difficulty: str = "medium", seed: Optional[int] = None) -> DisasterTriageEnv:
        """
        Factory helper — creates AND resets the environment.

        Always calls reset() so zones and resources are populated immediately.

        Example
        -------
        env = DisasterTriageEnv.make("hard", seed=0)
        """
        instance = cls(difficulty=Difficulty(difficulty.lower()), seed=seed)
        instance.reset(seed=seed)   # ← populate zones + resources
        return instance

    # ==========================================================================
    # Full internal state (ground-truth, not masked)
    # ==========================================================================

    def get_full_state(self) -> dict:
        """
        Return the complete internal state of the environment including
        true severity and demand for every zone (unmasked).

        Used by the /state endpoint — NOT the agent-facing observation.

        Returns
        -------
        dict with keys:
          step_count, max_steps, done, available_resources, zones
        Each zone entry includes true_severity, true_demand, allocated,
        urgency_signal (float 0-1), revealed, known_severity, known_demand.
        """
        zones_state = []
        for zone in self.zones:          # explicitly iterate self.zones
            zones_state.append({
                "zone_id":        zone.zone_id,
                # Ground-truth (always present in /state)
                "true_severity":  zone.true_severity,
                "true_demand":    zone.true_demand.to_dict(),
                # Agent-visible fields
                "urgency_signal": round(zone.urgency_signal, 4),  # float [0, 1]
                "revealed":       zone.revealed,
                "known_severity": zone.known_severity if zone.revealed else -1,
                "known_demand":   (
                    zone.known_demand.to_dict()
                    if (zone.revealed and zone.known_demand is not None)
                    else None
                ),
                # Current allocation tally
                "allocated":      zone.allocated.to_dict(),
            })

        return {
            "step_count":          self.step_count,
            "max_steps":           self.config.max_steps,
            "done":                self.done,
            # available_resources is initialized in reset() and decremented on allocate
            "available_resources": self.available_resources.to_dict(),
            "zones":               zones_state,
        }
