"""
app/models.py
=============
Core data models for the Disaster Triage RL Environment.

Defines the fundamental data structures:
  - ResourceBundle   : food / water / medicine triple
  - Zone             : a disaster zone with hidden/revealed attributes
  - Action           : one of request_info | allocate_resource | finalize
  - Observation      : what the agent sees each step
  - DifficultyConfig : per-tier env parameters
  - EpisodeResult    : final episode summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResourceType(str, Enum):
    FOOD     = "food"
    WATER    = "water"
    MEDICINE = "medicine"


class ActionType(str, Enum):
    REQUEST_INFO       = "request_info"
    ALLOCATE_RESOURCE  = "allocate_resource"
    FINALIZE           = "finalize"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# ResourceBundle
# ---------------------------------------------------------------------------

@dataclass
class ResourceBundle:
    """Holds counts of the three resource types."""
    food:     float = 0.0
    water:    float = 0.0
    medicine: float = 0.0

    # --- arithmetic helpers ------------------------------------------------

    def __add__(self, other: ResourceBundle) -> ResourceBundle:
        return ResourceBundle(
            food=self.food + other.food,
            water=self.water + other.water,
            medicine=self.medicine + other.medicine,
        )

    def __sub__(self, other: ResourceBundle) -> ResourceBundle:
        return ResourceBundle(
            food=self.food - other.food,
            water=self.water - other.water,
            medicine=self.medicine - other.medicine,
        )

    def __mul__(self, scalar: float) -> ResourceBundle:
        return ResourceBundle(
            food=self.food * scalar,
            water=self.water * scalar,
            medicine=self.medicine * scalar,
        )

    def can_afford(self, other: ResourceBundle) -> bool:
        return (
            self.food     >= other.food
            and self.water    >= other.water
            and self.medicine >= other.medicine
        )

    def clamp_positive(self) -> ResourceBundle:
        return ResourceBundle(
            food=max(0.0, self.food),
            water=max(0.0, self.water),
            medicine=max(0.0, self.medicine),
        )

    def total(self) -> float:
        return self.food + self.water + self.medicine

    # --- serialization helpers --------------------------------------------

    def to_array(self) -> np.ndarray:
        return np.array([self.food, self.water, self.medicine], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        return {
            "food":     round(self.food, 4),
            "water":    round(self.water, 4),
            "medicine": round(self.medicine, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResourceBundle:
        return cls(food=d["food"], water=d["water"], medicine=d["medicine"])


# ---------------------------------------------------------------------------
# Zone
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    """
    A single disaster zone.

    Hidden (ground truth) attributes are only accessible internally;
    the agent only sees them after calling request_info(zone_id).
    """
    zone_id:        str            # e.g. "Z0", "Z1", ...
    true_severity:  int            # 1–5, hidden until revealed
    true_demand:    ResourceBundle  # hidden until revealed
    urgency_signal: float          # noisy proxy for severity (0–1), always visible

    # Observability state
    revealed:       bool                     = False
    known_severity: int                      = -1    # -1 = unknown
    known_demand:   Optional[ResourceBundle] = None

    # Running allocation tally
    allocated: ResourceBundle = field(default_factory=ResourceBundle)

    # --- reveal ------------------------------------------------------------

    def reveal(self) -> None:
        """Mark this zone as revealed, exposing its true attributes."""
        self.revealed       = True
        self.known_severity = self.true_severity
        self.known_demand   = ResourceBundle(
            food=self.true_demand.food,
            water=self.true_demand.water,
            medicine=self.true_demand.medicine,
        )

    # --- serialization -----------------------------------------------------

    def _zone_index(self) -> int:
        """Return the numeric index embedded in the zone_id string (e.g. 'Z3' → 3)."""
        return int(self.zone_id[1:])

    def to_observation_vector(self) -> np.ndarray:
        """
        7-element float32 vector:
          [zone_index, revealed_flag, known_severity, urgency_signal,
           known_demand_food, known_demand_water, known_demand_medicine]

        Hidden values masked with -1.0.
        """
        if self.revealed and self.known_demand is not None:
            demand = self.known_demand.to_array()
            sev    = float(self.known_severity)
        else:
            demand = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
            sev    = -1.0

        return np.array(
            [float(self._zone_index()), float(self.revealed), sev, self.urgency_signal],
            dtype=np.float32,
        ), demand

    def to_flat_vector(self) -> np.ndarray:
        header, demand = self.to_observation_vector()
        return np.concatenate([header, demand])

    def to_dict(self) -> dict:
        return {
            "zone_id":        self.zone_id,
            "revealed":       self.revealed,
            "urgency_signal": round(self.urgency_signal, 4),
            "known_severity": self.known_severity if self.revealed else -1,
            "known_demand":   self.known_demand.to_dict()
                              if (self.revealed and self.known_demand) else None,
            "allocated":      self.allocated.to_dict(),
        }


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """
    Represents one agent action.

    Variants
    --------
    ActionType.REQUEST_INFO
        requires: zone_id

    ActionType.ALLOCATE_RESOURCE
        requires: zone_id, resource_type, amount (> 0)

    ActionType.FINALIZE
        no extra fields required
    """
    action_type:   ActionType
    zone_id:       Optional[str]          = None   # e.g. "Z0", "Z1", ...
    resource_type: Optional[ResourceType] = None
    amount:        Optional[float]        = None

    def validate(self) -> Tuple[bool, str]:
        if self.action_type == ActionType.REQUEST_INFO:
            if self.zone_id is None:
                return False, "request_info requires zone_id"

        elif self.action_type == ActionType.ALLOCATE_RESOURCE:
            if self.zone_id is None:
                return False, "allocate_resource requires zone_id"
            if self.resource_type is None:
                return False, "allocate_resource requires resource_type"
            if self.amount is None or self.amount <= 0:
                return False, "allocate_resource requires a positive amount"

        elif self.action_type == ActionType.FINALIZE:
            pass  # always valid

        else:
            return False, f"Unknown action_type: {self.action_type}"

        return True, ""

    def to_dict(self) -> dict:
        return {
            "action_type":   self.action_type.value,
            "zone_id":       self.zone_id,
            "resource_type": self.resource_type.value if self.resource_type else None,
            "amount":        self.amount,
        }

    @classmethod
    def request_info(cls, zone_id: str) -> Action:
        return cls(action_type=ActionType.REQUEST_INFO, zone_id=zone_id)

    @classmethod
    def allocate(cls, zone_id: str, resource_type: ResourceType, amount: float) -> Action:
        return cls(
            action_type=ActionType.ALLOCATE_RESOURCE,
            zone_id=zone_id,
            resource_type=resource_type,
            amount=amount,
        )

    @classmethod
    def finalize(cls) -> Action:
        return cls(action_type=ActionType.FINALIZE)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    What the agent receives at each timestep.

    Contains:
      - zones              : list of Zone objects (partially masked)
      - available_resources: current resource counts
      - step_count         : steps taken so far
      - max_steps          : episode horizon
      - data_completeness  : fraction of zones revealed (0.0–1.0)
      - done               : whether the episode has terminated
      - info               : auxiliary diagnostic dict
    """
    zones:               List[Zone]
    available_resources: ResourceBundle
    step_count:          int
    max_steps:           int
    data_completeness:   float
    done:                bool = False
    info:                dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "zones":               [z.to_dict() for z in self.zones],
            "available_resources": self.available_resources.to_dict(),
            "step_count":          self.step_count,
            "max_steps":           self.max_steps,
            "data_completeness":   round(self.data_completeness, 4),
            "done":                self.done,
            "info":                self.info,
        }

    def to_tensor(self) -> np.ndarray:
        """
        Flatten the entire observation into a 1-D float32 numpy array
        suitable for direct use with neural networks.

        Layout:
          [zone_0_flat (7), ..., zone_N_flat (7),
           food, water, medicine, step_count, max_steps, data_completeness]
        """
        zone_vecs = np.concatenate([z.to_flat_vector() for z in self.zones])
        meta = np.array(
            [
                self.available_resources.food,
                self.available_resources.water,
                self.available_resources.medicine,
                float(self.step_count),
                float(self.max_steps),
                self.data_completeness,
            ],
            dtype=np.float32,
        )
        return np.concatenate([zone_vecs, meta])


# ---------------------------------------------------------------------------
# Difficulty Configuration
# ---------------------------------------------------------------------------

@dataclass
class DifficultyConfig:
    difficulty:        Difficulty
    num_zones:         int
    max_steps:         int
    initial_resources: ResourceBundle
    observability:     str    # "full" | "partial" | "hidden"
    noise_level:       float  # std-dev for urgency signal noise


DIFFICULTY_CONFIGS: Dict[Difficulty, DifficultyConfig] = {
    Difficulty.EASY: DifficultyConfig(
        difficulty=Difficulty.EASY,
        num_zones=3,
        max_steps=30,
        initial_resources=ResourceBundle(food=100.0, water=100.0, medicine=100.0),
        observability="full",
        noise_level=0.05,
    ),
    Difficulty.MEDIUM: DifficultyConfig(
        difficulty=Difficulty.MEDIUM,
        num_zones=5,
        max_steps=50,
        initial_resources=ResourceBundle(food=120.0, water=120.0, medicine=120.0),
        observability="partial",
        noise_level=0.20,
    ),
    Difficulty.HARD: DifficultyConfig(
        difficulty=Difficulty.HARD,
        num_zones=7,
        max_steps=70,
        initial_resources=ResourceBundle(food=100.0, water=100.0, medicine=100.0),
        observability="hidden",
        noise_level=0.50,
    ),
}


# ---------------------------------------------------------------------------
# Episode Result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Summary of a completed episode. Returned in the finalize() info dict."""
    total_reward:          float
    step_costs:            float
    final_score:           float
    efficiency_bonus:      float
    prioritization_score:  float
    allocation_efficiency: float
    resource_utilization:  float
    steps_taken:           int
    data_completeness:     float

    def to_dict(self) -> dict:
        return {
            "total_reward":          round(self.total_reward, 4),
            "step_costs":            round(self.step_costs, 4),
            "final_score":           round(self.final_score, 4),
            "efficiency_bonus":      round(self.efficiency_bonus, 4),
            "prioritization_score":  round(self.prioritization_score, 4),
            "allocation_efficiency": round(self.allocation_efficiency, 4),
            "resource_utilization":  round(self.resource_utilization, 4),
            "steps_taken":           self.steps_taken,
            "data_completeness":     round(self.data_completeness, 4),
        }
