# Disaster Triage Environment Specification

This document provides a technical overview of the observation and action spaces for the Disaster Triage Reinforcement Learning environment.

---

## 1. Observation Space

The environment is modeled as a **Partially Observable Markov Decision Process (POMDP)**.  
At each step, the agent receives an `Observation` object representing the current state of the disaster response system.

### Zone-Level Fields

Each zone contains the following attributes:

- **`zone_id`** *(string)*  
  Unique identifier for the zone (e.g., `"Z0"`, `"Z1"`).

- **`urgency_signal`** *(float ∈ [0.0, 1.0])*  
  A noisy but always-visible proxy for crisis severity.  
  Used for prioritization before ground-truth information is revealed.

- **`revealed`** *(boolean)*  
  Indicates whether ground-truth data has been obtained via `request_info`.

- **`known_severity`** *(int ∈ {1,2,3,4,5} or -1)*  
  The ground-truth severity level of the zone.  
  Visible only if `revealed = true`; otherwise `-1`.

- **`known_demand`** *(object or null)*  
  The specific resource requirements for the zone. Visible only if `revealed = true`.
  ```json
  {
    "food": float,
    "water": float,
    "medicine": float
  }
  ```

- **`allocated`** *(object)*  
  The total volume of resources successfully deployed to this zone so far.
  ```json
  {
    "food": float,
    "water": float,
    "medicine": float
  }
  ```

### Global State Fields

- **`available_resources`** *(object)*  
  The global stockpile remaining for deployment across all zones.
- **`step_count`** *(int)*  
  Number of actions taken in the current episode.
- **`max_steps`** *(int)*  
  The total action budget (Difficulty-dependent: 7, 10, or 13).
- **`data_completeness`** *(float ∈ [0.0, 1.0])*  
  Percentage of the environment that has been revealed via reconnaissance.

---

## 2. Action Space

The environment expects actions in **Strict JSON format** via the `/step` endpoint. 

### Reconnaissance: `request_info`
Used to reveal the ground-truth severity and demand of a specific zone.
```json
{
  "action_type": "request_info",
  "zone_id": "Z2"
}
```

### Deployment: `allocate_resource`
Deploys a specific amount of a resource to a zone. Note that amounts are rounded to 1 decimal place.
```json
{
  "action_type": "allocate_resource",
  "zone_id": "Z0",
  "resource_type": "medicine",
  "amount": 15.0
}
```

### Termination: `finalize`
Terminates the episode and triggers the final grading logic.
```json
{
  "action_type": "finalize"
}
```

---

## 3. Operational Logic (The "Why")

### Decision Efficiency (7 / 10 / 13 Steps)
We enforce tight step budgets to simulate the compressed timeline of a real-world disaster response.
- **Combatting Hallucination**: Short horizons prevent LLMs from entering repetitive loops or "action spamming."
- **Forcing Prioritization**: With only ~10 steps, an agent cannot visit every zone. It must use the `urgency_signal` to choose which information is worth the cost of acquisition.

### Reward Design: Precision Incentives
Our reward function is designed to separate "guessing" from "triage reasoning."
- **Intermediate Costs**: Every step (except `finalize`) carries a small negative cost to discourage inefficient movement.
- **Triple-Axis Grading**: Final success is not just about saving everyone; it’s about **resource stewardship**. Over-allocating to a low-severity zone is penalized just as heavily as under-allocating to a high-severity one.

This encourages agents to behave as **Strategic Orchestrators**, optimizing for the maximum possible utility within the constraints of the supply chain.
