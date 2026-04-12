# Case Study: Disaster Triage as Resource Constrained Optimization

## Overview
Recent Reinforcement Learning (RL) benchmarks often focus on recreational games (Atari, Chess, Go) or simulated physics (MuJoCo). While impressive, these environments frequently lack the high-stakes, probabilistic gravity of real-world logistical crises. 

The **Disaster Triage Environment** was designed to bridge this gap by framing disaster response not as a game, but as a problem of **Resource Constrained Optimization under Uncertainty**.

## Core Mechanics & Real-Life Modeling

### 1. Priority (Severity-Weighted Triage)
In a real-life mass casualty incident, responders use the SALT or START protocols to categorize patients. Our environment models this through `known_severity` and `urgency_signal`. 
*   **The Logic**: A Severity 5 zone (Immediate) yields exponentially higher rewards for demand fulfillment than a Severity 1 zone (Minor). 
*   **The Trade-off**: Agents must decide whether to fully save one critical zone or partially stabilize three moderate ones—a classic ethical and logistical dilemma in disaster medicine.

### 2. Resource Scarcity & Systemic Bottlenecks
Traditional RL agents often operate in environments with infinite "ammunition" or movement. Here, resources (Food, Water, Medicine) are finite and non-renewable within a session.
*   **The Logic**: Over-allocating to a revealed zone early might leave the agent bankrupt when a more critical zone is discovered later.
*   **The Modeling**: This mirrors the "Last Mile" logistics challenge where the cost of transport and limited global stock requires precise, one-shot allocation rather than iterative trial-and-error.

### 3. Information Decay & Partial Observability
In the wake of a disaster, communication infrastructure is often the first thing to fail.
*   **The Logic**: The environment starts in a state of high entropy. The agent identifies "signals" but lacks "ground truth" without spending time (steps) on reconnaissance (`request_info`).
*   **The Cost of knowledge**: Every step spent gathering information introduces temporal degradation, where the expected utility of unmet demand decreases over time. This creates a dynamic trade-off between exploration (information gathering) and exploitation (resource allocation). This simulates the **Temporal Decay** of relevance in disaster data—stale information is as dangerous as no information.

All transitions and reward calculations are deterministic given the current state, ensuring reproducibility and fair benchmarking across agents.

## Closing "The Gap"
By shifting the focus from "winning a game" to **minimizing regret in a resource-constrained system**, we provide a benchmark that tests an LLM's ability to:
1.  **Reason under Noise**: Interpret noisy urgency signals.
2.  **Execute Strategic Patience**: Spend budget on info-gathering before committing resources.
3.  **Perform Multi-Objective Optimization**: Balance time, resource volume, and zone priority.

This environment moves beyond traditional RL benchmarks by modeling time-sensitive, high-stakes decision-making under uncertainty, where delayed or incorrect actions have compounding consequences.

It provides a realistic testbed for evaluating whether modern LLM agents can transition from passive reasoning systems to active decision-makers in operational domains such as disaster response, supply chain logistics, and crisis management.

In doing so, it helps answer a critical question for the next generation of AI:
**Can agents not only reason, but act effectively when information is incomplete, resources are limited, and time is critical?**
