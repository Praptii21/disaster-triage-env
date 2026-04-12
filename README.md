---
title: Disaster Triage Environment
emoji: 🚑
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
tags:
  - openenv
  - disaster-triage
  - agentic-ai
  - pomdp
  - rescue-logistics
short_description: Strategic triage benchmark for resource-constrained agents.
---
# 🚨 Disaster Triage Environment (DTE)

An OpenEnv-compliant reinforcement learning benchmark for evaluating agentic decision-making in resource-constrained disaster logistics.

---

## 🔹 Problem Statement
In the immediate aftermath of a disaster, responders are faced with a chaotic "Fog of War." Resources are finite, timelines are compressed, and information is often missing or noisy. This is not just a pattern recognition task—it is **high-stakes decision-making under severe constraints.**

Existing RL benchmarks (like Atari or MuJoCo) focus on motor control or games. Our environment fills the gap by modeling **Operational Triage**, where the cost of a wrong move is measured in system-wide failure and misallocated survival resources.

👉 *“This is not just detection — it is decision-making under constraints.”*

---

## 🌍 Real-World Utility

This environment models real-world disaster response logistics where:

- **Information is incomplete** (communication failure)
- **Resources are finite** (supply chain constraints)
- **Decisions are irreversible** (allocation cannot be undone)
- **Time directly impacts outcomes** (delayed response reduces effectiveness)

Unlike traditional RL benchmarks, this environment evaluates whether an agent can act as a decision-maker in high-stakes operational settings such as:

- Emergency response coordination
- Supply chain disruption management
- Crisis logistics planning

This makes it a benchmark for **Agentic AI**, not just predictive models.

---

## 💡 Novelty

This environment introduces a structured benchmark for:

**Resource-Constrained Decision Making under Partial Observability**

Unlike existing RL benchmarks:
- Not a game
- Not static reasoning
- Not single-step evaluation

It combines:
- POMDP dynamics
- Multi-objective optimization
- Irreversible actions
- Time-constrained planning

This makes it a test of true agentic behavior rather than pattern matching.

---

## ⭐ Design Principles
- **Determinism**: All rewards and transitions are 100% reproducible given the same seed.
- **Bounded Scoring**: Final rewards are strictly in **(0.01, 0.99)** to comply with OpenEnv validation and avoid degenerate extremes.
- **Efficiency Pressure**: Limited action horizons (7/10/13 steps) enforce meaningful, high-impact decisions.
- **No Reward Hacking**: Over-allocation, "action spamming," and random guessing are heavily penalized through the prioritization and utilization axes.
- **Exploration Cost**: Information gathering (`request_info`) has explicit trade-offs and temporal costs.

---

## 🔹 Solution Overview
The environment is modeled as a **POMDP (Partially Observable Markov Decision Process)** simulating a lead disaster architect coordinating multiple zones.
- **Zones**: Multiple crisis areas with unique, hidden severity and resource demands.
- **Resources**: Fixed global stockpiles of **Food, Water, and Medicine**.
- **Metrics**: Agents must balance noise-filtered signals (`urgency_signal`) against the cost of ground-truth reconnaissance.

---

## ⭐ Key Features

### 🔍 Partial Observability
Ground-truth severity and demand are hidden and must be actively revealed via `request_info`, forcing informed exploration.

### 📦 Global Resource Constraints
A shared, finite pool of Food, Water, and Medicine must be distributed across all zones, introducing real trade-offs.

### ⏱️ Temporal Pressure
Each action consumes a step and reduces achievable reward, simulating time-sensitive disaster response.

### ⚖️ Multi-Objective Optimization
Agents must simultaneously optimize prioritization, efficiency, and resource utilization, not just maximize a single metric.

### 🎯 Deterministic Evaluation
All transitions and rewards are fully reproducible, ensuring fair and consistent benchmarking across agents.

---

## 🎯 Tasks & Difficulty Levels

| Difficulty | Description | Step Budget | Observability | Core Challenge |
| :--- | :--- | :---: | :--- | :--- |
| **Easy** | 3 Zones with fully visible data | 7 | Full | Allocation correctness |
| **Medium** | 5 Zones with partial hidden information | 10 | ~50% Revealed | Explore vs Exploit |
| **Hard** | 7 Zones with high uncertainty and tight budget | 13 | Fully Hidden | Prioritization under constraints |

---

## 🧠 Environment Design

### 🎮 Action Space
- **`request_info`**
  - Reveals true severity and demand for a zone
  - → Trade-off: costs a step, but reduces uncertainty
- **`allocate_resource`**
  - Allocates a specific quantity of a resource to a zone
  - → Constraint: irreversible and bounded by global supply
- **`finalize`**
  - Terminates the episode and triggers final evaluation
  - → Risk: premature finalization reduces achievable score

### 👁️ Observation Space
At each step, the agent receives:
- **`zones`**: Array of zone objects containing:
  - `urgency_signal` (noisy indicator)
  - `revealed` (boolean)
  - `known_severity` (if revealed)
  - `known_demand` (if revealed)
- **`available_resources`**: Remaining global stockpile of Food, Water, Medicine
- **`step_count` / `max_steps`**: Tracks remaining decision budget
- **`data_completeness`**: Fraction of zones with revealed ground-truth data

---

## 💰 Reward Design
The reward function is structured to discourage guessing and encourage precise, high-impact decisions:

### Step-wise Signals
- Small positive rewards for meaningful actions (information gain, correct allocation)
- Implicit cost of time via limited step budget

### Terminal Reward
- Final score computed using a deterministic multi-axis grader
- Normalized to (0.01 – 0.99) for OpenEnv compliance

---

## 📊 Grader Design (Deterministic)
The final score is a weighted combination of three axes:

**Final Score = (0.35 × Prioritization) + (0.40 × Efficiency) + (0.25 × Utilization)**

1.  **Prioritization (35%)**
    - Rewards serving high-severity zones first
    - Uses severity-weighted scaling (e.g., 2^severity)
    - Ensures critical zones contribute disproportionately more
2.  **Efficiency (40%)**
    - Measures how accurately the agent matches actual demand
    - Penalizes under-allocation and missed needs
3.  **Utilization (25%)**
    - Penalizes over-allocation and wasted resources
    - Encourages optimal use of limited supply

👉 **Key Property:** High scores are only achievable through correct prioritization + precise allocation + minimal waste

---

## 🤖 Baseline Agent (`inference.py`)
The baseline agent simulates an LLM-driven disaster response coordinator:
- Executes a full agent loop using `/reset` and `/step`
- Interprets observations into structured decision context
- Balances exploration (`request_info`) and execution (`allocate_resource`)
- Emits strictly formatted logs: `[START]`, `[STEP]`, `[END]`
- Produces fully reproducible evaluation runs for all difficulty tiers
### 📊 Sample Output
```text
[START] task=medium env=disaster-triage-env model=llama-3.1-8b-instant
[STEP] step=1 action={"action_type":"request_info","zone_id":"Z2"} reward=0.0200 done=false error=null
[STEP] step=2 action={"action_type":"allocate_resource","zone_id":"Z2","resource_type":"food","amount":35.0} reward=0.0500 done=false error=null
[END] success=true steps=10 score=0.684 rewards=0.02,0.05,0.45...
```

---

## 🔹 Setup & Usage

### 🚀 Local Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure variables (see .env.example)
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_token_here"

# 3. Start Environment Server
python server/app.py

# 4. Run Baseline Inference
python inference.py --task medium
```

### 🐳 Docker Instructions
```bash
docker build -t disaster-triage .
docker run -p 7860:7860 --env-file .env disaster-triage
```

---

## 🔹 API Endpoints
- `POST /reset`: Initialize or restart a session (returns first observation).
- `POST /step`: Execute an agent action.
- `GET /state`: Global ground-truth view (for debugging/diagnostic only).
- `GET /health`: Server liveness and active session summary.

---

## 🔹 Performance Constraints
- **Runtime**: Full evaluation (Easy, Medium, Hard) runs in **< 20 minutes**.
- **Hardware**: Designed to run on standard **2 vCPU / 8GB RAM** instances.

---

## 🔹 Creativity & Uniqueness
- **Beyond Toy Problems**: This environment forces agents to contend with genuine operational fog, simulating logistics rather than arcade physics.
- **Agentic Stress Test**: The tight step budgets and exploration costs separate simple chatbots from capable action-oriented agents.
- **Logistics Modeling**: Every detail, from the resource types to the noise in the urgency signal, is designed to mirror real-world supply chain optimization during a crisis.

---

**Disaster Triage Environment** — *Moving Intelligence from Chat to Execution.*
