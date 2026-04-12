# 🚨 Disaster Triage Environment (DTE)

An OpenEnv-compliant reinforcement learning benchmark for evaluating agentic decision-making in resource-constrained disaster logistics.

---

## 🔹 Problem Statement
In the immediate aftermath of a disaster, responders are faced with a chaotic "Fog of War." Resources are finite, timelines are compressed, and information is often missing or noisy. This is not just a pattern recognition task—it is **high-stakes decision-making under severe constraints.**

Existing RL benchmarks (like Atari or MuJoCo) focus on motor control or games. Our environment fills the gap by modeling **Operational Triage**, where the cost of a wrong move is measured in system-wide failure and misallocated survival resources.

👉 *“This is not just detection — it is decision-making under constraints.”*

---

## ⭐ Why This Benchmark Matters
Most RL environments focus on games or static reasoning tasks. This environment evaluates **sequential decision-making under uncertainty**, where:
- Information is incomplete (Partial Observability)
- Resources are limited (Finite Stockpiles)
- Decisions are irreversible (Resource Consumption)
- Time is constrained (Hard Step Budgets)

It provides a realistic testbed for evaluating AI agents in operational domains such as:
- **Disaster Response**
- **Supply Chain Logistics**
- **Crisis Management**

This makes it a benchmark for **Agentic AI**, not just predictive models.

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

## ⭐ Agent Decision Flow

<img width="642" height="696" alt="image" src="https://github.com/user-attachments/assets/d74422ce-f5f1-4891-a86f-66e3b30474f4" />

<img width="749" height="804" alt="image" src="https://github.com/user-attachments/assets/53c9b01d-b228-4480-b516-7762fb3f30ed" />



---

## 🔹 Key Features
- **Partial Observability**: Ground-truth demand is hidden behind the `request_info` action.
- **Resource Constraints**: Total budget for food/water/medicine is shared across all zones.
- **Temporal Decay**: Rewards diminish every step, simulating the increasing urgency of a real-life crisis.
- **Multi-Axis Grader**: Performance is judged on prioritization, demand fulfillment, and resource stewardship.

---

## 🔹 Tasks & Difficulty Levels
| Difficulty | Description | Budget | Observability |
| :--- | :--- | :---: | :--- |
| **Easy** | 3 Zones. Fully observable, deterministic allocation tasks. | 7 Steps | Full Visibility |
| **Medium** | 5 Zones. Partial observability, selective exploration required. | 10 Steps | ~50% Revealed |
| **Hard** | 7 Zones. High uncertainty + tight budget + prioritization critical. | 13 Steps | Fully Hidden |

---

## 🔹 Environment Design

### 🎮 Actions
- `request_info`: Reveal zone-level severity and demand (Cost: -0.01 reward).
- `allocate_resource`: Dispatch specific volume of a resource (Cost: -0.02 reward).
- `finalize`: Termination command to trigger the final grader.

### 👁️ Observations
- `zones`: Array containing `urgency_signal`, `revealed`, and (if revealed) `known_severity` and `known_demand`.
- `available_resources`: Current stockpiles of Food, Water, and Medicine.
- `step_count` / `max_steps`: Current action budget progress.
- `data_completeness`: Ratio of environmental knowledge vs. the unknown.

### 💰 Rewards
- **Step-wise Rewards**: Encourages finding new info (+0.05) and correct demand filling (+0.10).
- **Final Reward**: A blended score from the 3-axis Grader, scaled by difficulty.

---

## 🔹 Grader Details
The terminal reward is a weighted blend of three axes:
1.  **Prioritization (0.35)**: Rewards helping Severity 5 zones before Severity 1 zones (calculated using exponential weighting: `2^severity`).
2.  **Efficiency (0.40)**: Accuracy in matching the exact demand volume requested.
3.  **Utilization (0.25)**: Minimizing waste (over-allocation) and stockpile leftovers.

---

## 🔹 Baseline Agent (`inference.py`)
Our baseline agent uses an **LLM-based "Lead Architect" strategy**:
- It manages the `/reset` and `/step` lifecycle.
- It parses observations into structured decision histories.
- It produces **EXACT hackathon-compliant logs** to stdout.

### 📊 Sample Output
```text
[START] task=medium env=disaster-triage-env model=llama-3.1-8b-instant
[STEP] step=1 action={"action_type":"request_info","zone_id":"Z2"} reward=0.0800 done=false error=null
[STEP] step=2 action={"action_type":"allocate_resource","zone_id":"Z2","resource_type":"food","amount":35.0} reward=0.1000 done=false error=null
[END] success=true steps=10 score=0.6842 rewards=0.0800,0.1000,0.4500...
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
