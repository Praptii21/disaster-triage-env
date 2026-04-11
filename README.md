---
title: "DeadCore: Disaster Triage Benchmark (DTB)"
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
license: mit
tags:
  - openenv
  - deadcore
  - ai-benchmark
  - pomdp
  - safety-critical
  - disaster-response
short_description: Deterministic evaluation of AI decision-making in disasters.
---

# Disaster Triage Benchmark (DTB)
**A Deterministic Evaluation Framework for Safety-Critical Resource Allocation**

> [!NOTE]
> This repository contains the Phase 2 submission for the **Meta × HuggingFace OpenEnv Challenge 2026**.  
> **DTB** is a zero-LLM deterministic benchmark evaluating autonomic AI agents in high-stakes triage scenarios.

---

## 🔬 Motivation: The Cost of Information
Autonomous systems operating in real-world disaster scenarios face a fundamental trade-off: **Information Acquisition vs. Immediate Action**. In emergency response, every second spent gathering data is a second subtracted from life-saving deployment. 

Existing RL benchmarks (Atari, MuJoCo) often provide agents with perfect or free state observations. **DTB** fills this research gap by treating **Observation as a Costly Resource**. It evaluates an agent's ability to minimize uncertainty while maximizing impact across a multi-zone environment under a strict temporal budget.

---

## 🏗️ Environment Architecture

DeadCore DTB is structured as a **Partially Observable Markov Decision Process (POMDP)**.

### Agent Objectives
The AI agent operates as a **Response Commander**, tasked with distributing discrete resource bundles (Food, Water, Medicine) across $N$ disaster zones with varying severity [1-5].

### Core Constraints
- **Scarcity**: Initial resources are insufficient to meet 100% of global demand; the agent must prioritize.
- **Observability**: Zones are "Hidden" by default. Agents must expend a budget-step to `request_info`.
- **Temporal Budget**: Episodes terminate at $T$ steps, forcing the agent to balance exploration and execution.

---

## 📊 Evaluation Tiers

The benchmark rotates through three task scenarios of increasing complexity:

| Tier | Task ID | Complexity | Observability | Reasoning Axis |
| :--- | :--- | :--- | :--- | :--- |
| **Easy** | `easy` | 3 Zones | Full | **Execution Speed**: Strategic allocation pattern under full visibility. |
| **Medium** | `medium` | 5 Zones | Partial | **Uncertainty Management**: Balancing intel gathering with partial data. |
| **Hard** | `hard` | 7 Zones | Hidden | **Risk Mitigation**: Prioritizing high-severity detection under extreme scarcity. |

---

## ⚖️ Deterministic Grading Engine
All grading is performed via **DeadCore's Zero-LLM Evaluation Engine**, ensuring absolute reproducibility and fairness. Performance is quantified across three primary axes (75% of final reward):

1. **Prioritization Accuracy (50%)**: Measures the degree to which high-severity zones received priority over low-severity ones.
2. **Allocation Efficiency (30%)**: Normalized MAE between allocated resources and true ground-truth demand.
3. **Resource Utilization (20%)**: Assesses the conversion rate of available resources into productive impact vs. over-allocation waste.

---

## ⚡ Dense Reward System & Anti-Hacking
To support robust agent learning, DeadCore DTB provides a continuous, step-wise reward signal $r_t \in [0, 1]$.

| Signal Type | Condition | Reward ($r_t$) | Positioning |
| :--- | :--- | :--- | :--- |
| **Strategic Discovery** | Revealed Severity 4-5 Zone | 0.04 | Rewards high-intel discovery. |
| **Perfect Match** | Medicine → Severity 5 Zone | 0.10 | Penalizes priority inversion. |
| **Redundant Intel** | Repeated Info Request | 0.005 | Discourages budget-wasting loops. |
| **Over-Allocation** | Allocation > 110% Demand | -0.005 | Prevents "dumping" heuristic. |

**Final Reward Blending:** The terminal signal is a composite of $75\%$ Grader Score and $25\%$ Step Quality. This design **prevents reward hacking**—an agent cannot achieve a high score through luck or "lazy" finalization; it must demonstrate consistent reasoning at every step.

---

## ⚡ Reward Logic Summary
The DeadCore DTB utilizes a **dense, step-wise reward signal** to ensure consistent learning gradients:
- **High-Impact Discovery**: +0.04 for identifying severity 4-5 zones (prioritizing critical risk).
- **Resource Match**: +0.10 for perfect sector/resource alignment (e.g., Medicine for S5).
- **Exploration Signal**: +0.02 for new zone discovery; +0.005 for repeat requests (minimal signal).
- **Sustainability Penalty**: -0.005 for over-allocation beyond 110% of true demand.

---

## 🔌 API Endpoints
The environment server exposes a standard REST interface on Port 7860:
- `POST /reset`: Initialize new episode (requires `task_id`).
- `POST /step`: Dispatch agent action; returns observation, reward, and done flags.
- `GET /tasks`: Retrieve supported difficulty levels.
- `GET /state`: Diagnostic endpoint for ground-truth inspection.
- `GET /health`: Liveness probe for monitoring.

---

## 📊 Baseline Performance Log Sample
Evaluation utilizes the OpenEnv standard `[START]`, `[STEP]`, and `[END]` protocol for automated parsing.

```text
[START] task=easy env=deadcore-dtb model=llama-3.1-8b-instant
[STEP] step=1 action={"action_type":"request_info","zone_id":"Z0"} reward=0.04 done=false error=null
[STEP] step=2 action={"action_type":"allocate_resource","zone_id":"Z0","resource_type":"medicine","amount":43.7} reward=0.10 done=false error=null
...
[STEP] step=10 action={"action_type":"finalize"} reward=0.88 done=true error=null
[END] success=true steps=10 rewards=0.04,0.10,0.05,0.06,0.03,0.03,0.03,0.06,0.05,0.88
```

---

## 🚀 Deployment & Usage

### 1. Local Evaluation
```bash
git clone https://github.com/Praptii21/disaster-triage-env.git
cd disaster-triage-env
pip install -r requirements.txt
python inference.py --task medium
```

### 2. Infrastructure (Docker)
The environment exposes a RESTful API (FastAPI) configured for **HuggingFace Spaces** (Port 7860).
```bash
docker build -t deadcore_dtb:latest .
docker run -p 7860:7860 deadcore_dtb:latest
```

---

## 🔮 Research Roadmap
To evolve DeadCore DTB into an enterprise-standard SRE/Response benchmark, we have mapped the following trajectories:
- **Phase 3 (Active POMDP)**: Introducing dynamic environment updates where zone severities escalate if left unserved.
- **Phase 4 (Multi-Agent)**: Evaluating collaborative triage protocols between distributed AI commanders.
- **Phase 5 (Tool-Augmented)**: Integrating external diagnostic tools (e.g., API calls to simulated telemetry) for root-cause prioritization.

---

## 📜 Citation
```bibtex
@software{deadcore_dtb_2026,
  title   = {DeadCore Disaster Triage Benchmark: Evaluating Safety-Critical Scarcity Management},
  author  = {Team DeadCore (Prapti)},
  year    = {2026},
  url     = {https://huggingface.co/spaces/Praptii21/disaster-triage-env},
  note    = {Deterministic POMDP Evaluation Environment for AI Agents}
}
```

---
*Developed for the Meta OpenEnv Challenge 2026.*
