import time
import os
import json
import uvicorn
import requests
import gradio as gr
from typing import Any, Dict, List, Optional, Generator
from contextlib import asynccontextmanager
from fastapi import Body, FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import numpy as np

from app.env import DisasterTriageEnv
from app.models import (
    Action,
    ActionType,
    Difficulty,
    Observation,
    ResourceType,
)

# Import the brain for simulation decisions
try:
    from inference import get_scripted_action, get_llm_action
except Exception as e:
    print(f"[SYSTEM] Error importing inference: {e}")
    def get_scripted_action(obs, step, task): return {"action_type": "finalize"}
    def get_llm_action(hist, obs, step, task): return {"action_type": "finalize"}

# ---------------------------------------------------------------------------
# Session store  (task_id → env)
# ---------------------------------------------------------------------------

_sessions:   Dict[str, DisasterTriageEnv] = {}
_created_at: Dict[str, float]             = {}

def _get_session(task_id: str) -> DisasterTriageEnv:
    if task_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{task_id}' not found. "
                "Call POST /reset first to initialise an environment."
            ),
        )
    return _sessions[task_id]

# ---------------------------------------------------------------------------
# Lifespan — bootstrap default sessions for each difficulty
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    for diff in ("easy", "medium", "hard"):
        try:
            env = DisasterTriageEnv.make(diff)
            env.reset()
            _sessions[diff]   = env
            _created_at[diff] = time.time()
            print(f"[startup] Session '{diff}' initialized OK")
        except Exception as e:
            print(f"[startup] ERROR initializing '{diff}': {e}")
    yield
    _sessions.clear()
    _created_at.clear()

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Disaster Triage RL Environment",
    description="OpenEnv-compliant HTTP wrapper & Gradio Dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetResponse(BaseModel):
    task_id:     str
    observation: Dict[str, Any]
    done:        bool = False
    info:        Dict[str, Any]

class StepResponse(BaseModel):
    task_id:     str
    observation: Dict[str, Any]
    reward:      float
    done:        bool
    info:        Dict[str, Any]

class HealthResponse(BaseModel):
    status:          str
    version:         str
    active_sessions: int
    sessions:        Dict[str, Dict[str, Any]]

class ActionRequest(BaseModel):
    task_id:       str             = Field("medium")
    action_type:   str             = Field(..., description="request_info | allocate_resource | finalize")
    zone_id:       Optional[str]   = Field(None)
    resource_type: Optional[str]   = Field(None)
    amount:        Optional[float] = Field(None, gt=0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_action(req: ActionRequest) -> Action:
    action_type = ActionType(req.action_type)
    resource_type = ResourceType(req.resource_type) if req.resource_type else None
    action = Action(
        action_type=action_type,
        zone_id=req.zone_id,
        resource_type=resource_type,
        amount=req.amount,
    )
    return action

def _strip_reward_from_info(info: dict) -> dict:
    return {k: v for k, v in info.items() if k != "reward"}

def _clamp_reward(reward: float) -> float:
    return round(float(np.clip(reward, 0.01, 0.99)), 3)

# ---------------------------------------------------------------------------
# REST API Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="Server liveness check", tags=["System"])
async def health() -> HealthResponse:
    session_info = {}
    for tid, env in _sessions.items():
        session_info[tid] = {
            "difficulty":  env.difficulty.value,
            "done":        env.done,
            "step_count":  env.step_count,
            "age_seconds": round(time.time() - _created_at.get(tid, 0), 1),
        }
    return HealthResponse(status="ok", version="1.0.0", active_sessions=len(_sessions), sessions=session_info)

@app.post("/reset", response_model=ResetResponse, summary="Reset environment", tags=["Environment"])
async def reset(request: Request) -> ResetResponse:
    try: body = await request.json()
    except: body = {}
    task_id = body.get("task_id", "medium")
    difficulty = body.get("difficulty", "medium")
    seed = body.get("seed")
    env = DisasterTriageEnv.make(difficulty, seed=seed)
    obs = env.reset(seed=seed)
    _sessions[task_id] = env
    _created_at[task_id] = time.time()
    return ResetResponse(task_id=task_id, observation=obs.to_dict(), done=False, info={"task_id": task_id, "difficulty": difficulty})

@app.post("/step", response_model=StepResponse, summary="Step action", tags=["Environment"])
async def step(request: Request) -> StepResponse:
    try: body = await request.json()
    except: body = {}
    task_id = body.get("task_id", "medium")
    env = _get_session(task_id)
    req = ActionRequest(**body)
    action = _parse_action(req)
    obs, reward, done, info = env.step(action)
    return StepResponse(task_id=task_id, observation=obs.to_dict(), reward=_clamp_reward(reward), done=done, info=_strip_reward_from_info(info))

# ---------------------------------------------------------------------------
# Gradio Dashboard Logic
# ---------------------------------------------------------------------------

def run_simulation(task: str) -> Generator:
    # Use internal unified port 7860
    base_url = "http://127.0.0.1:7860"
    logs = [f"> Initializing simulation: {task.upper()}"]
    yield "\n".join(logs)
    
    try:
        resp = requests.post(f"{base_url}/reset", json={"task_id": "dashboard", "difficulty": task})
        if resp.status_code != 200:
            logs.append(f"ERROR: Reset failed ({resp.status_code})")
            yield "\n".join(logs); return
        
        data = resp.json(); obs = data.get("observation", {}); done = data.get("done", False)
        history = []; step = 0; total_reward = 0
        logs.append(f"> Connected to backend at {base_url}")
        yield "\n".join(logs)

        while not done:
            step += 1
            try: action = get_llm_action(history, obs, step, task)
            except Exception as e:
                logs.append(f"AGENT_WARN: {str(e)[:50]}... Falling back to scripted.")
                action = get_scripted_action(obs, step, task)

            resp = requests.post(f"{base_url}/step", json={"task_id": "dashboard", **action})
            if resp.status_code != 200:
                logs.append(f"ERROR: Step failed ({resp.status_code})"); break
                
            res = resp.json(); obs = res.get("observation", {}); reward = res.get("reward", 0.0)
            done = res.get("done", False); total_reward += reward
            logs.append(f"step={step} action={json.dumps(action)} reward={reward:.4f} done={str(done).lower()}")
            yield "\n".join(logs)
            time.sleep(1.0)

        logs.append(f"\n[END] MISSION COMPLETE | steps={step} score={total_reward:.3f}")
        yield "\n".join(logs)
    except Exception as e:
        logs.append(f"fatal_error: {str(e)}"); yield "\n".join(logs)

with gr.Blocks(title="Disaster Triage Console", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("# 🚑 Disaster Triage Console")
    gr.Markdown("Real-time mission coordination and environment logs.")
    with gr.Row():
        task_input = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Mission Priority")
        start_btn = gr.Button("🚀 Start Simulation", variant="primary")
    log_output = gr.Textbox(label="Terminal Feed", lines=25, max_lines=30, interactive=False, autoscroll=True)
    start_btn.click(fn=run_simulation, inputs=[task_input], outputs=[log_output])

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
