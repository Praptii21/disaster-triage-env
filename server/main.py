"""
server/app.py
=============
FastAPI server that wraps DisasterTriageEnv in an OpenEnv-compliant HTTP API.
Integrated with a Gradio Console UI for real-time monitoring.
"""

from __future__ import annotations
import uvicorn
import time
import json
import os
import requests
import gradio as gr
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Generator

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Local environment imports
from app.env import DisasterTriageEnv
from app.models import (
    Action,
    ActionType,
    Difficulty,
    Observation,
    ResourceType,
)

# Agent imports for the UI
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
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    for diff in ("easy", "medium", "hard"):
        try:
            env = DisasterTriageEnv.make(diff)
            env.reset()
            _sessions[diff]   = env
            _created_at[diff] = time.time()
        except Exception as e:
            print(f"[WARN] Failed to bootstrap {diff}: {e}")
    yield
    _sessions.clear()
    _created_at.clear()

# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Disaster Triage RL Environment",
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
# Pydantic Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:          str
    version:         str
    active_sessions: int
    sessions:        Dict[str, Dict[str, Any]]

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

class ActionRequest(BaseModel):
    task_id:       str            = Field("medium")
    action_type:   str            = Field(..., description="request_info | allocate_resource | finalize")
    zone_id:       Optional[str]  = Field(None)
    resource_type: Optional[str]  = Field(None)
    amount:        Optional[float] = Field(None)

# ---------------------------------------------------------------------------
# Helper Logic
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

def _clamp_reward(reward: float) -> float:
    return round(float(np.clip(reward, 0.01, 0.99)), 4)

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    session_info = {
        tid: {
            "difficulty": env.difficulty.value,
            "step_count": env.step_count,
            "age_seconds": round(time.time() - _created_at.get(tid, 0), 1),
        } for tid, env in _sessions.items()
    }
    return HealthResponse(
        status="ok",
        version="1.0.0",
        active_sessions=len(_sessions),
        sessions=session_info,
    )

@app.post("/reset", response_model=ResetResponse, tags=["Environment"])
async def reset(request: Request):
    try: body = await request.json()
    except: body = {}
    
    task_id = body.get("task_id", "easy")
    diff_enum = Difficulty(str(body.get("difficulty", task_id)).lower())
    
    env = DisasterTriageEnv(difficulty=diff_enum)
    obs = env.reset()
    _sessions[task_id] = env
    _created_at[task_id] = time.time()
    
    return ResetResponse(
        task_id=task_id, 
        observation=obs.to_dict(), 
        info={"message": "Reset OK"}
    )

@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(request: Request):
    try: body = await request.json()
    except: body = {}
    
    task_id = body.get("task_id", "easy")
    env = _get_session(task_id)
    
    req = ActionRequest(**body)
    action = _parse_action(req)
    obs, reward, done, info = env.step(action)
    
    return StepResponse(
        task_id=task_id,
        observation=obs.to_dict(),
        reward=_clamp_reward(reward),
        done=done,
        info=info
    )

# ---------------------------------------------------------------------------
# Gradio UI Integration
# ---------------------------------------------------------------------------

def run_ui_simulation(task: str) -> Generator:
    """Internal UI simulation logic that talks to the local API."""
    port = os.getenv("PORT", "7860")
    base_url = f"http://127.0.0.1:{port}"
    logs = [f"> Initializing simulation: {task.upper()}"]
    yield "\n".join(logs)
    
    try:
        # Reset
        resp = requests.post(f"{base_url}/reset", json={"task_id": "ui_session", "difficulty": task})
        data = resp.json()
        obs = data.get("observation", {})
        done = False
        step = 0
        total_reward = 0
        logs.append(f"> Triage Engine Ready | Connected to Port {port}")
        yield "\n".join(logs)

        while not done:
            step += 1
            action = get_llm_action([], obs, step, task)
            
            resp = requests.post(f"{base_url}/step", json={"task_id": "ui_session", **action})
            res = resp.json()
            obs = res.get("observation", {})
            reward = res.get("reward", 0.0)
            done = res.get("done", False)
            total_reward += reward

            logs.append(f"step={step} action={json.dumps(action)} reward={reward:.4f} done={str(done).lower()}")
            yield "\n".join(logs)
            time.sleep(1.0)

        logs.append(f"\n[END] MISSION COMPLETE | score={total_reward:.3f}")
        yield "\n".join(logs)
    except Exception as e:
        yield f"\nERROR: {str(e)}"

with gr.Blocks(title="Disaster Triage Console", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("# 🚑 Disaster Triage Console")
    gr.Markdown("Real-time monitoring of the Triage Engine.")
    with gr.Row():
        task_input = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Task")
        start_btn = gr.Button("🚀 Start Mission", variant="primary")
    log_output = gr.Textbox(label="Logs", lines=20, autoscroll=True)
    start_btn.click(run_ui_simulation, inputs=[task_input], outputs=[log_output])

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    # Mount the UI BEFORE launching uvicorn
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=port)
