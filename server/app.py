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

try:
    from inference import get_scripted_action, get_llm_action
except Exception as e:
    print(f"[SYSTEM] Error importing inference: {e}")
    def get_scripted_action(obs, step, task): return {"action_type": "finalize"}
    def get_llm_action(hist, obs, step, task): return {"action_type": "finalize"}

_sessions:   Dict[str, DisasterTriageEnv] = {}
_created_at: Dict[str, float]             = {}

def _get_session(task_id: str) -> DisasterTriageEnv:
    if task_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{task_id}' not found.")
    return _sessions[task_id]

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

app = FastAPI(
    title="Disaster Triage RL Environment",
    description="OpenEnv-compliant HTTP wrapper & Gradio Dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ResetResponse(BaseModel):
    task_id: str; observation: Dict[str, Any]; done: bool = False; info: Dict[str, Any]

class StepResponse(BaseModel):
    task_id: str; observation: Dict[str, Any]; reward: float; done: bool; info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str; version: str; active_sessions: int; sessions: Dict[str, Dict[str, Any]]

class ActionRequest(BaseModel):
    task_id:       str             = Field("medium")
    action_type:   str             = Field(...)
    zone_id:       Optional[str]   = Field(None)
    resource_type: Optional[str]   = Field(None)
    amount:        Optional[float] = Field(None, gt=0)

def _parse_action(req: ActionRequest) -> Action:
    action_type = ActionType(req.action_type)
    resource_type = ResourceType(req.resource_type) if req.resource_type else None
    return Action(action_type=action_type, zone_id=req.zone_id, resource_type=resource_type, amount=req.amount)

def _strip_reward_from_info(info: dict) -> dict:
    return {k: v for k, v in info.items() if k != "reward"}

def _clamp_reward(reward: float) -> float:
    return round(float(np.clip(reward, 0.01, 0.99)), 3)

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    session_info = {}
    for tid, env in _sessions.items():
        session_info[tid] = {"difficulty": env.difficulty.value, "done": env.done, "step_count": env.step_count, "age_seconds": round(time.time() - _created_at.get(tid, 0), 1)}
    return HealthResponse(status="ok", version="1.0.0", active_sessions=len(_sessions), sessions=session_info)

@app.post("/reset", response_model=ResetResponse, tags=["Environment"])
async def reset(request: Request) -> ResetResponse:
    try: body = await request.json()
    except: body = {}
    task_id = body.get("task_id", "medium"); difficulty = body.get("difficulty", "medium"); seed = body.get("seed")
    env = DisasterTriageEnv.make(difficulty, seed=seed); obs = env.reset(seed=seed)
    _sessions[task_id] = env; _created_at[task_id] = time.time()
    return ResetResponse(task_id=task_id, observation=obs.to_dict(), done=False, info={"task_id": task_id, "difficulty": difficulty})

@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(request: Request) -> StepResponse:
    try: body = await request.json()
    except: body = {}
    task_id = body.get("task_id", "medium"); env = _get_session(task_id)
    req = ActionRequest(**body); action = _parse_action(req)
    obs, reward, done, info = env.step(action)
    return StepResponse(task_id=task_id, observation=obs.to_dict(), reward=_clamp_reward(reward), done=done, info=_strip_reward_from_info(info))

# ---------------------------------------------------------------------------
# Gradio Dashboard
# ---------------------------------------------------------------------------

def run_simulation(task: str) -> Generator:
    base_url = "http://127.0.0.1:7860"
    logs = [f"◈ INITIALIZING MISSION: {task.upper()}"]
    yield "\n".join(logs)

    try:
        resp = requests.post(f"{base_url}/reset", json={"task_id": "dashboard", "difficulty": task})
        if resp.status_code != 200:
            logs.append(f"✗ RESET FAILED ({resp.status_code})")
            yield "\n".join(logs); return

        data = resp.json(); obs = data.get("observation", {}); done = data.get("done", False)
        history = []; step_n = 0; rewards_list = []
        model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

        logs.append(f"[START] task={task} env=disaster-triage-env model={model}")
        logs.append("─" * 60)
        yield "\n".join(logs)

        while not done:
            step_n += 1
            try:
                action = get_llm_action(history, obs, step_n, task)
            except Exception as e:
                logs.append(f"  ⚠ LLM fallback: {str(e)[:40]}...")
                action = get_scripted_action(obs, step_n, task)

            resp = requests.post(f"{base_url}/step", json={"task_id": "dashboard", **action})
            if resp.status_code != 200:
                logs.append(f"✗ STEP FAILED ({resp.status_code})"); break

            res = resp.json(); obs = res.get("observation", {}); reward = res.get("reward", 0.0)
            done = res.get("done", False)
            rewards_list.append(reward)

            action_type = action.get("action_type", "")
            zone        = action.get("zone_id", "")
            resource    = action.get("resource_type", "")
            amount      = action.get("amount", "")
            action_json = json.dumps(action, separators=(',', ':'))

            resource_icons = {"food": "🍱", "water": "💧", "medicine": "🧪"}
            icon = resource_icons.get(resource, "📦")

            if action_type == "request_info":
                logs.append(f"  🔍 [RECON]    Zone: {zone} | Status: Scanning")
            elif action_type == "allocate_resource":
                logs.append(f"  {icon} [DISPATCH] Zone: {zone} | Supply: {resource.upper()} | Qty: {amount} units")
                logs.append(f"     [TELEMETRY] Step: {step_n} | Reward: {reward:.4f} | Status: Operational")
            elif action_type == "finalize":
                logs.append(f"  ✔  [FINALIZE] Closing episode — triggering grader evaluation")

            logs.append(f"     [STEP] step={step_n} action={action_json} reward={reward:.4f} done={str(done).lower()} error=null")
            yield "\n".join(logs)
            time.sleep(1.0)

        logs.append("─" * 60)
        mission_score = rewards_list[-1] if rewards_list else 0.001
        rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
        status = "✔ MISSION SUCCESS" if done else "✗ MISSION INCOMPLETE"
        logs.append(f"{status}")
        logs.append(f"[END] success={str(done).lower()} steps={step_n} score={mission_score:.3f} rewards={rewards_str}")
        yield "\n".join(logs)

    except Exception as e:
        logs.append(f"✗ FATAL: {str(e)}")
        yield "\n".join(logs)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

body, .gradio-container {
    background: #0a0e17 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

.label-wrap label span {
    color: #4a5568 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

select, .gr-dropdown select {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 4px !important;
}

button.primary {
    background: #ff4444 !important;
    border: none !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}

button.primary:hover {
    background: #cc2222 !important;
    transform: translateY(-1px) !important;
}

textarea {
    background: #050810 !important;
    border: 1px solid #1f2937 !important;
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.7 !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}

footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="DISASTER TRIAGE // COMMAND") as demo:
    gr.HTML("""
    <div style="border-left: 3px solid #ff4444; padding-left: 1.2rem; margin-bottom: 2rem;">
        <h1 style="font-family: Syne, sans-serif; font-size: 2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin: 0 0 0.3rem 0;">
            DISASTER TRIAGE
        </h1>
        <p style="color: #64748b; font-size: 0.85rem; font-style: italic; margin-top: 0.5rem; line-height: 1.4;">
            Strategic multi-objective allocation of food, water, and medicine under partial observability.
        </p>
    </div>
    """)

    with gr.Row():
        task_input = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="medium",
            label="MISSION PRIORITY",
            scale=1,
        )
        start_btn = gr.Button("▶ DEPLOY AGENT", variant="primary", scale=1)

    gr.HTML("<div style='height:12px'></div>")

    log_output = gr.Textbox(
        label="MISSION FEED",
        lines=28,
        max_lines=35,
        interactive=False,
        autoscroll=True,
        placeholder="Awaiting deployment order...",
    )

    gr.Markdown("<p style='color:#4a5568; font-size:0.7rem; margin-top:8px; font-family:monospace;'>Logs follow OpenEnv format &nbsp;·&nbsp; [START] → [STEP] × N → [END] &nbsp;·&nbsp; 🍱 Food &nbsp; 💧 Water &nbsp; 🧪 Medicine</p>")

    start_btn.click(fn=run_simulation, inputs=[task_input], outputs=[log_output])

app = gr.mount_gradio_app(app, demo, path="/")

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
