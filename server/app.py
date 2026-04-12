"""
server/app.py
=============
FastAPI server that wraps DisasterTriageEnv in an OpenEnv-compliant HTTP API.

Endpoints
---------
GET  /health              → server liveness
POST /reset               → start / restart an episode
                            returns: { observation, done=false, info }
POST /step                → execute one action
                            returns: { observation, reward, done, info }
                            reward is ONLY at the top level (not inside info)
GET  /state               → full internal state (true values, not masked)
GET  /explain             → optional grader diagnostic (not required by OpenEnv)
GET  /sessions            → list active sessions
DELETE /session/{task_id} → destroy a session

Sessions are keyed by task_id ("easy" | "medium" | "hard" or any string).
"""

from __future__ import annotations
import uvicorn
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import gradio as gr
from fastapi import Body, FastAPI, HTTPException, Path, Request
from ui import demo  # Import our Gradio console
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.env import DisasterTriageEnv
from app.models import (
    Action,
    ActionType,
    Difficulty,
    Observation,
    ResourceType,
)


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
    # Bootstrap one session per difficulty — make() already calls reset(),
    # but we call reset() again explicitly so the lifespan is self-contained
    # even if make() changes in the future.
    for diff in ("easy", "medium", "hard"):
        env = DisasterTriageEnv.make(diff)   # make() calls reset() internally
        env.reset()                           # explicit reset — belt-and-suspenders
        _sessions[diff]   = env
        _created_at[diff] = time.time()
    yield
    _sessions.clear()
    _created_at.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Disaster Triage RL Environment",
    description=(
        "OpenEnv-compliant HTTP wrapper around the DisasterTriageEnv POMDP. "
        "An agent acts as a disaster coordinator allocating limited resources "
        "across partially observable zones under a step budget."
    ),
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

class ResetRequest(BaseModel):
    """POST /reset — OpenEnv reset request."""
    task_id:    str           = Field(
        "medium",
        description="Session identifier: 'easy' | 'medium' | 'hard' (or any custom string)",
    )
    difficulty: str           = Field("medium", description="easy | medium | hard")
    seed:       Optional[int] = Field(None,     description="Optional random seed")

    model_config = {"json_schema_extra": {
        "examples": [
            {"task_id": "easy",   "difficulty": "easy"},
            {"task_id": "medium", "difficulty": "medium", "seed": 42},
            {"task_id": "hard",   "difficulty": "hard",   "seed": 7},
        ]
    }}


class ActionRequest(BaseModel):
    """POST /step — agent action."""
    task_id:       str            = Field("medium")
    action_type:   str            = Field(..., description="request_info | allocate_resource | finalize")
    zone_id:       Optional[str]  = Field(None, description="Zone ID: 'Z0', 'Z1', ...")
    resource_type: Optional[str]  = Field(None, description="food | water | medicine")
    amount:        Optional[float] = Field(None, gt=0)

    model_config = {"json_schema_extra": {
        "examples": [
            {"task_id": "medium", "action_type": "request_info",      "zone_id": "Z2"},
            {"task_id": "medium", "action_type": "allocate_resource",  "zone_id": "Z0",
             "resource_type": "food", "amount": 30.0},
            {"task_id": "medium", "action_type": "finalize"},
        ]
    }}


# ---------------------------------------------------------------------------
# OpenEnv-compliant response schemas
# ---------------------------------------------------------------------------

class ResetResponse(BaseModel):
    """
    OpenEnv /reset response.
    NO reward field — reward is only meaningful after actions.
    """
    task_id:     str
    observation: Dict[str, Any]
    done:        bool = False
    info:        Dict[str, Any]


class StepResponse(BaseModel):
    """
    OpenEnv /step response.
    reward is ONLY at the top level — never duplicated inside info.
    Reward is clamped to [0, 1] at finalize; negative on intermediate steps.
    """
    task_id:     str
    observation: Dict[str, Any]
    reward:      float           # top-level only; clamped to [0,1] at terminal step
    done:        bool
    info:        Dict[str, Any]  # must NOT contain a "reward" key


class StateResponse(BaseModel):
    """
    GET /state — full internal (ground-truth) state.
    Exposes true severity + demand (not masked), unlike /observation.
    """
    task_id:             str
    step_count:          int
    max_steps:           int
    done:                bool
    available_resources: Dict[str, float]
    zones:               List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status:          str
    version:         str
    active_sessions: int
    sessions:        Dict[str, Dict[str, Any]]


class SessionInfo(BaseModel):
    task_id:     str
    difficulty:  str
    done:        bool
    step_count:  int
    age_seconds: float


# ---------------------------------------------------------------------------
# Helper — parse Action from ActionRequest
# ---------------------------------------------------------------------------

def _parse_action(req: ActionRequest) -> Action:
    try:
        action_type = ActionType(req.action_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown action_type '{req.action_type}'. "
                "Must be one of: request_info, allocate_resource, finalize."
            ),
        )

    resource_type: Optional[ResourceType] = None
    if req.resource_type is not None:
        try:
            resource_type = ResourceType(req.resource_type)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown resource_type '{req.resource_type}'. "
                    "Must be one of: food, water, medicine."
                ),
            )

    action = Action(
        action_type=action_type,
        zone_id=req.zone_id,
        resource_type=resource_type,
        amount=req.amount,
    )

    valid, err = action.validate()
    if not valid:
        raise HTTPException(status_code=422, detail=f"Invalid action: {err}")

    return action


def _strip_reward_from_info(info: dict) -> dict:
    """Ensure 'reward' never leaks into the info dict (OpenEnv compliance)."""
    return {k: v for k, v in info.items() if k != "reward"}


def _clamp_reward(reward: float) -> float:
    """Clamp terminal reward strictly between (0, 1) for compliance."""
    return round(float(np.clip(reward, 0.01, 0.99)), 3)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Server liveness check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """Returns server status and a summary of all active sessions."""
    session_info = {}
    for tid, env in _sessions.items():
        session_info[tid] = {
            "difficulty":  env.difficulty.value,
            "done":        env.done,
            "step_count":  env.step_count,
            "num_zones":   env.config.num_zones,
            "age_seconds": round(time.time() - _created_at.get(tid, 0), 1),
        }

    return HealthResponse(
        status="ok",
        version="1.0.0",
        active_sessions=len(_sessions),
        sessions=session_info,
    )


@app.post(
    "/reset",
    response_model=ResetResponse,
    summary="Reset (or create) an environment session",
    tags=["Environment"],
)
async def reset(request: Request) -> ResetResponse:
    """
    STRICT COMPLIANCE: Uses raw Request to bypass Pydantic body validation.
    Handles missing or empty bodies by defaulting to 'easy'.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id    = body.get("task_id", "easy")
    difficulty = body.get("difficulty", task_id)
    seed       = body.get("seed")

    try:
        diff_enum = Difficulty(str(difficulty).lower())
    except (ValueError, AttributeError):
        diff_enum = Difficulty.EASY

    env = DisasterTriageEnv(difficulty=diff_enum, seed=seed)
    obs = env.reset(seed=seed)

    _sessions[task_id]   = env
    _created_at[task_id] = time.time()

    return ResetResponse(
        task_id=task_id,
        observation=obs.to_dict(),
        done=False,
        info={
            "message":    "Environment reset successfully.",
            "task_id":    task_id,
            "difficulty": diff_enum.value,
            "seed":       seed,
        },
    )


@app.post(
    "/step",
    response_model=StepResponse,
    summary="Execute one action in the environment",
    tags=["Environment"],
)
async def step(request: Request) -> StepResponse:
    """
    STRICT COMPLIANCE: Uses raw Request to bypass Pydantic body validation.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id = body.get("task_id", "easy")
    env     = _get_session(task_id)

    # Use default action if body is missing/empty
    try:
        req = ActionRequest(**body)
    except Exception:
        req = ActionRequest(task_id=task_id, action_type="request_info", zone_id="Z0")

    action = _parse_action(req)
    obs, reward, done, info = env.step(action)

    # All rewards (including intermediate) must be strictly in (0, 1)
    final_reward = _clamp_reward(reward)
    clean_info   = _strip_reward_from_info(info)

    return StepResponse(
        task_id=task_id,
        observation=obs.to_dict(),
        reward=final_reward,
        done=done,
        info=clean_info,
    )


@app.get(
    "/state",
    response_model=StateResponse,
    summary="Full internal ground-truth state (not masked)",
    tags=["Environment"],
)
async def state(task_id: str = "medium") -> StateResponse:
    """
    Returns the **full internal state** of the environment — including
    true severity and demand for every zone (not hidden/masked).

    This is NOT the agent's observation. Use `/step` response for that.

    Fields returned per zone:
    - `zone_id`, `true_severity`, `true_demand`
    - `allocated` (resources deployed so far)
    - `urgency_signal` (noisy float 0–1, always visible)
    - `revealed` (whether the agent has called request_info)
    - `known_severity` (-1 if not revealed)
    - `known_demand` (null if not revealed)
    """
    env = _get_session(task_id)

    # Delegate to env.get_full_state() which explicitly iterates self.zones
    # and pulls true_severity + true_demand for every zone.
    full = env.get_full_state()

    return StateResponse(
        task_id=task_id,
        step_count=full["step_count"],
        max_steps=full["max_steps"],
        done=full["done"],
        available_resources=full["available_resources"],
        zones=full["zones"],
    )


@app.get(
    "/explain",
    summary="[Optional] Grader breakdown of the current allocation",
    tags=["Diagnostic"],
)
async def explain(task_id: str = "medium") -> Dict[str, Any]:
    """
    **Optional diagnostic — not required by OpenEnv.**

    Returns a per-zone grader breakdown showing how the current allocation
    would score across the three axes (prioritization, efficiency, utilization).

    Safe to ignore; does not advance the environment.
    """
    env = _get_session(task_id)
    report = env.grader.explain(
        zones=env.zones,
        initial_resources=env._initial_resources,
        available_resources=env.available_resources,
    )
    return {
        "task_id":    task_id,
        "step_count": env.step_count,
        "note":       "Diagnostic only — not part of OpenEnv spec.",
        **report,
    }


@app.get(
    "/sessions",
    summary="List all active sessions",
    tags=["System"],
)
async def list_sessions() -> Dict[str, Any]:
    """Lists all task_ids currently tracked by the server."""
    return {
        "active_sessions": len(_sessions),
        "sessions": [
            SessionInfo(
                task_id=tid,
                difficulty=env.difficulty.value,
                done=env.done,
                step_count=env.step_count,
                age_seconds=round(time.time() - _created_at.get(tid, 0), 1),
            ).model_dump()
            for tid, env in _sessions.items()
        ],
    }


@app.delete(
    "/session/{task_id}",
    summary="Destroy a session",
    tags=["System"],
)
async def delete_session(
    task_id: str = Path(..., description="Session to delete"),
) -> Dict[str, str]:
    """Removes the session from the store and frees memory."""
    if task_id in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete built-in session '{task_id}'.",
        )
    if task_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{task_id}' not found.")
    del _sessions[task_id]
    _created_at.pop(task_id, None)
    return {"message": f"Session '{task_id}' deleted."}

if __name__ == "__main__":
    # Unified launch: FastAPI + Gradio
    port = int(os.getenv("PORT", "7860"))
    app = gr.mount_gradio_app(app, demo, path="/") # Mount UI at root
    uvicorn.run(app, host="0.0.0.0", port=port)
