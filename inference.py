"""
inference.py
============
Hackathon inference script for DisasterTriageEnv.

Connects an LLM agent (via OpenAI-compatible API) to the running
DisasterTriage FastAPI server and runs one full episode.

Logging output (stdout only):
  [START] task=<task_name> env=disaster-triage-env model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2...rn>

Usage:
  python inference.py [--task easy|medium|hard] [--base-url http://127.0.0.1:8000]

Environment variables (loaded from .env):
  API_BASE_URL  — LLM API base URL  (default: https://api.openai.com/v1)
  MODEL_NAME    — model identifier   (default: gpt-4.1-mini)
  HF_TOKEN      — API key / HF token (REQUIRED — raises error if missing)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN:     str | None = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("[END] success=false steps=0 rewards=")
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set. "
        "Add it to your .env file or export it before running."
    )

# ---------------------------------------------------------------------------
# OpenAI client (OpenAI-compatible, pointed at API_BASE_URL)
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# Server interaction helpers
# ---------------------------------------------------------------------------

ENV_SERVER_URL: str = "http://127.0.0.1:8000"   # local FastAPI server


def _post(path: str, payload: dict) -> dict:
    resp = requests.post(f"{ENV_SERVER_URL}{path}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _get(path: str, params: Optional[dict] = None) -> dict:
    resp = requests.get(f"{ENV_SERVER_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def reset_env(task_id: str, difficulty: str, seed: Optional[int] = None) -> dict:
    return _post("/reset", {"task_id": task_id, "difficulty": difficulty, "seed": seed})


def step_env(task_id: str, action: dict) -> dict:
    return _post("/step", {"task_id": task_id, **action})


def get_state(task_id: str) -> dict:
    return _get("/state", params={"task_id": task_id})


# ---------------------------------------------------------------------------
# LLM agent helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a disaster triage coordinator. Your job is to allocate limited \
resources (food, water, medicine) across disaster zones to maximise survival.

Each step you must output ONE JSON action object. Valid formats:
  {"action_type": "request_info",     "zone_id": "Z0"}
  {"action_type": "allocate_resource","zone_id": "Z0","resource_type":"food","amount":30.0}
  {"action_type": "finalize"}

Rules:
- Prioritise high-severity zones (severity 4–5) with ≥60% of their demand.
- Keep low-severity zones (severity 1–2) below 20% of demand.
- Call finalize when done or if resources are low.
- Output ONLY the JSON object, nothing else.
"""


def _build_user_message(obs: dict, step_n: int) -> str:
    """Summarise the current observation for the LLM."""
    zones_summary = []
    for z in obs.get("zones", []):
        entry = (
            f"  {z['zone_id']}: urgency={z['urgency_signal']:.2f}"
            f"  revealed={z['revealed']}"
            f"  known_severity={z['known_severity']}"
        )
        if z.get("known_demand"):
            d = z["known_demand"]
            entry += (
                f"  demand=(food={d['food']:.1f}, water={d['water']:.1f},"
                f" medicine={d['medicine']:.1f})"
            )
        entry += f"  allocated={z['allocated']}"
        zones_summary.append(entry)

    res = obs.get("available_resources", {})
    return (
        f"Step {step_n} / {obs.get('max_steps', '?')}\n"
        f"Available resources: food={res.get('food', 0):.1f}  "
        f"water={res.get('water', 0):.1f}  medicine={res.get('medicine', 0):.1f}\n"
        f"Zones:\n" + "\n".join(zones_summary)
    )


def get_llm_action(history: List[dict], obs: dict, step_n: int) -> dict:
    """Call the LLM and parse its JSON action."""
    history.append({"role": "user", "content": _build_user_message(obs, step_n)})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        temperature=0.2,
        max_tokens=128,
    )

    raw = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": raw})

    # Parse JSON — strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    action = json.loads(raw.strip())
    return action


# ---------------------------------------------------------------------------
# Logging — exact hackathon format
# ---------------------------------------------------------------------------

def log_start(task_name: str, model: str) -> None:
    print(f"[START] task={task_name} env=disaster-triage-env model={model}", flush=True)


def log_step(
    step_n: int,
    action: dict,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    action_str = json.dumps(action, separators=(",", ":"))
    done_str   = "true" if done else "false"
    error_str  = "null" if error is None else error
    print(
        f"[STEP] step={step_n} action={action_str} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    total_reward = sum(rewards)
    score = min(max(total_reward, 0.0), 1.0)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

def run_episode(task_id: str, difficulty: str, seed: Optional[int] = None) -> None:
    rewards: List[float] = []
    steps:   int         = 0
    success: bool        = False

    log_start(task_name=task_id, model=MODEL_NAME)

    try:
        # ── Reset ────────────────────────────────────────────────────────────
        reset_resp = reset_env(task_id=task_id, difficulty=difficulty, seed=seed)
        obs:  dict = reset_resp["observation"]
        done: bool = reset_resp.get("done", False)

        history: List[dict] = []

        # ── Episode loop ─────────────────────────────────────────────────────
        while not done:
            steps += 1
            error_msg: Optional[str] = None
            action: dict = {}

            try:
                action = get_llm_action(history, obs, steps)
            except Exception as llm_err:
                error_msg = f"llm_error:{llm_err}"
                # Fall back to finalize so the episode ends cleanly
                action = {"action_type": "finalize"}

            try:
                step_resp = step_env(task_id=task_id, action=action)
                obs    = step_resp["observation"]
                reward = float(step_resp.get("reward", 0.0))
                done   = bool(step_resp.get("done", False))
            except Exception as env_err:
                reward = 0.0
                done   = True
                if error_msg is None:
                    error_msg = f"env_error:{env_err}"

            rewards.append(reward)
            log_step(steps, action, reward, done, error_msg)

        # ── Determine success ────────────────────────────────────────────────
        # Success = episode completed normally (done=True) with a positive
        # terminal reward (last reward > 0).
        success = done and bool(rewards) and rewards[-1] > 0.0

    except Exception as fatal:
        # Always emit [END] even on fatal errors
        error_line = f"fatal_error:{fatal}"
        log_step(steps + 1, {}, 0.0, True, error_line)

    finally:
        log_end(success=success, steps=steps, rewards=rewards)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM inference episodes on DisasterTriageEnv.\n"
            "Runs ALL tasks (easy, medium, hard) by default when no --task is given."
        )
    )
    parser.add_argument(
        "--task",
        default=None,                            # None = run all three
        choices=["easy", "medium", "hard"],
        help="Single task to run. Omit to run all: easy, medium, hard.",
    )
    parser.add_argument(
        "--base-url",
        default=ENV_SERVER_URL,
        help=f"DisasterTriage server URL (default: {ENV_SERVER_URL})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (applied to all tasks when running all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Allow overriding the server URL at runtime
    ENV_SERVER_URL = args.base_url  # type: ignore[assignment]

    # Run all tasks or just the one requested
    # Req 7: `python inference.py` with no args runs easy → medium → hard
    tasks_to_run = [args.task] if args.task else ["easy", "medium", "hard"]

    for task in tasks_to_run:
        run_episode(
            task_id=task,
            difficulty=task,
            seed=args.seed,
        )
