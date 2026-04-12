import gradio as gr
import requests
import json
import time
import os
from typing import List, Generator

# Import the brain for simulation decisions
try:
    from inference import get_scripted_action, get_llm_action
except Exception as e:
    print(f"[SYSTEM] Error importing inference: {e}")
    def get_scripted_action(obs, step, task): return {"action_type": "finalize"}
    def get_llm_action(hist, obs, step, task): return {"action_type": "finalize"}

# Backend Configuration
# Use internal port 8000 for backend communication on HF
_PORT = os.getenv("BACKEND_PORT", "8000")
API_BASE_URL = f"http://127.0.0.1:{_PORT}"

def run_simulation(task: str) -> Generator:
    logs = [f"> Initializing simulation: {task.upper()}"]
    yield "\n".join(logs)
    
    try:
        # 1. Reset
        resp = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task, "difficulty": task})
        if resp.status_code != 200:
            logs.append(f"ERROR: Reset failed ({resp.status_code})")
            yield "\n".join(logs)
            return
        
        data = resp.json()
        obs = data.get("observation", {})
        done = data.get("done", False)
        history = []
        step = 0
        total_reward = 0
        
        logs.append(f"> Connected to backend at {API_BASE_URL}")
        yield "\n".join(logs)

        # 2. Loop
        while not done:
            step += 1
            
            # Decision
            try:
                # Use a small delay to avoid hitting rate limits too aggressively
                # The user can still control this by restarting or we can hardcode 1.0s
                action = get_llm_action(history, obs, step, task)
            except Exception as e:
                logs.append(f"AGENT_WARN: {str(e)[:50]}... Falling back to scripted.")
                action = get_scripted_action(obs, step, task)

            # Step
            resp = requests.post(f"{API_BASE_URL}/step", json={"task_id": task, **action})
            if resp.status_code != 200:
                logs.append(f"ERROR: Step failed ({resp.status_code})")
                break
                
            res = resp.json()
            obs = res.get("observation", {})
            reward = res.get("reward", 0.0)
            done = res.get("done", False)
            total_reward += reward

            # Log Formatting: step=X action={...} reward=X done=X error=X
            action_json = json.dumps(action, separators=(',', ':'))
            logs.append(f"step={step} action={action_json} reward={reward:.4f} done={str(done).lower()} error=null")
            yield "\n".join(logs)
            
            # Minimum safety delay
            time.sleep(1.0)

        logs.append(f"\n[END] MISSION COMPLETE | steps={step} score={total_reward:.3f}")
        yield "\n".join(logs)

    except Exception as e:
        logs.append(f"fatal_error: {str(e)}")
        yield "\n".join(logs)

# UI Construction with a presentable "Soft" theme
with gr.Blocks(title="Disaster Triage Console", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("# 🚑 Disaster Triage Console")
    gr.Markdown("Mission oversight and real-time environment logs.")
    
    with gr.Row():
        task_input = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Mission Priority")
        start_btn = gr.Button("🚀 Start Simulation", variant="primary")
    
    log_output = gr.Textbox(
        label="Terminal Execution Feed",
        lines=25,
        max_lines=30,
        interactive=False,
        autoscroll=True,
        elem_classes="console-box"
    )

    start_btn.click(
        fn=run_simulation,
        inputs=[task_input],
        outputs=[log_output]
    )

if __name__ == "__main__":
    # HF_PORT is usually 7860
    hf_port = int(os.getenv("PORT", "7860"))
    print(f"Starting Console on port {hf_port}")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=hf_port
    )