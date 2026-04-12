#!/bin/bash
# start.sh
# Unified Startup for Disaster Triage Console (API + UI on Port 7860)

echo "Starting Unified Disaster Triage Server (API + Dashboard)..."

# 1. Start the unified FastAPI + Gradio server on the public port
# Hugging Face provides the PORT environment variable (default 7860).
python -m server.app
