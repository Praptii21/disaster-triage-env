#!/bin/bash
# start.sh

# 1. Start the Unified Server (FastAPI + Gradio)
# HF will provide the PORT env var (7860)
echo "Starting Unified Triage Server on port ${PORT:-7860}..."
python server/app.py &

# 2. Wait for the server to be ready
echo "Waiting for unified server to start..."
while ! curl -s http://127.0.0.1:${PORT:-7860}/health > /dev/null; do
  sleep 1
done
echo "Unified Server is UP!"

# 3. Run the initial baseline inference (Background)
# This will log the mission results to the HF console.
echo "Running baseline mission..."
python inference.py &

# 4. Keep the container alive
wait
