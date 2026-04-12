#!/bin/bash
# start.sh

# 1. Start the FastAPI server in the background on an internal port
echo "Starting FastAPI server on port 8000..."
uvicorn server.app:app --host 0.0.0.0 --port 8000 &

# 2. Wait for the server to be ready
echo "Waiting for server to start..."
while ! curl -s http://127.0.0.1:8000/health > /dev/null; do
  sleep 1
done
echo "Backend Server is UP!"

# 3. Start the Gradio Console UI on the public port (Background)
# Occupying port 7860 immediately prevents HF timeouts.
echo "Starting Gradio Console on public port ${PORT:-7860}..."
python ui.py &

# 4. Run the initial baseline inference (Background)
echo "Running baseline inference..."
python inference.py &

# 5. Keep the container alive
wait
