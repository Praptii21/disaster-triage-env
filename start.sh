#!/bin/bash
# start.sh

# 1. Start the FastAPI server in the background
# We use & to put it in the background. uvicorn will use the PORT set by HF (7860).
echo "Starting FastAPI server..."
uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860} &

# 2. Wait for the server to be ready
echo "Waiting for server to start..."
while ! curl -s http://127.0.0.1:${PORT:-7860}/health > /dev/null; do
  sleep 1
done
echo "Server is UP!"

# 3. Run the inference script
# This will print the [START], [STEP], and [END] logs to the Space's stdout.
echo "Running inference..."
python inference.py

# 4. Keep the container alive after inference finishes
# (Hugging Face expects the main process to keep running to keep the Space 'Running')
wait
