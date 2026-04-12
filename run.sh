#!/bin/bash

# Start the FastAPI server in the background
echo "[SYSTEM] Starting environment server on port 7860..."
uvicorn server.main:app --host 0.0.0.0 --port 7860 &

# Wait for the server to be healthy
echo "[SYSTEM] Waiting for server health check..."
MAX_RETRIES=30
RETRY_COUNT=0
until curl -s http://localhost:7860/health | grep -q "ok" || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
  sleep 2
  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "[SYSTEM] Health check retry $RETRY_COUNT..."
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo "[SYSTEM] ERROR: Server failed to start in time. Logs follow:"
  exit 1
fi

echo "[SYSTEM] Server is healthy. Running inference agent..."

# Run the inference script
python inference.py

echo "[SYSTEM] Inference complete. Keeping server alive..."

# Bring the background server process to the foreground
wait
