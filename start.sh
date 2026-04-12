#!/bin/bash
# start.sh
# Optimized for Hugging Face Spaces

# Hugging Face usually provides the PORT via variable, defaulting to 7860
PORT_NUM=${PORT:-7860}

echo "Starting FastAPI server on port $PORT_NUM..."
# Start the server in the background
uvicorn server.app:app --host 0.0.0.0 --port $PORT_NUM &
SERVER_PID=$!

echo "Waiting for server to be healthy..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:$PORT_NUM/health > /dev/null; then
    echo "Server is UP!"
    break
  fi
  echo "Retry $i/30: waiting for health check..."
  sleep 2
done

echo "Running environment inference..."
# Explicitly point to the local port
python inference.py --base-url http://127.0.0.1:$PORT_NUM

echo "Inference routine complete. Keeping server alive for logs."
# Keeps the container 'Running' as long as the uvicorn server is alive
wait $SERVER_PID
