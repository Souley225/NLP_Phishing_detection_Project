#!/bin/bash
# PhishGuard startup script
# Starts FastAPI (port 8000) then Streamlit (port 7860)

set -e

export PYTHONPATH=/app
export PYTHONUNBUFFERED=1

echo "=== Starting PhishGuard services ==="

# Start FastAPI backend in background
echo "[1/2] Starting FastAPI on port 8000..."
python -m uvicorn src.serving.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 &
API_PID=$!
echo "      FastAPI PID: $API_PID"

# Give the API time to load the model
echo "      Waiting 5s for API to initialize..."
sleep 5

# Start Streamlit in foreground (keeps container alive)
echo "[2/2] Starting Streamlit on port 7860..."
python -m streamlit run src/ui/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false

# If Streamlit exits, stop FastAPI too
kill $API_PID 2>/dev/null || true
