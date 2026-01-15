#!/bin/bash
# Start FastAPI inference service
# Usage: bash scripts/start_service.sh <checkpoint> [port]

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/start_service.sh <checkpoint> [port]"
    echo "Example: bash scripts/start_service.sh runs/paraformer_full/checkpoints/best.pt 8000"
    exit 1
fi

CHECKPOINT=$1
PORT=${2:-8000}

echo "Starting Paraformer ASR Service"
echo "================================"
echo "Checkpoint: $CHECKPOINT"
echo "Port: $PORT"
echo ""
echo "Service will be available at: http://localhost:$PORT"
echo "API docs at: http://localhost:$PORT/docs"
echo ""

python scripts/serve.py \
    --checkpoint "$CHECKPOINT" \
    --port "$PORT"
