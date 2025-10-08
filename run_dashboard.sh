#!/bin/bash
echo "🚀 Starting OSRS AI Flipper Dashboard..."

cd "$(dirname "$0")"

# Start backend
echo "🧠 Launching backend..."
(uvicorn backend.main:app --reload --port 8000 &) 

# Start frontend
echo "💻 Launching frontend..."
(cd frontend && npm run dev)

trap "echo '🛑 Stopping...'; kill 0" EXIT
