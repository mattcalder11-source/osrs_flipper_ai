#!/bin/bash
# Auto-run top flips prediction every 5 minutes

cd /root/osrs_flipper_ai
source .venv/bin/activate

while true; do
    echo "🚀 Running top 10 flip generation..."
    python src/train_model.py >> logs/auto_predict.log 2>&1
    echo "✅ Done. Sleeping for 5 minutes..."
    sleep 300  # 5 minutes
done
