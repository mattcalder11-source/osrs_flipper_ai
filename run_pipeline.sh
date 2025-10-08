#!/bin/bash
# ==============================================
# OSRS AI Flipper Automation Script
# ==============================================

PROJECT_DIR="/osrs_flipper_ai"
VENV="$PROJECT_DIR/.venv/bin/activate"
LOG_DIR="$PROJECT_DIR/logs"

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

# Activate virtual environment
source "$VENV"
cd "$PROJECT_DIR"

timestamp=$(date +"%Y-%m-%d %H:%M:%S")

echo "[$timestamp] 🚀 Starting pipeline..." >> "$LOG_DIR/pipeline.log"

# 1️⃣ Ingest latest data
python src/ingest.py >> "$LOG_DIR/ingest.log" 2>&1

# 2️⃣ If "train" passed as first argument, train the model
if [ "$1" = "train" ]; then
  python models/train_model.py >> "$LOG_DIR/train.log" 2>&1
fi

# 3️⃣ Predict flips
python models/predict_flips.py >> "$LOG_DIR/predict.log" 2>&1

# 4️⃣ Recommend sells
python models/recommend_sell.py >> "$LOG_DIR/recommend.log" 2>&1

echo "[$timestamp] ✅ Pipeline finished." >> "$LOG_DIR/pipeline.log"
