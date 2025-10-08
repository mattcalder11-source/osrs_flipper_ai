#!/bin/bash
# ==============================================
# OSRS AI Flipper Automation Script
# ==============================================

set -e  # Exit immediately on error
set -o pipefail

PROJECT_DIR="/osrs_flipper_ai"
VENV="$PROJECT_DIR/.venv/bin/activate"
LOG_DIR="$PROJECT_DIR/logs"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

# Activate virtual environment
source "$VENV"
cd "$PROJECT_DIR"

# Helper for timestamps + colored output
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" | tee -a "$PIPELINE_LOG"; }

# ANSI color codes
GREEN="\033[1;32m"
RED="\033[1;31m"
YELLOW="\033[1;33m"
RESET="\033[0m"

# ----------------------------------------------
log "=============================================="
log "🚀 ${GREEN}Starting OSRS Flipper Pipeline${RESET}"
log "=============================================="

START_TIME=$(date +%s)

# ----------------------------------------------
log "🧾 [1/4] Ingesting latest market data..."
if python src/ingest.py >> "$LOG_DIR/ingest.log" 2>&1; then
  log "✅ Ingestion completed successfully."
else
  log "${RED}❌ Ingestion failed! Check $LOG_DIR/ingest.log${RESET}"
  exit 1
fi

# ----------------------------------------------
if [ "$1" = "train" ]; then
  log "🧠 [2/4] Training model..."
  if python models/train_model.py >> "$LOG_DIR/train.log" 2>&1; then
    log "✅ Model training completed."
  else
    log "${RED}❌ Model training failed! Check $LOG_DIR/train.log${RESET}"
    exit 1
  fi
else
  log "⏭️  [2/4] Skipping model training (no 'train' arg provided)."
fi

# ----------------------------------------------
log "📊 [3/4] Generating flip predictions..."
if python models/predict_flips.py >> "$LOG_DIR/predict.log" 2>&1; then
  log "✅ Flip prediction step completed."
else
  log "${RED}❌ Prediction failed! Check $LOG_DIR/predict.log${RESET}"
  exit 1
fi

# ----------------------------------------------
log "💰 [4/4] Recommending sells..."
if python models/recommend_sell.py >> "$LOG_DIR/recommend.log" 2>&1; then
  log "✅ Sell recommendations complete."
else
  log "${RED}❌ Recommend-sell step failed! Check $LOG_DIR/recommend.log${RESET}"
  exit 1
fi

# ----------------------------------------------
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
log "----------------------------------------------"
log "🏁 ${GREEN}Pipeline finished successfully in ${RUNTIME}s${RESET}"
log "----------------------------------------------"
