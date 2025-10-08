#!/bin/bash
# ==============================================
# OSRS AI Flipper Automation Script (Package Mode)
# ==============================================

source /osrs_flipper_ai/.venv/bin/activate

set -e
set -o pipefail

PROJECT_DIR="/osrs_flipper_ai"
VENV="$PROJECT_DIR/.venv/bin/activate"
LOG_DIR="$PROJECT_DIR/logs"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

mkdir -p "$LOG_DIR"

# Activate venv and enter project root
source "$VENV"
cd "$PROJECT_DIR"

# Ensure Python sees package root
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Logging helpers
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" | tee -a "$PIPELINE_LOG"; }

GREEN="\033[1;32m"
RED="\033[1;31m"
YELLOW="\033[1;33m"
RESET="\033[0m"

log "🧩 DEBUG: Script args received -> '$*'"

log "=============================================="
log "🚀 ${GREEN}Starting OSRS Flipper Pipeline (Package Mode)${RESET}"
log "=============================================="

START_TIME=$(date +%s)

# ----------------------------------------------
log "🧾 [1/4] Ingesting latest market data..."
if python -m osrs_flipper_ai.data_ingest.ingest >> "$LOG_DIR/ingest.log" 2>&1; then
  log "✅ Ingestion completed successfully."
else
  log "${RED}❌ Ingestion failed! Check $LOG_DIR/ingest.log${RESET}"
  exit 1
fi

# ----------------------------------------------
if [ "$1" = "train" ]; then
  log "🧠 [2/4] Training model..."
  if python -m osrs_flipper_ai.models.train_model >> "$LOG_DIR/train.log" 2>&1; then
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
if python -m osrs_flipper_ai.models.predict_flips >> "$LOG_DIR/predict.log" 2>&1; then
  log "✅ Flip prediction step completed."
else
  log "${RED}❌ Prediction failed! Check $LOG_DIR/predict.log${RESET}"
  exit 1
fi

# ----------------------------------------------
log "💰 [4/4] Recommending sells..."
if python -m osrs_flipper_ai.models.recommend_sell >> "$LOG_DIR/recommend.log" 2>&1; then
  log "✅ Sell recommendations complete."
else
  log "${RED}❌ Recommend-sell step failed! Check $LOG_DIR/recommend.log${RESET}"
  exit 1
fi

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

log "----------------------------------------------"
log "🏁 ${GREEN}Pipeline finished successfully in ${RUNTIME}s${RESET}"
log "----------------------------------------------"
