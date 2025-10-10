#!/bin/bash
# ==============================================
# OSRS AI Flipper Full Pipeline (Feature → Train → Predict → Recommend)
# ==============================================

set -e
set -o pipefail

PROJECT_DIR="/root/osrs_flipper_ai"
VENV="$PROJECT_DIR/venv/bin/activate"
LOG_DIR="$PROJECT_DIR/logs"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

mkdir -p "$LOG_DIR"
source "$VENV"
cd "$PROJECT_DIR"

# Ensure Python can see modules under osrs_flipper_ai/
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" | tee -a "$PIPELINE_LOG"; }

GREEN="\033[1;32m"
RED="\033[1;31m"
YELLOW="\033[1;33m"
RESET="\033[0m"

log "=============================================="
log "🚀 ${GREEN}Starting OSRS AI Flipper Pipeline${RESET}"
log "=============================================="

START_TIME=$(date +%s)

# ----------------------------------------------
log "🧾 [1/5] Ingesting latest market data..."
if python -m osrs_flipper_ai.data_ingest.ingest >> "$LOG_DIR/ingest.log" 2>&1; then
  log "✅ Ingestion complete."
else
  log "${RED}❌ Ingestion failed! Check $LOG_DIR/ingest.log${RESET}"
  exit 1
fi

# ----------------------------------------------
log "🧠 [2/5] Computing features..."
if python -m osrs_flipper_ai.features.features >> "$LOG_DIR/features.log" 2>&1; then
  log "✅ Feature computation complete."
else
  log "${RED}❌ Feature computation failed! Check $LOG_DIR/features.log${RESET}"
  exit 1
fi

# ----------------------------------------------
if [ "$1" = "train" ]; then
  log "🧩 [3/5] Training model..."
  if python -m osrs_flipper_ai.models.train_model >> "$LOG_DIR/train.log" 2>&1; then
    log "✅ Model training complete."
  else
    log "${RED}❌ Model training failed! Check $LOG_DIR/train.log${RESET}"
    exit 1
  fi
else
  log "⏭️  [3/5] Skipping model training (no 'train' arg provided)."
fi

# ----------------------------------------------
log "📊 [4/5] Predicting top flips..."
if python -m osrs_flipper_ai.models.predict_flips >> "$LOG_DIR/predict.log" 2>&1; then
  log "✅ Flip prediction complete."
else
  log "${RED}❌ Prediction failed! Check $LOG_DIR/predict.log${RESET}"
  exit 1
fi

# ----------------------------------------------
log "💰 [5/5] Generating sell recommendations..."
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
