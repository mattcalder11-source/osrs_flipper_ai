# src/api.py (fixed JSON-safe version)

import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from datetime import datetime
import numpy as np
import traceback

MODEL_PATH = "models/latest_model.pkl"
LATEST_PRED_FILE = "data/predictions/latest_top_flips.csv"
FEATURES_FILE = "data/features/latest_train_enriched.parquet"

app = FastAPI(title="OSRS Flipper AI API", version="1.0")


@app.get("/")
def root():
    return {"status": "ok", "message": "OSRS Flipper AI API is running."}


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model file not found.")
    return joblib.load(MODEL_PATH)


@app.get("/top_flips")
def get_top_flips(limit: int = 50):
    """Return top N predicted flips (default: 50)."""
    if not os.path.exists(LATEST_PRED_FILE):
        raise HTTPException(status_code=404, detail="No prediction file found yet.")
    try:
        df = pd.read_csv(LATEST_PRED_FILE).head(limit)
        # Sanitize bad float values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        return df.to_dict(orient="records")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict_item/{item_id}")
def predict_item(item_id: int):
    """Return model prediction for a specific item_id."""
    if not os.path.exists(FEATURES_FILE):
        raise HTTPException(status_code=404, detail="No feature data available.")
    try:
        df = pd.read_parquet(FEATURES_FILE)
        model = load_model()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    if item_id not in df["item_id"].values:
        raise HTTPException(status_code=404, detail=f"Item ID {item_id} not found.")

    feature_cols = [
        "spread", "mid_price", "spread_ratio", "momentum",
        "potential_profit", "margin_pct", "hour",
        "liquidity_1h", "volatility_1h",
        "rsi", "roc", "macd", "technical_score"
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing columns: {missing}")

    try:
        item_features = df[df["item_id"] == item_id][feature_cols].iloc[0]
        item_features = item_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        pred = float(model.predict([item_features])[0])
        return {
            "item_id": int(item_id),
            "predicted_margin": pred,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
