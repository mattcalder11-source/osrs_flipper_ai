# src/api.py
import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from datetime import datetime

MODEL_PATH = "models/latest_model.pkl"
LATEST_PRED_FILE = "data/predictions/latest_top_flips.csv"
FEATURES_FILE = "data/features/latest_train_enriched.parquet"

app = FastAPI(title="OSRS Flipper AI API", version="1.0")

# --- Root Healthcheck ---
@app.get("/")
def root():
    return {"status": "ok", "message": "OSRS Flipper AI API is running."}

# --- Load model utility ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model file not found.")
    return joblib.load(MODEL_PATH)

# --- Return Top Predicted Flips ---
@app.get("/top_flips")
def get_top_flips(limit: int = 50):
    """Return top N predicted flips (default: 50)."""
    if not os.path.exists(LATEST_PRED_FILE):
        raise HTTPException(status_code=404, detail="No prediction file found yet.")

    try:
        df = pd.read_csv(LATEST_PRED_FILE)
        df = df.head(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading predictions: {e}")

    # Convert to clean JSON-safe output
    return df.to_dict(orient="records")

# --- Predict a single item dynamically ---
@app.get("/predict_item/{item_id}")
def predict_item(item_id: int):
    """
    Return the model's predicted profit margin for a specific item_id.
    """
    if not os.path.exists(FEATURES_FILE):
        raise HTTPException(status_code=404, detail="No feature data available.")
    
    try:
        df = pd.read_parquet(FEATURES_FILE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading features: {e}")

    if item_id not in df["item_id"].values:
        raise HTTPException(status_code=404, detail=f"Item ID {item_id} not found in feature data.")

    # Prepare features for prediction
    feature_cols = [
        "spread", "mid_price", "spread_ratio", "momentum",
        "potential_profit", "margin_pct", "hour",
        "liquidity_1h", "volatility_1h",
        "rsi", "roc", "macd", "technical_score"
    ]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=500, detail=f"Missing columns: {missing_cols}")

    item_features = df[df["item_id"] == item_id][feature_cols].iloc[0]

    try:
        model = load_model()
        predicted_margin = float(model.predict([item_features])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "item_id": int(item_id),
        "predicted_margin": predicted_margin,
        "timestamp": datetime.utcnow().isoformat()
    }
