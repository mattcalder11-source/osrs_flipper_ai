# src/api.py
import os
import pandas as pd
import joblib
import threading
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

MODEL_PATH = "models/latest_model.pkl"
LATEST_PRED_FILE = "data/predictions/latest_top_flips.csv"
FEATURES_FILE = "data/features/latest_train_enriched.parquet"

app = FastAPI(title="OSRS Flipper AI API", version="2.1")

# In-memory cache
CACHE = {
    "model": None,
    "model_mtime": None,
    "predictions": None,
    "pred_mtime": None,
    "features": None,
    "features_mtime": None,
}


# --- Helper Functions ---
def get_mtime(path: str):
    """Return modification time or None."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    except FileNotFoundError:
        return None


def safe_load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found.")
    model = joblib.load(MODEL_PATH)
    CACHE["model_mtime"] = get_mtime(MODEL_PATH)
    print(f"✅ Model reloaded from {MODEL_PATH}")
    return model


def safe_load_predictions():
    if not os.path.exists(LATEST_PRED_FILE):
        raise FileNotFoundError("Predictions file not found.")
    df = pd.read_csv(LATEST_PRED_FILE)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    CACHE["pred_mtime"] = get_mtime(LATEST_PRED_FILE)
    print(f"✅ Predictions reloaded ({len(df)} rows)")
    return df


def safe_load_features():
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError("Feature file not found.")
    df = pd.read_parquet(FEATURES_FILE)
    CACHE["features_mtime"] = get_mtime(FEATURES_FILE)
    print(f"✅ Features reloaded ({len(df)} rows)")
    return df


# --- File Watcher ---
class ReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        path = event.src_path
        if path.endswith(".pkl") and MODEL_PATH in path:
            CACHE["model"] = safe_load_model()
        elif path.endswith(".csv") and LATEST_PRED_FILE in path:
            CACHE["predictions"] = safe_load_predictions()
        elif path.endswith(".parquet") and FEATURES_FILE in path:
            CACHE["features"] = safe_load_features()


def start_watcher():
    observer = Observer()
    handler = ReloadHandler()
    observer.schedule(handler, "models", recursive=False)
    observer.schedule(handler, "data/predictions", recursive=False)
    observer.schedule(handler, "data/features", recursive=False)
    observer.daemon = True
    observer.start()
    print("👀 File watcher started — automatic reload active.")


# --- API Endpoints ---
@app.on_event("startup")
def startup_event():
    """Initialize cache and file watcher on startup."""
    try:
        CACHE["model"] = safe_load_model()
        CACHE["predictions"] = safe_load_predictions()
        CACHE["features"] = safe_load_features()
        start_watcher()
    except Exception as e:
        print(f"⚠ Startup load error: {e}")


@app.get("/")
def root():
    return {"status": "ok", "message": "OSRS Flipper AI API is running with auto-reload."}


@app.get("/status")
def status():
    """Return the current load status and timestamps."""
    return {
        "status": "ok",
        "model_loaded": CACHE["model"] is not None,
        "model_last_reload": CACHE["model_mtime"],
        "predictions_loaded": CACHE["predictions"] is not None,
        "predictions_last_reload": CACHE["pred_mtime"],
        "features_loaded": CACHE["features"] is not None,
        "features_last_reload": CACHE["features_mtime"],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/top_flips")
def get_top_flips(limit: int = 50):
    """Return top N predicted flips."""
    if CACHE["predictions"] is None:
        raise HTTPException(status_code=503, detail="Predictions not loaded.")
    df = CACHE["predictions"].head(limit)
    return df.to_dict(orient="records")


@app.get("/predict_item/{item_id}")
def predict_item(item_id: int):
    """Predict flip margin for a specific item."""
    if CACHE["model"] is None or CACHE["features"] is None:
        raise HTTPException(status_code=503, detail="Model or features not loaded.")

    df = CACHE["features"]
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

    item_features = df[df["item_id"] == item_id][feature_cols].iloc[0]
    item_features = item_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    pred = float(CACHE["model"].predict([item_features])[0])

    return {
        "item_id": int(item_id),
        "predicted_margin": pred,
        "timestamp": datetime.utcnow().isoformat(),
    }
