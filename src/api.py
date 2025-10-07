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

REFRESH_INTERVAL = 300  # 5 minutes

app = FastAPI(title="OSRS Flipper AI API", version="2.2")

# In-memory cache
CACHE = {
    "model": None,
    "model_features": None,
    "model_mtime": None,
    "predictions": None,
    "pred_mtime": None,
    "features": None,
    "features_mtime": None,
    "last_refresh": None,
}

# --- Utility Functions ---
def get_mtime(path: str):
    """Return modification time or None."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    except FileNotFoundError:
        return None


def safe_load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found.")
    model_dict = joblib.load(MODEL_PATH)

    # Support models saved as { "model": estimator, "features": list[str] }
    if isinstance(model_dict, dict):
        model = model_dict.get("model", model_dict)
        features = model_dict.get("features", None)
    else:
        model = model_dict
        features = None

    CACHE["model_mtime"] = get_mtime(MODEL_PATH)
    CACHE["model_features"] = features
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


# --- File Watcher for Auto-Reload ---
class ReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        path = event.src_path
        if MODEL_PATH in path and path.endswith(".pkl"):
            CACHE["model"] = safe_load_model()
        elif LATEST_PRED_FILE in path and path.endswith(".csv"):
            CACHE["predictions"] = safe_load_predictions()
        elif FEATURES_FILE in path and path.endswith(".parquet"):
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


# --- Background Refresh Thread ---
def refresh_predictions_periodically():
    """Refresh predictions every REFRESH_INTERVAL seconds."""
    while True:
        time.sleep(REFRESH_INTERVAL)
        try:
            if os.path.exists(LATEST_PRED_FILE):
                new_mtime = get_mtime(LATEST_PRED_FILE)
                if new_mtime != CACHE["pred_mtime"]:
                    CACHE["predictions"] = safe_load_predictions()
                    CACHE["last_refresh"] = datetime.utcnow().isoformat()
                    print("🔁 Predictions auto-refreshed.")
        except Exception as e:
            print(f"⚠ Prediction refresh failed: {e}")


# --- Startup ---
@app.on_event("startup")
def startup_event():
    """Initial load and start background services."""
    try:
        CACHE["model"] = safe_load_model()
        CACHE["predictions"] = safe_load_predictions()
        CACHE["features"] = safe_load_features()
        CACHE["last_refresh"] = datetime.utcnow().isoformat()
        start_watcher()

        threading.Thread(target=refresh_predictions_periodically, daemon=True).start()
        print("🧠 Background refresh thread running (every 5 min).")

    except Exception as e:
        print(f"⚠ Startup load error: {e}")


# --- API Endpoints ---
@app.get("/")
def root():
    return {"status": "ok", "message": "OSRS Flipper AI API is running with auto-reload + 5m refresh."}


@app.get("/status")
def status():
    """Return current system load status and timestamps."""
    return {
        "status": "ok",
        "model_loaded": CACHE["model"] is not None,
        "model_last_reload": CACHE["model_mtime"],
        "predictions_loaded": CACHE["predictions"] is not None,
        "predictions_last_reload": CACHE["pred_mtime"],
        "features_loaded": CACHE["features"] is not None,
        "features_last_reload": CACHE["features_mtime"],
        "last_auto_refresh": CACHE["last_refresh"],
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/top_flips")
def get_top_flips(limit: int = 10):
    """Return top N predicted flips (default = 10)."""
    if CACHE["predictions"] is None:
        raise HTTPException(status_code=503, detail="Predictions not loaded.")

    df = CACHE["predictions"].head(limit)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df.to_dict(orient="records")


@app.get("/predict_item/{item_id}")
def predict_item(item_id: int):
    """Predict flip margin for a specific item using the cached model."""
    if CACHE["model"] is None or CACHE["features"] is None:
        raise HTTPException(status_code=503, detail="Model or features not loaded.")

    df = CACHE["features"]
    if item_id not in df["item_id"].values:
        raise HTTPException(status_code=404, detail=f"Item ID {item_id} not found in features dataset.")

    feature_cols = CACHE.get("model_features")
    if not feature_cols:
        # fallback to known default if metadata missing
        feature_cols = [
            "spread", "mid_price", "spread_ratio", "momentum",
            "potential_profit", "margin_pct", "hour",
            "liquidity_1h", "volatility_1h",
            "rsi", "roc", "macd", "technical_score"
        ]

    item_features = df[df["item_id"] == item_id][feature_cols].iloc[0]
    item_features = item_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    try:
        pred = float(CACHE["model"].predict([item_features])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "item_id": int(item_id),
        "predicted_margin": pred,
        "timestamp": datetime.utcnow().isoformat(),
    }
