import os
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = BASE_DIR / "data" / "features"
MODEL_DIR = BASE_DIR / "osrs_flipper_ai" / "models" / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# LOAD FEATURES (no recomputation)
# ---------------------------------------------------------------------
def load_latest_features():
    """Load the latest feature snapshot (historical or live)."""
    feature_files = sorted(FEATURE_DIR.glob("features_*.parquet"), key=os.path.getmtime, reverse=True)
    if not feature_files:
        raise FileNotFoundError(f"❌ No feature snapshots found in {FEATURE_DIR}")
    latest_file = feature_files[0]
    print(f"📊 Using feature file: {latest_file}")
    return pd.read_parquet(latest_file)

# ---------------------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------------------
def train_model(df: pd.DataFrame):
    """Train a GradientBoostingRegressor using precomputed features only."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["mid_price"], how="any")

    # Feature columns used for model input
    feature_cols = [
        "spread_ratio",
        "volatility_1h",
        "rsi_norm",
        "roc_norm",
        "macd_norm",
        "technical_score",
    ]
    available = [c for c in feature_cols if c in df.columns]

    if not available:
        raise ValueError("❌ No required feature columns found in dataset.")

    df = df.dropna(subset=available)

    X = df[available]
    y = df["spread_ratio"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print(f"✅ Training complete — R²={r2:.4f}, MAE={mae:.6f}")
    return model, available, r2, mae

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting model training...")
    df = load_latest_features()
    print(f"📦 Loaded {len(df):,} rows of precomputed features")

    model, features, r2, mae = train_model(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_dict = {
        "model": model,
        "features": features,
        "r2": r2,
        "mae": mae,
        "timestamp": timestamp,
    }

    model_path = MODEL_DIR / "latest_model.pkl"
    joblib.dump(model_dict, model_path)
    print(f"💾 Saved model → {model_path}")
