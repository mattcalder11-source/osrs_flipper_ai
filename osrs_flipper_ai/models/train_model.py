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
# LOAD FEATURES
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
    """Train GradientBoostingRegressor using whatever valid features are available."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["mid_price"], how="any")

    # -----------------------------------------------------------------
    # Define target: expected relative return (future sell vs buy)
    # -----------------------------------------------------------------
    if "high" in df.columns and "low" in df.columns:
        df["target_profit_ratio"] = (df["high"] / df["low"]) - 1
        # Clip extremes to remove outliers
        df["target_profit_ratio"] = df["target_profit_ratio"].clip(-0.2, 2.0)
    else:
        raise ValueError("❌ Missing 'high'/'low' columns — cannot compute target_profit_ratio.")


    # Preferred features (but optional)
    preferred_cols = [
        "spread_ratio",
        "volatility_1h",
        "rsi_norm",
        "roc_norm",
        "macd_norm",
        "technical_score",
    ]

    # Select available + non-empty
    available = [c for c in preferred_cols if c in df.columns and df[c].notna().sum() > 0]

    if not available:
        raise ValueError("❌ No usable feature columns found for training (all missing/empty).")

    if len(available) < 2:
        print(f"⚠️ Only one valid feature found: {available}. Training with a single predictor.")
    
    df = df.dropna(subset=available)
    print(f"✅ Using {len(df):,} rows with features: {available}")

    if len(df) < 10:
        raise ValueError(f"❌ Too few rows for training ({len(df)}). Check your feature generation step.")

    X = df[available]

    # Log-scale heavy-tailed numeric features
    for col in X.columns:
        if (X[col] > 0).all():
            X[col] = np.log1p(X[col])


    y = df["target_profit_ratio"]

    # Use 80/20 split if data allows
    if len(df) > 20:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds) if len(y_test) > 1 else 0.0
    mae = mean_absolute_error(y_test, preds) if len(y_test) > 1 else 0.0

    print(f"✅ Training complete — R²={r2:.4f}, MAE={mae:.6f}")
    return model, available, r2, mae


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting model training...")
    df = load_latest_features()
    print(f"📦 Loaded {len(df):,} rows of precomputed features")

    try:
        model, features, r2, mae = train_model(df)
    except ValueError as e:
        print(f"❌ Training aborted: {e}")
        exit(1)

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
