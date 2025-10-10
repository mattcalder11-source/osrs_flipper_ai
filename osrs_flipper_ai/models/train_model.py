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
from sklearn.preprocessing import RobustScaler

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
    """Train GradientBoostingRegressor with outlier filtering and robust scaling."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["mid_price"], how="any")

    # -----------------------------------------------------------------
    # Define target: expected relative return (future sell vs buy)
    # -----------------------------------------------------------------
    if "high" in df.columns and "low" in df.columns:
        df["target_profit_ratio"] = (df["high"] / df["low"]) - 1
        df["target_profit_ratio"] = df["target_profit_ratio"].clip(-0.2, 2.0)
    else:
        raise ValueError("❌ Missing 'high'/'low' columns — cannot compute target_profit_ratio.")

    # -----------------------------------------------------------------
    # Outlier filtering for cleaner targets
    # -----------------------------------------------------------------
    before = len(df)
    df = df[df["mid_price"] > 50]
    df = df[df["daily_volume"] > 0]
    for col in ["target_profit_ratio", "volatility_1h", "daily_volume", "mid_price"]:
        if col in df.columns:
            low, high = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=low, upper=high)
    after = len(df)
    print(f"🧹 Outlier filter: {before} → {after} rows (removed {before - after})")

    # -----------------------------------------------------------------
    # Select training features
    # -----------------------------------------------------------------
    preferred_cols = [
        "spread_ratio",
        "volatility_1h",
        "rsi_norm",
        "roc_norm",
        "macd_norm",
        "technical_score",
    ]
    available = [c for c in preferred_cols if c in df.columns and df[c].notna().sum() > 0]
    if not available:
        raise ValueError("❌ No usable feature columns found for training (all missing/empty).")
    df = df.dropna(subset=available)
    print(f"✅ Using {len(df):,} rows with features: {available}")

    if len(df) < 20:
        raise ValueError(f"❌ Too few rows for training ({len(df)}). Check your feature generation step.")

    X = df[available].copy()
    y = df["target_profit_ratio"]

    # Log-scale heavy-tailed features
    for col in X.columns:
        if (X[col] > 0).all():
            X[col] = np.log1p(X[col])

    # Robust scaling (less sensitive to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds) if len(y_test) > 1 else 0.0
    mae = mean_absolute_error(y_test, preds) if len(y_test) > 1 else 0.0
    print(f"✅ Training complete — R²={r2:.4f}, MAE={mae:.6f}")

    # Log feature importances
    importances = pd.Series(model.feature_importances_, index=available).sort_values(ascending=False)
    print("🏗️ Feature importances:")
    print(importances.round(4))

    # Return packaged model
    model_bundle = {
        "model": model,
        "features": available,
        "scaler": scaler,
        "metadata": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
            "r2": r2,
            "mae": mae,
            "feature_importances": importances.to_dict(),
        },
        "mode": "ratio",  # Ensures predict_flips interprets output correctly
    }

    return model_bundle

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting model training...")
    df = load_latest_features()
    print(f"📦 Loaded {len(df):,} rows of precomputed features")

    try:
        model_bundle = train_model(df)
    except ValueError as e:
        print(f"❌ Training aborted: {e}")
        exit(1)

    timestamp = model_bundle["metadata"]["timestamp"]
    out_path = MODEL_DIR / f"model_{timestamp}.pkl"
    latest_path = MODEL_DIR / "latest_model.pkl"

    joblib.dump(model_bundle, out_path)
    joblib.dump(model_bundle, latest_path)

    print(f"💾 Saved model → {out_path}")
    print("✅ Training pipeline complete.")
