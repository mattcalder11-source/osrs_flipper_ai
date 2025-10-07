# src/train_model.py
import os
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Import feature helpers (must be in src/features.py)
from features import compute_features, compute_technical_indicators

# Paths
DATA_PATH = "data/features/latest_train.parquet"
MODEL_DIR = "models"
LOG_PATH = "logs/train_metrics.csv"
PRED_DIR = "data/predictions"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


def load_data_and_ensure_features():
    """
    Loads feature data, ensures all required columns exist, computes missing features,
    and caches the enriched dataset for faster future training.
    Cache auto-refreshes every 24 hours.
    """

    BASE_DATA_PATH = DATA_PATH
    CACHE_PATH = "data/features/latest_train_enriched.parquet"
    REFRESH_INTERVAL_HOURS = 24  # <-- configurable refresh window

    # --- Check if cache exists and is still fresh ---
    if os.path.exists(CACHE_PATH):
        cache_age_hours = (time.time() - os.path.getmtime(CACHE_PATH)) / 3600
        if cache_age_hours < REFRESH_INTERVAL_HOURS:
            print(f"⚡ Using cached enriched dataset (age: {cache_age_hours:.2f}h): {CACHE_PATH}")
            df = pd.read_parquet(CACHE_PATH)
            print(f"📦 Cached dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        else:
            print(f"🕒 Cache is {cache_age_hours:.1f}h old — refreshing now...")

    # --- Load base feature file ---
    if not os.path.exists(BASE_DATA_PATH):
        raise FileNotFoundError(f"❌ Feature file not found: {BASE_DATA_PATH}")

    df = pd.read_parquet(BASE_DATA_PATH)
    print(f"📦 Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Ensure required columns
    required_cols = ["ts_utc", "item_id", "low", "high"]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"❌ Required column '{col}' missing from dataset.")

    # --- Compute rolling averages if missing ---
    if any(col not in df.columns for col in ["avg_5m_low", "avg_5m_high", "avg_1h_low"]):
        print("⚙️ Computing missing rolling averages (avg_5m_low, avg_5m_high, avg_1h_low)...")
        df = df.sort_values(["item_id", "ts_utc"])
        df["avg_5m_low"] = df.groupby("item_id")["low"].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df["avg_5m_high"] = df.groupby("item_id")["high"].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df["avg_1h_low"] = df.groupby("item_id")["low"].transform(
            lambda x: x.rolling(window=12, min_periods=1).mean()
        )
        print("✅ Rolling averages computed successfully.")

    # --- Compute engineered features ---
    try:
        df = compute_features(df)
        print("✅ compute_features() applied successfully.")
    except Exception as e:
        print(f"⚠️ compute_features() failed: {e}")

    # --- Compute technical indicators ---
    tech_cols = ["rsi", "roc", "macd", "macd_signal", "contrarian_flag", "technical_score"]
    missing_tech = [c for c in tech_cols if c not in df.columns]

    if missing_tech:
        print(f"⚙️ Computing missing technical indicators: {missing_tech}")
        try:
            df = compute_technical_indicators(df)
            print("✅ compute_technical_indicators() applied successfully.")
        except Exception as e:
            print(f"⚠️ Failed to compute technical indicators: {e}")

    # --- Clean and save ---
    df = df.dropna(subset=["margin_pct"])
    df = df[df["low"] > 0]

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)

    print(f"💾 Cached enriched dataset saved: {CACHE_PATH}")
    print("⏳ Next refresh in 24 hours.")
    return df


def train_model(df):
    # Define candidate feature columns (base + tech)
    base_features = [
        "spread", "mid_price", "spread_ratio", "momentum",
        "potential_profit", "hour", "liquidity_1h", "volatility_1h"
    ]
    tech_features = ["rsi", "roc", "macd", "macd_signal", "contrarian_flag", "technical_score"]

    feature_cols = [c for c in (base_features + tech_features) if c in df.columns]
    if len(feature_cols) < 5:
        raise RuntimeError(f"Too few features available for training: {feature_cols}")

    X = df[feature_cols].fillna(0)
    y = df["margin_pct"].fillna(0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        random_state=42
    )

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"✅ Model trained — MAE: {mae:.6f}, R²: {r2:.4f}")
    return model, mae, r2, feature_cols


def save_model_and_log(model, mae, r2, feature_cols):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")
    latest_path = os.path.join(MODEL_DIR, "latest_model.pkl")

    # Save model and the list of features used (so predict stage can align)
    joblib.dump({"model": model, "features": feature_cols}, model_path)
    joblib.dump({"model": model, "features": feature_cols}, latest_path)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mae": mae,
        "r2": r2,
        "model_path": model_path,
        "n_features": len(feature_cols)
    }

    if not os.path.exists(LOG_PATH):
        pd.DataFrame([row]).to_csv(LOG_PATH, index=False)
    else:
        pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=False, index=False)

    print(f"💾 Model saved: {model_path}")
    print(f"📊 Metrics logged: {LOG_PATH}")

    return model_path


def predict_top_flips_from_model_dict(model_dict, df, top_n=50):
    model = model_dict["model"]
    features = model_dict["features"]

    # Ensure all features exist in df
    for c in features:
        if c not in df.columns:
            df[c] = 0

    X = df[features].fillna(0)
    df = df.copy()
    df["predicted_margin"] = model.predict(X)
    df["predicted_profit_gp"] = df["predicted_margin"] * df["mid_price"]

    top_flips = (
        df.sort_values("predicted_profit_gp", ascending=False)
        .head(top_n)
        .loc[:, ["item_id", "name", "predicted_profit_gp", "predicted_margin", "mid_price", "liquidity_1h", "volatility_1h"] +
               [c for c in ["rsi", "roc", "macd", "technical_score"] if c in df.columns]]
    )

    timestamp = datetime.now().strftime("%Y%m%d")
    pred_file = os.path.join(PRED_DIR, f"top_flips_{timestamp}.csv")
    latest_file = os.path.join(PRED_DIR, "latest_top_flips.csv")
    top_flips.to_csv(pred_file, index=False)
    top_flips.to_csv(latest_file, index=False)
    print(f"💰 Top {len(top_flips)} flips saved: {pred_file}")
    return top_flips


if __name__ == "__main__":
    try:
        df = load_data_and_ensure_features()
        model, mae, r2, feature_cols = train_model(df)
        model_path = save_model_and_log(model, mae, r2, feature_cols)
        # reload the saved dict to ensure consistency for prediction
        model_dict = joblib.load(os.path.join(MODEL_DIR, "latest_model.pkl"))
        top = predict_top_flips_from_model_dict(model_dict, df, top_n=50)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
