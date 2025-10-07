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

# Import feature helpers
from features import compute_features, compute_technical_indicators


# ---------------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------------
FEATURE_COLS = [
    "spread", "mid_price", "spread_ratio", "momentum",
    "potential_profit", "hour",
    "liquidity_1h", "volatility_1h",
    "rsi", "roc", "macd", "macd_signal",
    "contrarian_flag", "technical_score"
]
TARGET_COL = "margin_pct"

DATA_PATH = "data/features/latest_train.parquet"
MODEL_DIR = "models"
LOG_PATH = "logs/train_metrics.csv"
PRED_DIR = "data/predictions"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# LOAD DATA AND ENSURE FEATURE CONSISTENCY
# ---------------------------------------------------------------------
def load_data_and_ensure_features():
    BASE_DATA_PATH = DATA_PATH
    CACHE_PATH = "data/features/latest_train_enriched.parquet"
    REFRESH_INTERVAL_HOURS = 24

    if os.path.exists(CACHE_PATH):
        cache_age_hours = (time.time() - os.path.getmtime(CACHE_PATH)) / 3600
        if cache_age_hours < REFRESH_INTERVAL_HOURS:
            print(f"⚡ Using cached enriched dataset ({cache_age_hours:.2f}h old): {CACHE_PATH}")
            df = pd.read_parquet(CACHE_PATH)
            print(f"📦 Cached dataset loaded: {df.shape[0]} rows, {df.shape[1]} cols")
            return df
        else:
            print(f"🕒 Cache is {cache_age_hours:.1f}h old — refreshing...")

    if not os.path.exists(BASE_DATA_PATH):
        raise FileNotFoundError(f"❌ Feature file not found: {BASE_DATA_PATH}")

    df = pd.read_parquet(BASE_DATA_PATH)
    print(f"📦 Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} cols")

    # Validate minimal required columns
    for col in ["ts_utc", "item_id", "low", "high"]:
        if col not in df.columns:
            raise RuntimeError(f"❌ Required column '{col}' missing from dataset.")

    # Compute missing rolling averages
    if any(col not in df.columns for col in ["avg_5m_low", "avg_5m_high", "avg_1h_low"]):
        print("⚙️ Computing missing rolling averages...")
        df = df.sort_values(["item_id", "ts_utc"])
        df["avg_5m_low"] = df.groupby("item_id")["low"].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df["avg_5m_high"] = df.groupby("item_id")["high"].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df["avg_1h_low"] = df.groupby("item_id")["low"].transform(lambda x: x.rolling(12, min_periods=1).mean())

    # Compute engineered + technical features
    try:
        df = compute_features(df)
        print("✅ compute_features() applied successfully.")
    except Exception as e:
        print(f"⚠️ compute_features() failed: {e}")

    try:
        df = compute_technical_indicators(df)
        print("✅ compute_technical_indicators() applied successfully.")
    except Exception as e:
        print(f"⚠️ Failed to compute technical indicators: {e}")

    # Drop bad data
    df = df.dropna(subset=[TARGET_COL])
    df = df[df["low"] > 0]

    # Ensure all training features exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"⚠ Missing feature '{col}', filling with 0.")
            df[col] = 0

    # Save cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    print(f"💾 Cached enriched dataset saved: {CACHE_PATH}")
    return df


# ---------------------------------------------------------------------
# TRAIN MODEL — with robust NaN / inf handling
# ---------------------------------------------------------------------
def train_model(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COL])

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Median-fill NaN/inf values
    print("🧹 Cleaning feature matrix...")
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
        if np.isinf(X[col]).any():
            X[col] = np.where(np.isinf(X[col]), X[col].median(), X[col])
        if X[col].isna().all():
            X[col] = 0

    y = y.fillna(y.median())

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
    return model, mae, r2


# ---------------------------------------------------------------------
# SAVE MODEL + LOG METRICS
# ---------------------------------------------------------------------
def save_model_and_log(model, mae, r2):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")
    latest_path = os.path.join(MODEL_DIR, "latest_model.pkl")

    model_info = {"model": model, "features": FEATURE_COLS, "timestamp": timestamp, "mae": mae, "r2": r2}
    joblib.dump(model_info, model_path)
    joblib.dump(model_info, latest_path)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mae": mae,
        "r2": r2,
        "model_path": model_path,
        "n_features": len(FEATURE_COLS)
    }

    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

    print(f"💾 Model saved: {model_path}")
    print(f"📊 Metrics logged: {LOG_PATH}")
    return model_path


# ---------------------------------------------------------------------
# PREDICT TOP FLIPS — with same imputation
# ---------------------------------------------------------------------
def predict_top_flips(model, df, top_n=10):
    print(f"🔮 Generating top {top_n} flip predictions...")

    df = df.copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"⚠ Adding missing feature '{col}' with default 0.")
            df[col] = 0

    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().all():
            X[col] = 0
        else:
            X[col] = X[col].fillna(X[col].median())

    df["predicted_margin"] = model.predict(X)
    df["predicted_profit_gp"] = df["predicted_margin"] * df["mid_price"]

    top_flips = (
        df.sort_values("predicted_profit_gp", ascending=False)
        .head(top_n)
        .loc[:, ["item_id", "name", "predicted_profit_gp", "predicted_margin",
                 "mid_price", "liquidity_1h", "volatility_1h", "technical_score"]]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    pred_file = os.path.join(PRED_DIR, f"top_flips_{timestamp}.csv")
    latest_file = os.path.join(PRED_DIR, "latest_top_flips.csv")

    top_flips.to_csv(pred_file, index=False)
    top_flips.to_csv(latest_file, index=False)

    print(f"💰 Top {top_n} flips saved: {pred_file}")
    return top_flips


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        df = load_data_and_ensure_features()
        model, mae, r2 = train_model(df)
        save_model_and_log(model, mae, r2)
        model_dict = joblib.load(os.path.join(MODEL_DIR, "latest_model.pkl"))
        top = predict_top_flips(model_dict["model"], df, top_n=10)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
