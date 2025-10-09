# models/train_model.py
import os
import glob
import time
import math
import pandas as pd
import numpy as np
import joblib
import gc
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from osrs_flipper_ai.features.features import compute_features, compute_technical_indicators

print(f"\n=== 🧠 TRAIN MODEL RUN START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
FEATURE_COLS = [
    "spread", "mid_price", "spread_ratio", "momentum",
    "potential_profit", "hour",
    "daily_volume", "volatility_1h",
    "rsi", "roc", "macd", "macd_signal",
    "contrarian_flag", "technical_score"
]
TARGET_COL = "net_margin_pct"

FEATURE_DIR = "data/features"
MODEL_DIR = "models"
LOG_PATH = "logs/train_metrics.csv"
PRED_DIR = "data/predictions"

RETENTION_DAYS = 28
WEIGHT_DECAY_RATE = 0.5
HISTORY_DAYS_BACK = 10

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ---------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------
def cleanup_old_features(retention_days=28):
    now = time.time()
    cutoff = now - retention_days * 86400
    feature_files = glob.glob(os.path.join(FEATURE_DIR, "features_*.parquet"))
    deleted = 0

    for f in feature_files:
        if os.path.getmtime(f) < cutoff:
            try:
                os.remove(f)
                deleted += 1
            except Exception as e:
                print(f"⚠️ Could not delete {f}: {e}")

    print(f"🧹 Cleaned {deleted} old feature files (> {retention_days}d).")


# ---------------------------------------------------------------------
# LOAD FEATURES (in chunks for memory safety)
# ---------------------------------------------------------------------
def load_recent_features(days_back=28, decay_rate=0.5, batch_limit=20):
    cutoff = time.time() - days_back * 86400
    feature_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "features_*.parquet")))

    if not feature_files:
        raise FileNotFoundError(f"❌ No feature files found in {FEATURE_DIR}")

    valid_files = [f for f in feature_files if os.path.getmtime(f) >= cutoff]
    if not valid_files:
        valid_files = [feature_files[-1]]

    print(f"📦 Loading {len(valid_files)} feature snapshots (last {days_back} days)...")

    dfs, deleted_files = [], []
    for f in valid_files[-batch_limit:]:
        try:
            df = pd.read_parquet(f)
            if df.empty or "item_id" not in df.columns:
                os.remove(f)
                deleted_files.append(f)
                continue
            df["source_file"] = os.path.basename(f)
            df["file_time"] = datetime.fromtimestamp(os.path.getmtime(f))
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipped {f}: {e}")
            try:
                os.remove(f)
                deleted_files.append(f)
            except: pass

    if not dfs:
        raise RuntimeError("❌ No valid feature data loaded after cleanup.")

    if deleted_files:
        print(f"🧹 Removed {len(deleted_files)} corrupted files.")

    combined = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()

    subset_cols = [c for c in ["item_id", "timestamp", "ts_utc"] if c in combined.columns]
    combined = combined.drop_duplicates(subset=subset_cols, keep="last")

    # Guarantee file_time
    if "file_time" not in combined.columns or combined["file_time"].isna().all():
        combined["file_time"] = datetime.utcnow()

    # Compute features safely
    combined = compute_features(combined)
    combined = compute_technical_indicators(combined)

    # Target: net_margin_pct
    if TARGET_COL not in combined.columns:
        combined["ge_tax"] = np.floor(0.02 * combined["high"])
        combined.loc[combined["ge_tax"] > 5_000_000, "ge_tax"] = 5_000_000
        combined.loc[combined["high"] < 50, "ge_tax"] = 0

        if "daily_volume" in combined.columns:
            combined["slippage_rate"] = np.where(
                combined["daily_volume"] > 100, 0.002,
                np.where(combined["daily_volume"] > 10, 0.005, 0.01)
            )
        else:
            combined["slippage_rate"] = 0.005

        combined["slippage"] = combined["slippage_rate"] * (
            (combined["high"] + combined["low"]) / 2
        )
        combined["net_margin_gp"] = (
            (combined["high"] - combined["ge_tax"]) - (combined["low"] + combined["slippage"])
        )
        combined["net_margin_pct"] = combined["net_margin_gp"] / (
            (combined["high"] + combined["low"]) / 2
        )

    # Cleanup
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(subset=[TARGET_COL])
    combined = combined[combined["low"] > 0]

    for col in FEATURE_COLS:
        if col not in combined.columns:
            combined[col] = 0

    now = datetime.utcnow()
    combined["age_days"] = (now - combined["file_time"]).dt.total_seconds() / 86400
    combined["sample_weight"] = np.exp(-decay_rate * combined["age_days"]).clip(0.1, 1.0)

    combined.to_parquet("data/features/latest_train_merged.parquet", index=False)
    print(f"✅ Combined {len(combined):,} samples.")
    return combined


# ---------------------------------------------------------------------
# TRAIN MODEL (memory-safe)
# ---------------------------------------------------------------------
def train_model(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COL])

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    weights = df.get("sample_weight", pd.Series(1.0, index=df.index))

    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
        if np.isinf(X[col]).any():
            X[col] = np.where(np.isinf(X[col]), X[col].median(), X[col])

    y = y.fillna(y.median())
    weights = weights.fillna(1.0).clip(0.1, 1.0)

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    print("🚀 Training memory-efficient model (HistGradientBoostingRegressor)...")
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=5,
        l2_regularization=0.1, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"✅ Model trained — MAE: {mae:.6f}, R²: {r2:.4f}")
    del X, y, X_train, X_val, y_train, y_val; gc.collect()
    return model, mae, r2


# ---------------------------------------------------------------------
# SAVE MODEL + LOG
# ---------------------------------------------------------------------
def save_model_and_log(model, mae, r2):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(MODEL_DIR, f"model_{ts}.pkl")
    latest_path = os.path.join(MODEL_DIR, "latest_model.pkl")

    model_info = {"model": model, "features": FEATURE_COLS, "timestamp": ts, "mae": mae, "r2": r2}
    joblib.dump(model_info, model_path)
    joblib.dump(model_info, latest_path)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mae": mae,
        "r2": r2,
        "model_path": model_path,
    }
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)
    print(f"💾 Model saved → {model_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        cleanup_old_features()
        df = load_recent_features(days_back=28, decay_rate=0.5)
        model, mae, r2 = train_model(df)
        save_model_and_log(model, mae, r2)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
