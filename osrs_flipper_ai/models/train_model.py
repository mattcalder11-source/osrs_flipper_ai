# src/train_model.py
import os
import glob
import time
import math
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from osrs_flipper_ai.features.features import compute_features, compute_technical_indicators

print(f"\n=== 🧠 TRAIN MODEL RUN START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
FEATURE_COLS = [
    "spread", "mid_price", "spread_ratio", "momentum",
    "potential_profit", "hour",
    "liquidity_1h", "volatility_1h",
    "rsi", "roc", "macd", "macd_signal",
    "contrarian_flag", "technical_score"
]
TARGET_COL = "net_margin_pct"

FEATURE_DIR = "data/features"
MODEL_DIR = "models"
LOG_PATH = "logs/train_metrics.csv"
PRED_DIR = "data/predictions"

RETENTION_DAYS = 28          # 🧹 keep snapshots < 28 days old
WEIGHT_DECAY_RATE = 0.5      # how fast weights decay with age
HISTORY_DAYS_BACK = 10        # how many days of data to load into model training

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ---------------------------------------------------------------------
# CLEANUP: DELETE OLD FEATURE FILES
# ---------------------------------------------------------------------
def cleanup_old_features(retention_days=28):
    """Deletes feature parquet files older than `retention_days`."""
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

    if deleted > 0:
        print(f"🧹 Cleaned up {deleted} old feature files (> {retention_days}d).")
    else:
        print(f"🧼 No old feature files found (within {retention_days}d).")


# ---------------------------------------------------------------------
# LOAD + MERGE FEATURES WITH TIME-DECAY WEIGHTING
# ---------------------------------------------------------------------
def load_recent_features(days_back=28, decay_rate=0.5):
    """Merge recent feature snapshots and apply time-decay weights.
    - Skips and deletes empty/broken Parquet files automatically.
    - Guarantees schema consistency and safe weighting.
    """
    cutoff = time.time() - days_back * 86400
    feature_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "features_*.parquet")))

    if not feature_files:
        raise FileNotFoundError(f"❌ No feature files found in {FEATURE_DIR}")

    valid_files = [f for f in feature_files if os.path.getmtime(f) >= cutoff]
    if not valid_files:
        valid_files = [feature_files[-1]]  # fallback to most recent file

    print(f"📦 Loading {len(valid_files)} feature snapshots (last {days_back} days)...")

    dfs, broken_files, deleted_files = [], [], []

    for f in valid_files:
        try:
            df = pd.read_parquet(f)

            # 🧩 Detect empty or schema-broken files
            if df.empty or "item_id" not in df.columns:
                print(f"⚠️ Broken feature file detected: {f} — deleting.")
                os.remove(f)
                deleted_files.append(f)
                continue

            df["source_file"] = os.path.basename(f)
            df["file_time"] = datetime.fromtimestamp(os.path.getmtime(f))
            dfs.append(df)

        except Exception as e:
            print(f"⚠️ Skipped {f} due to error: {e}")
            broken_files.append(f)
            try:
                os.remove(f)
                deleted_files.append(f)
            except Exception as del_err:
                print(f"⚠️ Could not delete {f}: {del_err}")

    if not dfs:
        raise RuntimeError("❌ No valid feature data loaded after cleanup.")

    # 🔧 Logging cleanup summary
    if deleted_files:
        print(f"🧹 Removed {len(deleted_files)} corrupted or empty files:")
        for f in deleted_files:
            print(f"   • {os.path.basename(f)}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["item_id", "ts_utc"], keep="last")

    # ------------------------------------------------------------------
    # 🧩 Guarantee file_time exists (older feature files may lack it)
    # ------------------------------------------------------------------
    if "file_time" not in combined.columns or combined["file_time"].isna().all():
        print("⚠️ Missing or invalid 'file_time' metadata — inferring from filenames...")
        combined["file_time"] = combined.get("source_file", "")
        combined["file_time"] = combined["file_time"].str.extract(r"(\d{8}_\d{6})")
        combined["file_time"] = pd.to_datetime(combined["file_time"], errors="coerce")
        combined["file_time"] = combined["file_time"].fillna(datetime.utcnow())

    # ------------------------------------------------------------------
    # Compute engineered + technical indicators
    # ------------------------------------------------------------------
    combined = compute_features(combined)
    combined = compute_technical_indicators(combined)

    # Compute net margin (GE tax + slippage)
    if TARGET_COL not in combined.columns:
        print("⚙️ Computing net_margin_pct (after GE tax + slippage)...")
        combined["ge_tax"] = np.floor(0.02 * combined["high"])
        combined.loc[combined["ge_tax"] > 5_000_000, "ge_tax"] = 5_000_000
        combined.loc[combined["high"] < 50, "ge_tax"] = 0

        if "liquidity_1h" in combined.columns:
            combined["slippage_rate"] = np.where(
                combined["liquidity_1h"] > 100, 0.002,
                np.where(combined["liquidity_1h"] > 10, 0.005, 0.01)
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

    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(subset=[TARGET_COL])
    combined = combined[combined["low"] > 0]

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in combined.columns:
            combined[col] = 0

    # ------------------------------------------------------------------
    # ⏳ Apply exponential time-decay weighting
    # ------------------------------------------------------------------
    now = datetime.utcnow()
    try:
        combined["age_days"] = (now - combined["file_time"]).dt.total_seconds() / 86400
    except Exception as e:
        print(f"⚠️ Failed computing age_days: {e} — defaulting to 0.")
        combined["age_days"] = 0

    combined["sample_weight"] = np.exp(-decay_rate * combined["age_days"])
    combined["sample_weight"] = combined["sample_weight"].clip(lower=0.1, upper=1.0)

    # ------------------------------------------------------------------
    # ✅ Final output
    # ------------------------------------------------------------------
    combined.to_parquet("data/features/latest_train_merged.parquet", index=False)
    print(f"✅ Combined {len(combined):,} samples (time-weighted).")
    return combined


# ---------------------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------------------
def train_model(df):
    """Train model using sample-weighted regression to emphasize recent data."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COL])

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    weights = df.get("sample_weight", pd.Series(1.0, index=df.index))

    # Clean up NaN or infinite values
    print("🧹 Cleaning feature matrix...")
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
        if np.isinf(X[col]).any():
            X[col] = np.where(np.isinf(X[col]), X[col].median(), X[col])
        if X[col].isna().all():
            X[col] = 0

    y = y.fillna(y.median())
    weights = weights.fillna(1.0).clip(lower=0.1, upper=1.0)

    # Split with stratified weighting
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        random_state=42
    )

    print("🚀 Training model (with recency weighting)...")
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"✅ Weighted model trained — MAE: {mae:.6f}, R²: {r2:.4f}")
    print(f"🧭 Weight summary — mean: {weights.mean():.3f}, min: {weights.min():.3f}, max: {weights.max():.3f}")

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
    return model_path

#---------------------------------------------------------------------
# RECENT TRAINING LOGGING
#---------------------------------------------------------------------
def summarize_recent_training(log_path=LOG_PATH, lookback_hours=48):
    """Summarize model performance metrics and trends."""
    if not os.path.exists(log_path):
        print("📭 No past logs found — first run, skipping summary.")
        return

    df = pd.read_csv(log_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.now() - pd.Timedelta(hours=lookback_hours)
    recent = df[df["timestamp"] > cutoff]

    if recent.empty:
        print(f"⚠ No recent logs (last {lookback_hours}h).")
        return

    avg_mae = recent["mae"].mean()
    avg_r2 = recent["r2"].mean()
    latest = recent.iloc[-1]

    print("\n📈 === MODEL PERFORMANCE SUMMARY ===")
    print(f"🕒 Window: last {lookback_hours} hours ({len(recent)} runs)")
    print(f"📊 Avg MAE: {avg_mae:.6f}")
    print(f"📈 Avg R²: {avg_r2:.4f}")
    print(f"📘 Latest Run: {latest['timestamp']} | MAE={latest['mae']:.6f} | R²={latest['r2']:.4f}")
    print("===================================")

#---------------------------------------------------------------------
# FLIP RECOMMENDATION TRACKING
#---------------------------------------------------------------------
def summarize_flip_recommendations(pred_dir="data/predictions", lookback_runs=5):
    """Compare recent top flip files to see how recommendations are evolving."""
    import glob

    files = sorted(glob.glob(f"{pred_dir}/top_flips_*.csv"), reverse=True)
    if len(files) < 2:
        print("📭 Not enough historical predictions to compare yet.")
        return

    latest = pd.read_csv(files[0])
    prev = pd.read_csv(files[1])
    latest_ids = set(latest["item_id"])
    prev_ids = set(prev["item_id"])

    overlap = len(latest_ids & prev_ids)
    new_items = latest_ids - prev_ids

    print("\n💰 === FLIP RECOMMENDATION TREND ===")
    print(f"🔁 Overlap with previous run: {overlap}/{len(latest_ids)} items")
    print(f"🆕 New recommendations: {len(new_items)} — {list(new_items)[:5]}")
    print("====================================")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        df = load_recent_features(days_back=28, decay_rate=0.5)
        model, mae, r2 = train_model(df)
        save_model_and_log(model, mae, r2)

        summarize_recent_training()
        summarize_flip_recommendations()

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

