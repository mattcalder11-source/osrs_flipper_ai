"""
predict_flips.py
Predict profitable OSRS flips using a trained ML model and precomputed features.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from osrs_flipper_ai.models.recommend_sell import batch_recommend_sell
from osrs_flipper_ai.src.fetch_latest_prices import save_latest_prices

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
BASE_DIR = "/root/osrs_flipper_ai/osrs_flipper_ai"
FEATURES_DIR = "/root/osrs_flipper_ai/data/features"
MODEL_DIR = f"{BASE_DIR}/models/trained_models"
PREDICTIONS_DIR = f"{BASE_DIR}/data/predictions"
LIMITS_FILE = f"{BASE_DIR}/data/ge_limits.json"
ITEM_BLACKLIST_FILE = f"{BASE_DIR}/data/blacklist.txt"
MIN_VOLUME_RATIO = 2 #dily_volume-to-buy_limit ratio threshold

os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# LOAD MODEL & FEATURES
# ---------------------------------------------------------------------
def load_latest_model():
    """Load the most recent trained model from MODEL_DIR."""
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"❌ Model directory not found: {MODEL_DIR}")

    model_files = [
        f for f in os.listdir(MODEL_DIR)
        if f.endswith(".pkl") and "model" in f
    ]
    if not model_files:
        raise FileNotFoundError("❌ No trained models found in trained_models/")

    latest_model_file = max(model_files, key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)))
    model_path = os.path.join(MODEL_DIR, latest_model_file)

    model_dict = joblib.load(model_path)
    timestamp = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y%m%d_%H%M")

    print(f"📦 Loaded model: {model_path}")
    print(f"   Trained {timestamp} (R²={model_dict.get('r2', 0):.4f})")

    model_dict["path"] = model_path
    model_dict["timestamp"] = timestamp
    return model_dict


def load_latest_features():
    """Load the most recent features parquet file from FEATURES_DIR."""
    if not os.path.exists(FEATURES_DIR):
        raise FileNotFoundError(f"❌ Feature directory not found: {FEATURES_DIR}")

    files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError("❌ No feature snapshots found in data/features/")

    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(FEATURES_DIR, f)))
    feature_path = os.path.join(FEATURES_DIR, latest_file)

    print(f"📊 Loaded features: {feature_path}")
    df = pd.read_parquet(feature_path)
    return df


# ---------------------------------------------------------------------
# LOAD ITEM BLACKLIST
# ---------------------------------------------------------------------
def load_blacklist():
    if not os.path.exists(ITEM_BLACKLIST_FILE):
        return set()
    with open(ITEM_BLACKLIST_FILE, "r") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    return set(lines)


ITEM_BLACKLIST = load_blacklist()
print(f"🧱 Loaded blacklist ({len(ITEM_BLACKLIST)} items): {sorted(list(ITEM_BLACKLIST))[:5]}...")


# ---------------------------------------------------------------------
# ADAPTIVE PRICE INTERPRETATION
# ---------------------------------------------------------------------
def _compute_sane_prices(preds: np.ndarray, mid: np.ndarray):
    """Interpret model predictions robustly, converting to realistic sell prices."""
    preds = np.nan_to_num(np.array(preds, dtype=float))
    mid = np.nan_to_num(np.array(mid, dtype=float))
    mid[mid <= 0] = np.nan  # avoid divide by zero

    # Detect type of model output
    if np.nanmax(np.abs(preds)) > 10:
        interpretation = "percent"
        profit_pct = preds
        profit_gp = (profit_pct / 100.0) * mid
    elif np.nanmax(np.abs(preds)) < 5:
        interpretation = "ratio"
        profit_pct = 100.0 * (preds - 1)
        profit_gp = (preds - 1) * mid
    else:
        interpretation = "absolute"
        profit_gp = preds
        profit_pct = 100.0 * (profit_gp / mid)

    profit_pct = np.clip(np.nan_to_num(profit_pct), -99, 2000)
    profit_gp = np.nan_to_num((profit_pct / 100.0) * mid)
    sell_price = np.maximum(mid + profit_gp, 1.0)

    diagnostics = f"{interpretation} mode (max pred={np.nanmax(preds):.2f})"
    print(f"⚙️ Interpreting model output as {diagnostics}")

    return sell_price, profit_gp, profit_pct


# ---------------------------------------------------------------------
# MAIN FLIP PREDICTION FUNCTION
# ---------------------------------------------------------------------
def predict_flips(model_dict, df, top_n=100):
    """Generate flip predictions and filter by blacklist and liquidity."""

    model = model_dict["model"]
    features = model_dict["features"]

    df = df.dropna(subset=features)
    preds = model.predict(df[features])

    sell_price, profit_gp, profit_pct = _compute_sane_prices(preds, df["mid_price"])
    df["buy_price"] = df["mid_price"].clip(lower=1)
    df["sell_price"] = sell_price
    df["predicted_profit_gp"] = profit_gp
    df["profit_pct"] = profit_pct

    # Apply blacklist
    before = len(df)
    df = df[~df["name"].isin(ITEM_BLACKLIST)]
    print(f"🚫 Blacklist filter: {before} → {len(df)} rows")

    # Load buy limits
    if os.path.exists(LIMITS_FILE):
        with open(LIMITS_FILE, "r") as f:
            limits = json.load(f)
        limits_df = pd.DataFrame(list(limits.items()), columns=["item_id", "buy_limit"])
        limits_df["item_id"] = limits_df["item_id"].astype(int)
        df = df.merge(limits_df, on="item_id", how="left")
    else:
        df["buy_limit"] = np.nan
        print("⚠️ No GE limit file found!")

    # Volume-to-limit ratio filter
    if "daily_volume" in df.columns:
        before = len(df)
        df["vol_to_limit_ratio"] = df["daily_volume"] / df["buy_limit"].replace(0, np.nan)
        df = df[df["vol_to_limit_ratio"] >= MIN_VOLUME_RATIO]
        print(f"💧 Volume-to-limit ratio ≥ {MIN_VOLUME_RATIO}: {before} → {len(df)} rows")
    else:
        print("⚠️ No daily_volume column found — skipping liquidity filter.")

    # Drop unrealistic predictions
    df = df[df["profit_pct"].between(-50, 1000)]

    # Rank and select top flips
    df = df.sort_values("predicted_profit_gp", ascending=False)
    top_flips = df.head(top_n).copy()

    # Apply sell recommendation logic
    latest_prices_dict = save_latest_prices()  # fetch + persist
    top_flips = batch_recommend_sell(top_flips, latest_prices_dict)
    top_flips = batch_recommend_sell(top_flips)

    return top_flips


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting flip prediction pipeline...")

    model_dict = load_latest_model()
    df = load_latest_features()

    top_flips = predict_flips(model_dict, df, top_n=100)

    latest_csv = os.path.join(PREDICTIONS_DIR, "latest_top_flips.csv")
    top_flips.to_csv(latest_csv, index=False)
    print(f"💾 Saved predictions → {latest_csv}")
    print(f"✅ {len(top_flips)} flips ready for dashboard.")
