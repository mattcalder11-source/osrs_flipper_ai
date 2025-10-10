"""
predict_flips.py — Generate OSRS flip predictions using the latest trained model.
Includes:
- Item blacklist (by name or ID)
- Daily volume / buy limit ratio filtering
- Correct buy/sell price fields for dashboard
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Fix relative imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from osrs_flipper_ai.models.recommend_sell import batch_recommend_sell
from osrs_flipper_ai.src.fetch_latest_prices import fetch_latest_prices_dict

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "osrs_flipper_ai" / "models" / "trained_models"
FEATURE_DIR = BASE_DIR / "data" / "features"
PRED_DIR = BASE_DIR / "osrs_flipper_ai" / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# ITEM BLACKLIST
# ---------------------------------------------------------------------
ITEM_BLACKLIST = {  
}

BLACKLIST_FILE = BASE_DIR / "osrs_flipper_ai" / "data" / "item_blacklist.txt"
if BLACKLIST_FILE.exists():
    with open(BLACKLIST_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ITEM_BLACKLIST.add(line)
    print(f"🧾 Loaded {len(ITEM_BLACKLIST)} blacklisted items total.")

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def _compute_sane_prices(preds: np.ndarray, mid: np.ndarray):
    """Interpret model output into safe predicted sell prices."""
    mid = np.array(mid, dtype=float)
    preds = np.array(preds, dtype=float)

    # Detect type of output (ratio vs percentage)
    if np.nanmax(np.abs(preds)) > 10:
        profit_pct = preds  # already %
        profit_gp = (preds / 100.0) * mid
    else:
        profit_pct = 100 * preds
        profit_gp = preds * mid

    # Clamp extreme predictions
    profit_pct = np.clip(np.nan_to_num(profit_pct), -99, 2000)
    profit_gp = np.nan_to_num((profit_pct / 100.0) * mid)
    sell_price = np.maximum(mid + profit_gp, 0)

    return sell_price, profit_gp, profit_pct

def load_latest_model():
    model_path = MODEL_DIR / "latest_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"❌ No trained model found at {model_path}")
    model_dict = joblib.load(model_path)
    print(f"📦 Loaded model: {model_path}")
    print(f"   Trained {model_dict.get('timestamp', 'unknown')} (R²={model_dict.get('r2', 0.0):.4f})")
    return model_dict

def load_latest_features():
    files = sorted(FEATURE_DIR.glob("features_*.parquet"), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError("❌ No feature snapshots found in data/features/")
    path = files[0]
    print(f"📊 Using snapshot: {path}")
    return pd.read_parquet(path)

# ---------------------------------------------------------------------
# PREDICT FLIPS
# ---------------------------------------------------------------------
def predict_flips(model_dict, df, top_n=100, min_volume_ratio=2.0):
    """Apply model and return flip predictions for top items."""
    before = len(df)
    df = df[~df["name"].isin(ITEM_BLACKLIST) & ~df["item_id"].isin(ITEM_BLACKLIST)]
    if len(df) != before:
        print(f"🧹 Filtered {before - len(df)} blacklisted items.")

    model = model_dict["model"]
    features = model_dict.get("features", [])
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    preds = model.predict(X)
    df["predicted_margin"] = preds

    # Compute safe prices
    sell_price, profit_gp, profit_pct = _compute_sane_prices(preds, df["mid_price"])
    df["buy_price"] = df["mid_price"]
    df["sell_price"] = sell_price
    df["predicted_profit_gp"] = profit_gp
    df["profit_pct"] = profit_pct

    # Load GE limits and compute volume ratio
    if "daily_volume" in df.columns:
        try:
            limits = pd.read_html("https://oldschool.runescape.wiki/w/Grand_Exchange/Buying_limits")[0]
            limits.columns = limits.columns.str.lower().str.strip()
            limit_map = dict(zip(limits["item"], limits["limit"]))
            df["limit"] = df["name"].map(limit_map).fillna(100)
            df["volume_ratio"] = df["daily_volume"].astype(float) / df["limit"].replace(0, np.nan)
            df = df[df["volume_ratio"] >= float(min_volume_ratio)]
            print(f"💧 Filtered by volume_ratio ≥ {min_volume_ratio}: {before} → {len(df)}")
        except Exception as e:
            print(f"⚠️ Could not load GE limits: {e}")
            df["limit"], df["volume_ratio"] = 100, 0
    else:
        df["limit"], df["volume_ratio"] = 100, 0

    # Clean and rank
    df = df[df["profit_pct"].between(-99, 2000)]
    ranked = df.sort_values("predicted_profit_gp", ascending=False).head(top_n)

    # Save outputs for dashboard
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    timestamped = PRED_DIR / f"top_flips_{ts}.csv"
    latest = PRED_DIR / "latest_top_flips.csv"
    out_cols = [
        "item_id", "name", "buy_price", "sell_price", "predicted_profit_gp",
        "profit_pct", "daily_volume", "limit", "volume_ratio", "volatility_1h"
    ]
    ranked.to_csv(timestamped, columns=[c for c in out_cols if c in ranked.columns], index=False)
    ranked.to_csv(latest, columns=[c for c in out_cols if c in ranked.columns], index=False)
    print(f"💰 Saved top {len(ranked)} flips → {timestamped}")

    return ranked

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting flip prediction pipeline...")
    model_dict = load_latest_model()
    df = load_latest_features()
    top_flips = predict_flips(model_dict, df, top_n=100)
    print(f"🔍 DEBUG: predict_flips() returned {len(top_flips)} rows")

    latest_prices = fetch_latest_prices_dict()
    top_flips["entry_price"] = top_flips.get("buy_price", top_flips["mid_price"])
    sell_recs = batch_recommend_sell(top_flips, latest_prices)
    sell_recs.to_csv(PRED_DIR / "sell_signals.csv", index=False)
