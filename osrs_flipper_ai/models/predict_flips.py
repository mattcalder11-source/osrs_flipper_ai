"""
predict_flips.py - Uses the latest trained model to generate OSRS flip predictions.
Now includes item blacklist filtering (by name or ID).
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Ensure relative imports work
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
# ITEM BLACKLIST (hardcoded + optional file)
# ---------------------------------------------------------------------
ITEM_BLACKLIST = {
    # Common junk/untradables
    "Coins", "Platinum token", "Bond",
    "Twisted bow (inactive)",
    "Clue scroll (beginner)", "Clue scroll (easy)",
    "Clue scroll (medium)", "Clue scroll (hard)", "Clue scroll (elite)",
    "Reward casket", "Scroll box",
    "Uncut opal", "Uncut jade", "Uncut red topaz",
    "Dragonfire shield (uncharged)", "Casket", "Lamp",

    # IDs
    995,   # Coins
    13204, # Platinum token
    13190, # Bond
}

BLACKLIST_FILE = BASE_DIR / "osrs_flipper_ai" / "data" / "item_blacklist.txt"
if BLACKLIST_FILE.exists():
    try:
        with open(BLACKLIST_FILE) as f:
            file_blacklist = {line.strip() for line in f if line.strip()}
            ITEM_BLACKLIST |= file_blacklist
        print(f"🧾 Loaded {len(file_blacklist)} additional blacklisted items from {BLACKLIST_FILE.name}")
    except Exception as e:
        print(f"⚠️ Could not read blacklist file: {e}")

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def _choose_margin_interpretation(preds: np.ndarray, mid: np.ndarray):
    eps = 1e-9
    mid_safe = np.where(mid <= 0, np.nan, mid)

    pct_if_ratio = 100.0 * preds
    pct_if_percent = preds
    pct_if_abs = 100.0 * (preds / (mid_safe + eps))

    def extreme_fraction(arr, threshold=500.0):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e9, neginf=-1e9)
        return float((np.abs(arr) > threshold).sum()) / max(1.0, float(len(arr)))

    thr = 500.0
    f_ratio = extreme_fraction(pct_if_ratio, thr)
    f_percent = extreme_fraction(pct_if_percent, thr)
    f_abs = extreme_fraction(pct_if_abs, thr)

    scores = [
        ("ratio", f_ratio, np.nanmedian(np.abs(pct_if_ratio))),
        ("percent", f_percent, np.nanmedian(np.abs(pct_if_percent))),
        ("absolute", f_abs, np.nanmedian(np.abs(pct_if_abs))),
    ]
    scores.sort(key=lambda x: (x[1], x[2]))
    chosen = scores[0][0]
    return chosen, {
        "ratio_extreme_frac": f_ratio,
        "percent_extreme_frac": f_percent,
        "abs_extreme_frac": f_abs,
        "median_pct_ratio": float(np.nanmedian(pct_if_ratio)),
        "median_pct_percent": float(np.nanmedian(pct_if_percent)),
        "median_pct_abs": float(np.nanmedian(pct_if_abs)),
    }

def _compute_sane_prices(preds: np.ndarray, mid: np.ndarray):
    mid = np.array(mid, dtype=float)
    preds = np.array(preds, dtype=float)
    chosen, diag = _choose_margin_interpretation(preds, mid)

    if chosen == "ratio":
        profit_pct = 100.0 * preds
        profit_gp = preds * mid
    elif chosen == "percent":
        profit_pct = preds
        profit_gp = (preds / 100.0) * mid
    else:
        profit_gp = preds
        with np.errstate(divide='ignore', invalid='ignore'):
            profit_pct = 100.0 * (profit_gp / np.where(mid == 0, np.nan, mid))

    profit_pct = np.nan_to_num(profit_pct, nan=0.0, posinf=1e6, neginf=-1e6)
    profit_pct_clamped = np.clip(profit_pct, -99.0, 2000.0)

    profit_gp_clamped = (profit_pct_clamped / 100.0) * mid
    predicted_sell = mid + profit_gp_clamped

    predicted_sell = np.where(predicted_sell < 0, mid, predicted_sell)
    profit_gp_clamped = np.where(predicted_sell < 0, 0.0, profit_gp_clamped)

    diagnostics = {"chosen_interpretation": chosen, **diag}
    return predicted_sell, profit_gp_clamped, profit_pct_clamped, diagnostics

def load_latest_model():
    model_path = MODEL_DIR / "latest_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"❌ No trained model found at {model_path}")
    model_dict = joblib.load(model_path)
    print(f"📦 Loaded model from {model_path}")
    print(f"   Trained {model_dict.get('timestamp', 'unknown')} (R²={model_dict.get('r2', 0.0):.4f})")
    return model_dict

def load_latest_features():
    feature_files = sorted(FEATURE_DIR.glob("features_*.parquet"), key=os.path.getmtime, reverse=True)
    if not feature_files:
        raise FileNotFoundError("❌ No feature snapshots found in data/features/")
    latest_file = feature_files[0]
    print(f"📊 Using feature snapshot: {latest_file}")
    df = pd.read_parquet(latest_file)
    return df

# ---------------------------------------------------------------------
# PREDICT FLIPS
# ---------------------------------------------------------------------
def predict_flips(model_dict, df, top_n=100, min_volume_ratio=2.0):
    """Apply model to precomputed features and generate flip rankings."""
    before_total = len(df)
    df = df[~df["name"].isin(ITEM_BLACKLIST) & ~df["item_id"].isin(ITEM_BLACKLIST)]
    removed = before_total - len(df)
    if removed > 0:
        print(f"🧹 Removed {removed} blacklisted items before prediction.")

    model = model_dict["model"]
    expected_features = model_dict.get("features", [])
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[expected_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    preds = model.predict(X)
    df["raw_predicted_margin"] = preds.astype(float)

    mid_arr = df.get("mid_price", pd.Series(0)).astype(float).to_numpy()
    predicted_sell, predicted_profit_gp, profit_pct, diagnostics = _compute_sane_prices(preds, mid_arr)
    df["predicted_sell_price"] = predicted_sell
    df["predicted_profit_gp"] = predicted_profit_gp
    df["profit_pct"] = profit_pct

    print(f"⚙️ Prediction interpretation: {diagnostics['chosen_interpretation']}")

    if "daily_volume" in df.columns:
        before = len(df)
        try:
            limits = pd.read_html("https://oldschool.runescape.wiki/w/Grand_Exchange/Buying_limits")[0]
            limits = limits.rename(columns=lambda c: c.lower().strip())
            limit_map = dict(zip(limits["item"], limits["limit"]))
            df["limit"] = df["name"].map(limit_map).fillna(100)
            print("✅ Loaded GE buying limits.")
        except Exception as e:
            print(f"⚠️ Could not fetch GE limits: {e}")
            df["limit"] = 100

        df["volume_ratio"] = df["daily_volume"].astype(float) / df["limit"].replace(0, np.nan).astype(float)
        df["volume_ratio"] = df["volume_ratio"].fillna(0.0)
        df = df[df["volume_ratio"] >= float(min_volume_ratio)]
        print(f"💧 Filtered by volume_ratio ≥ {min_volume_ratio}: {before} → {len(df)}")
    else:
        print("⚠️ No daily_volume column found — skipping liquidity ratio filter.")

    before_cull = len(df)
    df = df[df["profit_pct"].between(-99.0, 2000.0, inclusive="both")]
    if before_cull != len(df):
        print(f"⚠️ Culled {before_cull - len(df)} extreme profit_pct rows.")

    ranked = df.sort_values("predicted_profit_gp", ascending=False).head(top_n)
    out_cols = [
        "item_id", "name", "mid_price", "predicted_sell_price", "predicted_profit_gp",
        "profit_pct", "daily_volume", "limit", "volume_ratio", "volatility_1h", "raw_predicted_margin"
    ]
    out_cols = [c for c in out_cols if c in ranked.columns]

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    timestamped_path = PRED_DIR / f"top_flips_{ts}.csv"
    latest_path = PRED_DIR / "latest_top_flips.csv"
    ranked.to_csv(timestamped_path, columns=out_cols, index=False)
    ranked.to_csv(latest_path, columns=out_cols, index=False)
    print(f"💰 Saved top {len(ranked)} flips → {timestamped_path}")

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
    if "entry_price" not in top_flips.columns:
        top_flips["entry_price"] = top_flips["mid_price"]
    sell_recs = batch_recommend_sell(top_flips, latest_prices)
    sell_recs.to_csv(PRED_DIR / "sell_signals.csv", index=False)
