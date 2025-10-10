"""
predict_flips.py - Uses the latest trained model to generate OSRS flip predictions.
Loads precomputed features (no recomputation), applies model, filters by 24h volume, and saves ranked flips.
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
# LOADERS
# ---------------------------------------------------------------------
def load_latest_model():
    """Load the most recent trained model."""
    model_path = MODEL_DIR / "latest_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"❌ No trained model found at {model_path}")
    model_dict = joblib.load(model_path)
    print(f"📦 Loaded model from {model_path}")
    print(f"   Trained {model_dict.get('timestamp', 'unknown')} (R²={model_dict.get('r2', 0.0):.4f})")
    return model_dict


def load_latest_features():
    """Load precomputed feature snapshot — skip recomputing indicators."""
    feature_files = sorted(FEATURE_DIR.glob("features_*.parquet"), key=os.path.getmtime, reverse=True)
    if not feature_files:
        raise FileNotFoundError("❌ No feature snapshots found in data/features/")
    latest_file = feature_files[0]
    print(f"📊 Using feature snapshot: {latest_file}")
    df = pd.read_parquet(latest_file)
    print("⚙️ Skipping recomputation of technical indicators (already precomputed).")
    return df


# ---------------------------------------------------------------------
# PREDICTION + OUTPUT
# ---------------------------------------------------------------------
def predict_flips(model_dict, df, top_n=100):
    """Apply model to precomputed features and generate flip rankings."""
    model = model_dict["model"]
    expected_features = model_dict["features"]

    # Handle missing columns
    missing_cols = [c for c in expected_features if c not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns in feature data: {missing_cols}")
        for c in missing_cols:
            df[c] = 0.0

    # Align feature columns and fill missing values
    X = df[expected_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Predictions
    df["predicted_margin"] = model.predict(X)
    df["predicted_profit_gp"] = df["predicted_margin"] * df.get("mid_price", 0)

    # Filter by minimum 24h trade volume
    MIN_DAILY_VOLUME = 150
    if "daily_volume" in df.columns:
        before = len(df)
        df = df[df["daily_volume"] >= MIN_DAILY_VOLUME]
        print(f"💧 Filtered by daily_volume ≥ {MIN_DAILY_VOLUME}: {before} → {len(df)} rows")
    else:
        print("⚠️ No daily_volume column found — skipping liquidity filter.")

    # Rank by profit
    keep_cols = [
        "item_id", "name", "predicted_profit_gp", "predicted_margin",
        "mid_price", "daily_volume", "volatility_1h"
    ]
    for col in ["technical_score"]:
        if col not in df.columns:
            df[col] = 0.0
            keep_cols.append(col)

    ranked = (
        df.sort_values("predicted_profit_gp", ascending=False)
          .head(top_n)
          .loc[:, keep_cols]
    )

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    timestamped_path = PRED_DIR / f"top_flips_{ts}.csv"
    latest_path = PRED_DIR / "latest_top_flips.csv"

    if not ranked.empty:
        ranked.to_csv(timestamped_path, index=False)
        ranked.to_csv(latest_path, index=False)
        print(f"💰 Saved top {top_n} flips → {timestamped_path}")
    else:
        print("⚠️ No flips to save — writing placeholder CSV.")
        pd.DataFrame(columns=["item_id", "name", "predicted_profit_gp", "roi"]).to_csv(latest_path, index=False)

    return ranked


# ---------------------------------------------------------------------
# MULTI-TIER RECOMMENDATION
# ---------------------------------------------------------------------
CAPITAL_TIERS = [200_000_000, 100_000_000, 75_000_000, 50_000_000,
                 35_000_000, 20_000_000, 15_000_000, 10_000_000]
FLIPS_PER_TIER = 4
OUTPUT_DIR = BASE_DIR / "data" / "predictions" / "tiers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def recommend_flips_by_tier(df, capital_tiers=CAPITAL_TIERS, flips_per_tier=FLIPS_PER_TIER):
    results = []
    for capital in capital_tiers:
        sub = df[(df["mid_price"] > 0) & (df["predicted_profit_gp"] > 0)].copy()
        sub["limit"] = sub.get("limit", 100)

        sub["max_affordable_qty"] = np.floor(capital / sub["mid_price"]).astype(int)
        sub["suggested_qty"] = np.minimum(sub["limit"], sub["max_affordable_qty"])
        sub["investment_gp"] = sub["suggested_qty"] * sub["mid_price"]
        sub["expected_profit_gp_total"] = sub["suggested_qty"] * sub["predicted_profit_gp"]

        sub = sub[sub["investment_gp"] > 0]
        best = sub.sort_values("expected_profit_gp_total", ascending=False).head(flips_per_tier)
        best["capital_tier"] = capital
        results.append(best)

        print(f"\n💰 Top {flips_per_tier} flips for {capital:,} gp:")
        for _, r in best.iterrows():
            print(f"  - {r['name']}: {r['suggested_qty']}x @ {r['mid_price']:,} gp "
                  f"(Invest {r['investment_gp']:,}, Expect +{r['expected_profit_gp_total']:,} gp)")

        best.to_csv(OUTPUT_DIR / f"top_flips_{capital//1_000_000}M.csv", index=False)

    full = pd.concat(results, ignore_index=True)
    full.to_csv(OUTPUT_DIR / "top_flips_all_tiers.csv", index=False)
    print(f"\n✅ All tier results saved to {OUTPUT_DIR}")
    return full


# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting flip prediction pipeline...")

    model_dict = load_latest_model()
    df = load_latest_features()

    top_flips = predict_flips(model_dict, df, top_n=100)
    print(f"🔍 DEBUG: predict_flips() returned {len(top_flips)} rows")

    # Load GE buying limits
    try:
        limits = pd.read_html("https://oldschool.runescape.wiki/w/Grand_Exchange/Buying_limits")[0]
        limits = limits.rename(columns=lambda c: c.lower().strip())
        limits_map = dict(zip(limits["item"], limits["limit"]))
        top_flips["limit"] = top_flips["name"].map(limits_map).fillna(100)
        print("✅ Loaded GE buying limits.")
    except Exception as e:
        print(f"⚠️ Could not fetch GE limits: {e}")
        top_flips["limit"] = 100

    # Generate tiered recommendations
    all_tiers = recommend_flips_by_tier(top_flips)

    # Save unified output
    latest_path = PRED_DIR / "latest_top_flips.csv"
    all_tiers.to_csv(latest_path, index=False)
    print(f"💾 Saved unified flips → {latest_path}")

    # SELL RECOMMENDATIONS
    latest_prices = fetch_latest_prices_dict()
    all_tiers["entry_price"] = all_tiers.get("entry_price", all_tiers["mid_price"])
    sell_recs = batch_recommend_sell(all_tiers, latest_prices)
    sell_recs.to_csv(PRED_DIR / "sell_signals.csv", index=False)

    print("\n💰 === SELL RECOMMENDATIONS ===")
    if "should_sell" in sell_recs.columns:
        print(sell_recs[sell_recs["should_sell"]])
    else:
        print("⚠️ No 'should_sell' column found in sell_recs.")
