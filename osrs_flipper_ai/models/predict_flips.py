# scripts/predict_flips.py
"""
predict_flips.py - Uses the latest trained model to generate current OSRS flip predictions.
Loads the newest feature snapshot, applies the model, and saves ranked recommendations.
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from osrs_flipper_ai.models.recommend_sell import batch_recommend_sell
from osrs_flipper_ai.src.fetch_latest_prices import fetch_latest_prices_dict

# Allow imports from the parent "src" folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from osrs_flipper_ai.features.features import compute_features, compute_technical_indicators

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
MODEL_DIR = os.path.join("models")
FEATURE_DIR = os.path.join("data", "features")
PRED_DIR = os.path.join("data", "predictions")

os.makedirs(PRED_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------
def load_latest_model():
    """Load the latest trained model."""
    latest_model_path = os.path.join(MODEL_DIR, "latest_model.pkl")
    if not os.path.exists(latest_model_path):
        raise FileNotFoundError("❌ No trained model found in /models.")
    model_dict = joblib.load(latest_model_path)
    print(f"📦 Loaded model trained on {model_dict['timestamp']} (R²={model_dict['r2']:.4f})")
    return model_dict


def load_latest_features():
    """Load the newest feature snapshot (produced by ingest.py)."""
    feature_files = sorted(
        [os.path.join(FEATURE_DIR, f) for f in os.listdir(FEATURE_DIR) if f.startswith("features_")],
        key=os.path.getmtime,
        reverse=True
    )
    if not feature_files:
        raise FileNotFoundError("❌ No feature snapshots found in data/features/")

    latest_file = feature_files[0]
    print(f"📊 Using latest snapshot: {latest_file}")
    df = pd.read_parquet(latest_file)

    # Enrich with engineered + technical features
    df = compute_features(df)
    df = compute_technical_indicators(df)
    return df


# ---------------------------------------------------------------------
# PREDICTION + OUTPUT
# ---------------------------------------------------------------------
def predict_flips(model_dict, df, top_n=10):
    """Generate top-N flip predictions from the latest snapshot."""
    model = model_dict["model"]
    feature_cols = model_dict["features"]

    # Safe fill missing feature columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["predicted_margin"] = model.predict(X)
    df["predicted_profit_gp"] = df["predicted_margin"] * df.get("mid_price", 0)

    ranked = (
        df.sort_values("predicted_profit_gp", ascending=False)
          .head(top_n)
          .loc[:, ["item_id", "name", "predicted_profit_gp", "predicted_margin",
                   "mid_price", "liquidity_1h", "volatility_1h", "technical_score"]]
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(PRED_DIR, f"top_flips_{ts}.csv")
    ranked.to_csv(out_path, index=False)

    # Also update the "latest" prediction
    latest_path = os.path.join(PRED_DIR, "latest_top_flips.csv")
    ranked.to_csv(latest_path, index=False)

    print(f"💰 Saved top {top_n} flips → {out_path}")
    return ranked

def recommend_quantities(df, available_gp=50_000_000, allocation_ratio=0.1, liquidity_factor=0.5):
    """
    Recommend quantity per item to flip, based on available GP and market liquidity.
    """
    df = df.copy()

    if "mid_price" not in df.columns or df["mid_price"].isna().all():
        print("⚠️ mid_price missing — cannot compute quantities.")
        df["recommended_qty"] = 0
        return df

    # Base quantity limited by capital allocation
    df["max_affordable_qty"] = (available_gp * allocation_ratio) / df["mid_price"]

    # Adjust for liquidity — don’t exceed realistic trade throughput
    if "liquidity_1h" in df.columns:
        df["liquidity_cap"] = df["liquidity_1h"] * liquidity_factor
    else:
        df["liquidity_cap"] = np.inf

    # GE limit if available
    if "limit" in df.columns:
        df["ge_cap"] = df["limit"]
    else:
        df["ge_cap"] = np.inf

    # Take the most conservative quantity
    df["recommended_qty"] = df[["max_affordable_qty", "liquidity_cap", "ge_cap"]].min(axis=1).astype(int)

    # Compute projected total investment and profit
    df["investment_gp"] = df["recommended_qty"] * df["mid_price"]
    df["expected_profit_gp"] = df["recommended_qty"] * df["predicted_profit_gp"]

    print("\n💼 Quantity recommendations generated based on available GP.")
    print(f"Total capital: {available_gp:,} gp")
    print(f"Per-item allocation: {allocation_ratio*100:.1f}% of capital")
    print(f"Liquidity factor: {liquidity_factor:.2f}")

    return df

# --- CONFIG ---
CAPITAL_TIERS = [200_000_000, 100_000_000, 75_000_000, 50_000_000, 35_000_000, 20_000_000, 15_000_000, 10_000_000]
FLIPS_PER_TIER = 4
OUTPUT_DIR = "data/predictions/tiers"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def recommend_flips_by_tier(df, capital_tiers=CAPITAL_TIERS, flips_per_tier=FLIPS_PER_TIER):
    """
    Recommend top flips across a range of bankroll sizes.
    Respects GE buy limits and returns most profitable 4 flips per tier.
    """
    results = []

    for capital in capital_tiers:
        available_gp = capital
        df = df.copy()

        # Skip invalids
        df = df[(df["mid_price"] > 0) & (df["predicted_profit_gp"] > 0)]
        if "limit" not in df.columns:
            df["limit"] = 100  # fallback default

        df["max_affordable_qty"] = np.floor(available_gp / df["mid_price"]).astype(int)
        df["suggested_qty"] = np.minimum(df["limit"], df["max_affordable_qty"])
        df["investment_gp"] = df["suggested_qty"] * df["mid_price"]
        df["expected_profit_gp_total"] = df["suggested_qty"] * df["predicted_profit_gp"]

        # Filter out anything too large for capital
        df = df[df["investment_gp"] > 0]
        df = df.sort_values("expected_profit_gp_total", ascending=False)

        best_flips = df.head(flips_per_tier).copy()
        best_flips["capital_tier"] = capital
        results.append(best_flips)

        print(f"\n💰 Top {flips_per_tier} flips for {capital:,} gp:")
        for _, row in best_flips.iterrows():
            print(f"  - {row['name']}: {row['suggested_qty']}x @ {row['mid_price']:,} gp "
                  f"(Invest {row['investment_gp']:,}, Expect +{row['expected_profit_gp_total']:,} gp)")

        # Save tier results
        tier_file = Path(OUTPUT_DIR) / f"top_flips_{capital//1_000_000}M.csv"
        best_flips.to_csv(tier_file, index=False)

    # Combine all tiers for convenience
    full = pd.concat(results, ignore_index=True)
    full.to_csv(Path(OUTPUT_DIR) / "top_flips_all_tiers.csv", index=False)
    print(f"\n✅ All tier results saved to {OUTPUT_DIR}")
    return full

# Load active flips (your buy recommendations or holdings)
pred_path = os.path.join("data", "predictions", "latest_top_flips.csv")
if not os.path.exists(pred_path):
    print(f"⚠️ No predictions found at {pred_path}. Skipping post-prediction steps.")
    exit(0)

active = pd.read_csv(pred_path)

# Fetch current prices from API or cache
latest_prices = fetch_latest_prices_dict()  # e.g. {item_id: mid_price}

sell_recs = batch_recommend_sell(active, latest_prices)
sell_recs.to_csv("data/predictions/sell_signals.csv", index=False)

print("\n💰 === SELL RECOMMENDATIONS ===")
print(sell_recs[sell_recs["should_sell"]])

# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------
if __name__ == "__main__":
    model_dict = load_latest_model()
    df = load_latest_features()
    top_flips = predict_flips(model_dict, df, top_n=100)  # broader selection

    print(f"🔍 DEBUG: predict_flips() returned {len(top_flips)} rows")

    if len(top_flips) > 0:
        print(top_flips.head(10))
    else:
        print("⚠️ DEBUG: top_flips is empty — investigating why.")


    # Load GE buy limits
    try:
        limits = pd.read_html("https://oldschool.runescape.wiki/w/Grand_Exchange/Buying_limits")[0]
        limits = limits.rename(columns=lambda c: c.lower().strip())
        limits_map = dict(zip(limits["item"], limits["limit"]))
        top_flips["limit"] = top_flips["name"].map(limits_map).fillna(100)
        print("✅ Loaded GE buying limits.")
    except Exception as e:
        print(f"⚠️ Could not fetch GE limits: {e}")
        top_flips["limit"] = 100

    # Generate multi-tier recommendations
    all_tiers = recommend_flips_by_tier(top_flips)

    # Save the unified output for dashboard use
    output_dir = Path("data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest_top_flips.csv"
    all_tiers.to_csv(latest_path, index=False)
    print(f"💾 Saved unified flips → {latest_path}")

    # -----------------------------------------------------------------
    # SELL RECOMMENDATION STAGE
    # -----------------------------------------------------------------
    latest_prices = fetch_latest_prices_dict()  # e.g. {item_id: mid_price}
    sell_recs = batch_recommend_sell(all_tiers, latest_prices)
    sell_recs.to_csv(output_dir / "sell_signals.csv", index=False)

    print("\n💰 === SELL RECOMMENDATIONS ===")
    print(sell_recs[sell_recs["should_sell"]])
