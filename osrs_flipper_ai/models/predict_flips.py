"""
predict_flips.py - Uses the latest trained model to generate OSRS flip predictions.
Loads precomputed features (no recomputation), applies model, filters by
liquidity using volume-to-buy-limit ratio, converts model output into sane
predicted sell prices, and saves ranked flips.
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
# HELPERS: Interpret model output safely
# ---------------------------------------------------------------------
def _choose_margin_interpretation(preds: np.ndarray, mid: np.ndarray):
    """
    Given model outputs (preds) and mid prices, pick the most plausible interpretation:
      - preds is 'ratio' (spread_ratio): profit_gp = preds * mid
      - preds is 'percent' (e.g. 5 => 5%): profit_gp = (preds/100) * mid
      - preds is 'absolute gp': profit_gp = preds

    Strategy: compute profit_pct for each interpretation and choose the one
    with the fewest extreme outliers (> max_pct_threshold).
    """
    eps = 1e-9
    mid_safe = np.where(mid <= 0, np.nan, mid)

    # candidate profit_pct arrays (as percentages)
    pct_if_ratio = 100.0 * preds                        # preds already ratio -> pct = 100 * ratio
    pct_if_percent = preds                               # preds already percent (e.g. 5 => 5%)
    pct_if_abs = 100.0 * (preds / (mid_safe + eps))      # preds in gp -> pct = preds / mid *100

    # helper to compute extreme fraction
    def extreme_fraction(arr, threshold=500.0):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e9, neginf=-1e9)
        return float((np.abs(arr) > threshold).sum()) / max(1.0, float(len(arr)))

    thr = 500.0  # 500% is a rough "extreme" threshold
    f_ratio = extreme_fraction(pct_if_ratio, thr)
    f_percent = extreme_fraction(pct_if_percent, thr)
    f_abs = extreme_fraction(pct_if_abs, thr)

    # Choose interpretation with smallest extreme fraction; tie-breaker: smallest median abs pct
    scores = [
        ("ratio", f_ratio, np.nanmedian(np.abs(pct_if_ratio))),
        ("percent", f_percent, np.nanmedian(np.abs(pct_if_percent))),
        ("absolute", f_abs, np.nanmedian(np.abs(pct_if_abs))),
    ]
    scores.sort(key=lambda x: (x[1], x[2]))  # prefer lower extreme fraction, then lower median abs pct

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
    """
    Returns (predicted_sell_price_array, predicted_profit_gp_array, profit_pct_array, chosen_interpretation, diagnostics)
    with values clamped to reasonable bounds.
    """
    mid = np.array(mid, dtype=float)
    preds = np.array(preds, dtype=float)

    chosen, diag = _choose_margin_interpretation(preds, mid)

    if chosen == "ratio":
        profit_pct = 100.0 * preds
        profit_gp = preds * mid
    elif chosen == "percent":
        profit_pct = preds
        profit_gp = (preds / 100.0) * mid
    else:  # absolute
        profit_gp = preds
        with np.errstate(divide='ignore', invalid='ignore'):
            profit_pct = 100.0 * (profit_gp / np.where(mid == 0, np.nan, mid))

    # Sanity clamp profit_pct to avoid absurd values (e.g. model hallucinations)
    # Allow negative prices (losses) but clamp to sensible limits:
    profit_pct = np.nan_to_num(profit_pct, nan=0.0, posinf=1e6, neginf=-1e6)

    # Clamp pct to [-99%, +2000%] by default
    profit_pct_clamped = np.clip(profit_pct, -99.0, 2000.0)

    # Recompute profit_gp and sell price after clamping
    profit_gp_clamped = (profit_pct_clamped / 100.0) * mid
    predicted_sell = mid + profit_gp_clamped

    # Ensure non-negative sell price
    predicted_sell = np.where(predicted_sell < 0, mid, predicted_sell)
    profit_gp_clamped = np.where(predicted_sell < 0, 0.0, profit_gp_clamped)
    profit_pct_clamped = np.where(mid == 0, 0.0, profit_pct_clamped)

    diagnostics = {
        "chosen_interpretation": chosen,
        **diag,
        "max_pred": float(np.nanmax(preds)) if preds.size else 0.0,
        "min_mid": float(np.nanmin(mid)) if mid.size else 0.0,
    }

    return predicted_sell, profit_gp_clamped, profit_pct_clamped, diagnostics


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
def predict_flips(model_dict, df, top_n=100, min_volume_ratio=2.0):
    """Apply model to precomputed features and generate flip rankings."""
    model = model_dict["model"]
    expected_features = model_dict.get("features", [])

    # Ensure required columns exist
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[expected_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Model prediction
    preds = model.predict(X)
    df["raw_predicted_margin"] = preds.astype(float)

    # Compute sane prices from predictions
    mid_arr = df.get("mid_price", pd.Series(0)).astype(float).to_numpy()
    predicted_sell, predicted_profit_gp, profit_pct, diagnostics = _compute_sane_prices(preds, mid_arr)

    df["predicted_sell_price"] = predicted_sell
    df["predicted_profit_gp"] = predicted_profit_gp
    df["profit_pct"] = profit_pct

    print(f"⚙️ Prediction interpretation: {diagnostics['chosen_interpretation']}")
    print(f"   extreme fractions (ratio,percent,abs): "
          f"{diagnostics.get('ratio_extreme_frac'):.4f}, {diagnostics.get('percent_extreme_frac'):.4f}, {diagnostics.get('abs_extreme_frac'):.4f}")
    print(f"   median pct (ratio,percent,abs): "
          f"{diagnostics.get('median_pct_ratio'):.2f}, {diagnostics.get('median_pct_percent'):.2f}, {diagnostics.get('median_pct_abs'):.2f}")

    # ------------------------------------------------------------------
    # Liquidity filter using daily_volume / buy_limit ratio
    # ------------------------------------------------------------------
    if "daily_volume" in df.columns:
        before = len(df)
        try:
            limits = pd.read_html("https://oldschool.runescape.wiki/w/Grand_Exchange/Buying_limits")[0]
            limits = limits.rename(columns=lambda c: c.lower().strip())
            limit_map = dict(zip(limits["item"], limits["limit"]))
            df["limit"] = df["name"].map(limit_map).fillna(100)
            print("✅ Loaded GE buying limits for liquidity ratio filter.")
        except Exception as e:
            print(f"⚠️ Could not fetch GE limits for liquidity ratio filter: {e}")
            df["limit"] = df.get("limit", 100)

        # safe division and fill
        df["volume_ratio"] = df["daily_volume"].astype(float) / df["limit"].replace(0, np.nan).astype(float)
        df["volume_ratio"] = df["volume_ratio"].fillna(0.0)
        df = df[df["volume_ratio"] >= float(min_volume_ratio)]
        print(f"💧 Filtered by volume_ratio ≥ {min_volume_ratio}: {before} → {len(df)} rows")
    else:
        print("⚠️ No daily_volume column found — skipping liquidity ratio filter.")

    # Remove absurd extremes after filtering (double safety)
    # Keep reasonable profit_pct range: [-99%, 2000%]
    before_cull = len(df)
    df = df[df["profit_pct"].between(-99.0, 2000.0, inclusive="both")]
    culled = before_cull - len(df)
    if culled > 0:
        print(f"⚠️ Culled {culled} rows with absurd profit_pct values.")

    # Ranking
    ranked = df.sort_values("predicted_profit_gp", ascending=False).head(top_n)

    # Select output columns (make sure they exist)
    out_cols = [
        "item_id", "name",
        "mid_price", "predicted_sell_price", "predicted_profit_gp", "profit_pct",
        "daily_volume", "limit", "volume_ratio", "volatility_1h", "raw_predicted_margin"
    ]
    out_cols = [c for c in out_cols if c in ranked.columns]

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    timestamped_path = PRED_DIR / f"top_flips_{ts}.csv"
    latest_path = PRED_DIR / "latest_top_flips.csv"

    if not ranked.empty:
        ranked.to_csv(timestamped_path, columns=out_cols, index=False)
        ranked.to_csv(latest_path, columns=out_cols, index=False)
        print(f"💰 Saved top {len(ranked)} flips → {timestamped_path}")
    else:
        print("⚠️ No flips to save — writing placeholder CSV.")
        pd.DataFrame(columns=out_cols).to_csv(latest_path, index=False)

    # return dataframe (in memory) for downstream processing
    return ranked


# ---------------------------------------------------------------------
# MULTI-TIER RECOMMENDATION (keeps original behavior, but uses computed fields)
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

    # SELL RECOMMENDATIONS step (unchanged)
    latest_prices = fetch_latest_prices_dict()
    if "entry_price" not in top_flips.columns:
        top_flips["entry_price"] = top_flips["mid_price"]
    sell_recs = batch_recommend_sell(top_flips, latest_prices)
    sell_recs.to_csv(PRED_DIR / "sell_signals.csv", index=False)

    print("\n💰 === SELL RECOMMENDATIONS ===")
    if "should_sell" in sell_recs.columns:
        print(sell_recs[sell_recs["should_sell"]])
    else:
        print("⚠️ No 'should_sell' column found in sell_recs.")
