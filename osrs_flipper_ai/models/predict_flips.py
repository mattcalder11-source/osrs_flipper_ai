#!/usr/bin/env python3
"""
predict_flips.py
Predict profitable OSRS flips using a trained ML model and precomputed features.

Saves:
 - data/predictions/latest_top_flips.csv  -> BUY recommendations (raw predictions)
 - data/predictions/sell_signals.csv      -> SELL recommendations (evaluation)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from osrs_flipper_ai.models.recommend_sell import batch_recommend_sell
from osrs_flipper_ai.src.fetch_latest_prices import save_latest_prices

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
BASE_DIR = "/root/osrs_flipper_ai/osrs_flipper_ai"
FEATURES_DIR = "/root/osrs_flipper_ai/data/features"
MODEL_DIR = f"{BASE_DIR}/models/trained_models"
PREDICTIONS_DIR = Path(f"{BASE_DIR}/data/predictions")
LIMITS_FILE = f"{BASE_DIR}/data/ge_limits.json"
ITEM_BLACKLIST_FILE = f"{BASE_DIR}/data/item_blacklist.txt"
MIN_VOLUME_RATIO = 2  # daily_volume-to-buy_limit ratio threshold

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# LOAD MODEL & FEATURES
# ---------------------------------------------------------------------
def load_latest_model():
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"âŒ Model directory not found: {MODEL_DIR}")

    model_files = [
        f for f in os.listdir(MODEL_DIR)
        if f.endswith(".pkl") and "model" in f
    ]
    if not model_files:
        raise FileNotFoundError("âŒ No trained models found in trained_models/")

    latest_model_file = max(model_files, key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)))
    model_path = os.path.join(MODEL_DIR, latest_model_file)

    model_dict = joblib.load(model_path)
    timestamp = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y%m%d_%H%M")

    print(f"ðŸ“¦ Loaded model: {model_path}")
    print(f"   Trained {timestamp} (RÂ²={model_dict.get('r2', 0):.4f})")

    model_dict["path"] = model_path
    model_dict["timestamp"] = timestamp
    return model_dict


def load_latest_features():
    if not os.path.exists(FEATURES_DIR):
        raise FileNotFoundError(f"âŒ Feature directory not found: {FEATURES_DIR}")

    files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError("âŒ No feature snapshots found in data/features/")

    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(FEATURES_DIR, f)))
    feature_path = os.path.join(FEATURES_DIR, latest_file)

    print(f"ðŸ“Š Loaded features: {feature_path}")
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
print(f"ðŸ§± Loaded blacklist ({len(ITEM_BLACKLIST)} items): {sorted(list(ITEM_BLACKLIST))[:5]}...")


# ---------------------------------------------------------------------
# ADAPTIVE PRICE INTERPRETATION
# ---------------------------------------------------------------------
def _compute_sane_prices(preds: np.ndarray, mid: np.ndarray):
    preds = np.nan_to_num(np.array(preds, dtype=float))
    mid = np.nan_to_num(np.array(mid, dtype=float))
    mid[mid <= 0] = np.nan  # avoid divide by zero

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
    print(f"âš™ï¸ Interpreting model output as {diagnostics}")

    return sell_price, profit_gp, profit_pct


# ---------------------------------------------------------------------
# MAIN FLIP PREDICTION FUNCTION
# ---------------------------------------------------------------------
def predict_flips(model_dict, df, top_n=100):
    """
    Generate flip predictions and filter by blacklist and liquidity.
    This function returns the BUY recommendations dataframe (raw predictions).
    It also computes and writes sell_signals.csv separately for evaluations.
    """
    model = model_dict["model"]
    features = model_dict["features"]

    df = df.dropna(subset=features)
    preds = model.predict(df[features])

    sell_price, profit_gp, profit_pct = _compute_sane_prices(preds, df["mid_price"])
    df["buy_price"] = df["mid_price"].clip(lower=1)
    df["sell_price"] = sell_price
    df["predicted_profit_gp"] = profit_gp
    df["profit_pct"] = profit_pct

    # ------------------------------------------------------------------
    # Apply blacklist — robustly using mapping to resolve names → item_ids
    # ------------------------------------------------------------------
    before = len(df)

        # Load mapping to resolve item names to IDs if available
    mapping_path = Path(f"{BASE_DIR}/data/item_mapping.json")
    mapping_df = None
    if mapping_path.exists():
        try:
            # item_mapping.json is a nested dict keyed by item_id, so normalize it
            with open(mapping_path, "r") as f:
                raw_mapping = json.load(f)
            mapping_df = pd.DataFrame.from_dict(raw_mapping, orient="index")
            mapping_df["item_id"] = mapping_df["id"].astype(int)
            mapping_df["name"] = mapping_df["name"].astype(str).str.lower()
        except Exception as e:
            print(f"⚠️ Failed to load item_mapping.json for blacklist: {e}")
    else:
        print("⚠️ item_mapping.json not found — blacklist will be ID-only.")

    # Normalize blacklist entries
    blacklist_normalized = set()
    for entry in ITEM_BLACKLIST:
        entry = entry.strip().lower()
        if entry.isdigit():
            blacklist_normalized.add(int(entry))
        else:
            blacklist_normalized.add(entry)

    # If blacklist contains names but df has only IDs, map names → IDs
    if mapping_df is not None:
        name_to_id = mapping_df[mapping_df["name"].isin(blacklist_normalized)]["item_id"].astype(int).tolist()
        blacklist_ids = {i for i in blacklist_normalized if isinstance(i, int)} | set(name_to_id)
    else:
        blacklist_ids = {i for i in blacklist_normalized if isinstance(i, int)}

    # Apply blacklist filtering
    if "item_id" in df.columns:
        df = df[~df["item_id"].isin(blacklist_ids)]
    elif "name" in df.columns:
        df = df[~df["name"].str.lower().isin(blacklist_normalized)]
    else:
        print("⚠️ No item_id or name column found — skipping blacklist filter.")

    print(f"🚫 Blacklist filter: {before} → {len(df)} rows (filtered {before - len(df)})")



    # Load buy limits
    if os.path.exists(LIMITS_FILE):
        with open(LIMITS_FILE, "r") as f:
            limits = json.load(f)
        limits_df = pd.DataFrame(list(limits.items()), columns=["item_id", "buy_limit"])
        limits_df["item_id"] = limits_df["item_id"].astype(int)
        df = df.merge(limits_df, on="item_id", how="left")
    else:
        df["buy_limit"] = np.nan
        print("âš ï¸ No GE limit file found!")

    # Volume-to-limit ratio filter (defensive)
    if "daily_volume" in df.columns:
        before = len(df)
        # Avoid division by zero / NaN by filling buy_limit with large default where missing
        df["buy_limit"] = df["buy_limit"].replace(0, np.nan).fillna(1000)
        df["vol_to_limit_ratio"] = df["daily_volume"] / df["buy_limit"]
        df["vol_to_limit_ratio"] = df["vol_to_limit_ratio"].replace([np.inf, -np.inf], np.nan)
        if df["vol_to_limit_ratio"].notna().sum() > 0:
            df = df[df["vol_to_limit_ratio"] >= MIN_VOLUME_RATIO]
        else:
            print("âš ï¸ Skipping volume_limit_ratio filter â€” invalid data.")
        print(f"ðŸ’§ Volume-to-limit ratio â‰¥ {MIN_VOLUME_RATIO}: {before} â†’ {len(df)} rows")
    else:
        print("âš ï¸ No daily_volume column found â€” skipping liquidity filter.")

    # Drop unrealistic predictions
    df = df[df["profit_pct"].between(-50, 1000)]

    # Rank and select top flips (BUY recommendations)
    df = df.sort_values("predicted_profit_gp", ascending=False)
    raw_top_flips = df.head(top_n).copy()

    # Add prediction vector as a column for debugging / ranking analysis
    df["pred"] = preds

    # Log distribution and deduplication
    if "pred" in df.columns:
        print(f"📊 Pred summary:\n{df['pred'].describe()}")
    else:
        print("⚠️ 'pred' column missing — skipping summary print.")

    print(f"🔁 Unique item_ids before dedup: {df['item_id'].nunique()}")
    df = df.drop_duplicates("item_id")
    print(f"✅ After deduplication: {df['item_id'].nunique()} unique items")


    # --- Save BUY recommendations (raw predictions) ---
    buy_path = PREDICTIONS_DIR / "latest_top_flips.csv"
    raw_top_flips.to_csv(buy_path, index=False)
    print(f"ðŸ’¾ Saved BUY recommendations â†’ {buy_path} ({len(raw_top_flips)} rows)")

    # Fetch latest prices for sell evaluation and persist
    latest_prices_dict = save_latest_prices()  # this should save latest_prices.json and return dict

    # Prepare a copy to evaluate sell signals (if you want to treat buys as open positions)
    eval_df = raw_top_flips.copy()

    # Ensure compatibility with recommend_sell.py
    if "entry_price" not in eval_df.columns:
        eval_df["entry_price"] = eval_df["buy_price"]
    if "entry_time" not in eval_df.columns:
        # timezone-aware UTC timestamp
        eval_df["entry_time"] = pd.Timestamp.now(tz="UTC")

    # Run sell recommendation logic on eval_df
    sell_recs = batch_recommend_sell(eval_df, latest_prices_dict)

    # Save SELL evaluations (these are evaluations of the buy candidates)
    sell_path = PREDICTIONS_DIR / "sell_signals.csv"
    sell_recs.to_csv(sell_path, index=False)
    print(f"ðŸ’¾ Saved SELL evaluations â†’ {sell_path} ({len(sell_recs)} rows)")

    # Return raw buy recommendations for further use
    return raw_top_flips


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("ðŸš€ Starting flip prediction pipeline...")

    model_dict = load_latest_model()
    df = load_latest_features()

    top_flips = predict_flips(model_dict, df, top_n=100)

    # For backward compatibility the script still writes latest_top_flips.csv
    latest_csv = PREDICTIONS_DIR / "latest_top_flips.csv"
    top_flips.to_csv(latest_csv, index=False)
    print(f"ðŸ’¾ Saved final BUY predictions â†’ {latest_csv}")
    print(f"âœ… {len(top_flips)} buy flips ready for dashboard.")