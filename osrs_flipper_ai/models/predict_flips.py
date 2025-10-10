#!/usr/bin/env python3
"""
predict_flips.py — Generates buy recommendations from the trained model
and evaluates sell recommendations from active flips.

Outputs:
  /data/predictions/latest_top_flips.csv   → BUY recommendations
  /data/predictions/sell_signals.csv       → SELL evaluations
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from recommend_sell import batch_recommend_sell

# ---------------------------------------------------------------------
# Paths and globals
# ---------------------------------------------------------------------
BASE_DIR = "/root/osrs_flipper_ai/osrs_flipper_ai"
DATA_DIR = Path(f"{BASE_DIR}/data")
PRED_DIR = DATA_DIR / "predictions"
RAW_DIR = DATA_DIR / "raw"
LATEST_PRICES_PATH = RAW_DIR / "latest_prices.json"
MAPPING_PATH = DATA_DIR / "item_mapping.json"
BLACKLIST_PATH = Path(f"{BASE_DIR}/data/blacklist.txt")

PRED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def load_model_and_features():
    """Load trained model and latest feature set."""
    model_path = Path(f"{BASE_DIR}/models/trained_models/latest_model.pkl")
    features_path = Path(f"{BASE_DIR}/data/features/features_latest.parquet")

    print("🚀 Starting flip prediction pipeline...")
    model = joblib.load(model_path)
    print(f"📦 Loaded model: {model_path}")
    if hasattr(model, "meta") and "r2" in model.meta:
        print(f"   Trained {model.meta.get('trained_at', 'unknown')} (R²={model.meta['r2']:.4f})")

    df = pd.read_parquet(features_path)
    print(f"📊 Loaded features: {features_path}")
    return model, df


def load_blacklist():
    """Read blacklist.txt and return cleaned list."""
    if not BLACKLIST_PATH.exists():
        print("⚠️ No blacklist.txt found.")
        return []
    items = []
    with open(BLACKLIST_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(line)
    print(f"🧱 Loaded blacklist ({len(items)} items): {items[:5]}...")
    return items


def load_latest_prices():
    """Load latest Wiki prices."""
    if not LATEST_PRICES_PATH.exists():
        print("⚠️ latest_prices.json not found.")
        return {}
    with open(LATEST_PRICES_PATH, "r") as f:
        prices = json.load(f)
    print(f"✅ Loaded {len(prices):,} current prices from Wiki.")
    return prices


# ---------------------------------------------------------------------
# Predict flips
# ---------------------------------------------------------------------
def predict_flips(model_dict, df, top_n=100):
    """Predict potential flips and write recommendations."""
    df = df.copy()
    if "y" in df.columns:
        df = df.drop(columns=["y"])

    # Make predictions
    df["pred"] = model_dict.predict(df.drop(columns=["item_id"], errors="ignore"))

    max_pred = df["pred"].max()
    print(f"⚙️ Interpreting model output as ratio mode (max pred={max_pred:.2f})")

    df["predicted_profit_gp"] = (df["pred"] - 1) * df["buy_price"]
    df["profit_pct"] = (df["pred"] - 1) * 100
    df["buy_limit"] = df.get("buy_limit", 0)

    # Sort by highest predicted profit
    df = df.sort_values("predicted_profit_gp", ascending=False).reset_index(drop=True)

    # Apply blacklist ---------------------------------------------------
    ITEM_BLACKLIST = load_blacklist()
    before = len(df)

    # Load mapping for name→ID resolution
    mapping_df = None
    if MAPPING_PATH.exists():
        try:
            mapping_df = pd.read_json(MAPPING_PATH)
            mapping_df = mapping_df.rename(columns={"id": "item_id"})
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

    # ------------------------------------------------------------------
    # Volume / buy-limit ratio filter
    # ------------------------------------------------------------------
    before = len(df)
    if "volume" in df.columns and "buy_limit" in df.columns:
        df = df[(df["buy_limit"] > 0) & ((df["volume"] / df["buy_limit"]) >= 0.5)]
    print(f"💧 Volume-to-limit ratio ≥ 0.5: {before} → {len(df)} rows")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    latest_prices = load_latest_prices()
    top_flips = df.head(top_n)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    latest_path = PRED_DIR / "latest_top_flips.csv"
    top_flips.to_csv(latest_path, index=False)
    print(f"💾 Saved {len(top_flips)} flips → {latest_path}")

    # Also compute sell recommendations
    print("🧠 Generating sell evaluations...")
    sell_signals = batch_recommend_sell(top_flips, latest_prices)
    sell_path = PRED_DIR / "sell_signals.csv"
    sell_signals.to_csv(sell_path, index=False)
    print(f"💾 Saved {len(sell_signals)} sell signals → {sell_path}")

    return top_flips


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    model_dict, df = load_model_and_features()
    top_flips = predict_flips(model_dict, df, top_n=100)
    print("✅ Prediction pipeline complete.")
