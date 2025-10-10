#!/usr/bin/env python3
"""
flips.py — FastAPI backend for OSRS Flipping Dashboard

Serves:
  • /flips/buy-recommendations → latest_top_flips.csv (buy signals)
  • /flips/sell-recommendations → sell_signals.csv (active flips to evaluate)
Enriches both with item metadata (name, icon, buy_limit) from item_mapping.json.
"""

import json
import pandas as pd
from fastapi import APIRouter
from pathlib import Path

router = APIRouter()
DATA_DIR = Path("/root/osrs_flipper_ai/osrs_flipper_ai/data")
PREDICTIONS_DIR = DATA_DIR / "predictions"


# ---------------------------------------------------------------------
# Helper: Load and enrich dataframe with item metadata
# ---------------------------------------------------------------------
def enrich_with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Attach item name, icon, and buy limit using item_mapping.json."""
    mapping_path = DATA_DIR / "item_mapping.json"

    if "item_id" not in df.columns:
        # Try to auto-detect similar column
        for alt in ["id", "itemID", "item"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "item_id"})
                break


    if not mapping_path.exists():
        print(f"⚠️ {mapping_path} not found — enrichment skipped.")
        df["name"] = df["item_id"].astype(str)
        df["icon"] = None
        df["buy_limit"] = 100
        return df

    try:
        with open(mapping_path) as f:
            mapping = {int(item["id"]): item for item in json.load(f)}

        df["name"] = df["item_id"].map(lambda x: mapping.get(int(x), {}).get("name", "Unknown"))
        df["icon"] = df["item_id"].map(lambda x: mapping.get(int(x), {}).get("icon"))
        df["buy_limit"] = df["item_id"].map(lambda x: mapping.get(int(x), {}).get("buy_limit"))
        df["icon_url"] = df["icon"].apply(
            lambda f: f"https://oldschool.runescape.wiki/images/{f}" if pd.notnull(f) else None
        )

    except Exception as e:
        print(f"⚠️ Failed to enrich item metadata: {e}")
        df["name"] = df["item_id"].astype(str)
        df["icon"] = None
        df["buy_limit"] = 100

    return df


# ---------------------------------------------------------------------
# Loader: Buy recommendations (latest_top_flips.csv)
# ---------------------------------------------------------------------
def load_latest_predictions() -> pd.DataFrame:
    latest = PREDICTIONS_DIR / "latest_top_flips.csv"
    if not latest.exists() or latest.stat().st_size == 0:
        print(f"⚠️ No buy recommendations found at {latest}")
        return pd.DataFrame()

    df = pd.read_csv(latest)
    df = enrich_with_metadata(df)

    # Keep relevant columns for dashboard
    possible_cols = [
        "item_id", "name", "icon_url", "buy_price", "sell_price", "profit_pct",
        "predicted_profit_gp", "buy_limit", "volume_ratio", "volatility_1h",
    ]
    cols = [c for c in possible_cols if c in df.columns]
    return df[cols]


# ---------------------------------------------------------------------
# Loader: Sell recommendations (sell_signals.csv)
# ---------------------------------------------------------------------
def load_sell_signals() -> pd.DataFrame:
    sell_path = PREDICTIONS_DIR / "sell_signals.csv"
    if not sell_path.exists() or sell_path.stat().st_size == 0:
        print(f"⚠️ No sell signals found at {sell_path}")
        return pd.DataFrame()

    df = pd.read_csv(sell_path)
    df = enrich_with_metadata(df)

    possible_cols = [
        "item_id", "name", "icon_url", "current_price", "should_sell",
        "reason", "confidence", "profit_pct", "hold_hours", "urgency_score",
    ]
    cols = [c for c in possible_cols if c in df.columns]
    return df[cols]


# ---------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------
@router.get("/flips/buy-recommendations")
def get_buy_recommendations():
    df = load_latest_predictions()
    return {"count": len(df), "data": df.to_dict(orient="records")}


@router.get("/flips/sell-recommendations")
def get_sell_recommendations():
    df = load_sell_signals()
    return {"count": len(df), "data": df.to_dict(orient="records")}
