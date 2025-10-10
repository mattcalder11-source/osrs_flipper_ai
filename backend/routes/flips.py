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
from fastapi.responses import JSONResponse
import numpy as np
from datetime import datetime

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
        for alt in ["id", "itemID", "item"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "item_id"})
                break

    if not mapping_path.exists():
        print(f"⚠️ {mapping_path} not found — enrichment skipped.")
        if "name" not in df.columns:
            df["name"] = df["item_id"].astype(str)
        df["icon_url"] = None
        df["buy_limit"] = df.get("buy_limit", 100)
        return df

    try:
        with open(mapping_path) as f:
            mapping = {int(item["id"]): item for item in json.load(f)}

        if "name" not in df.columns:
            df["name"] = df["item_id"].map(lambda x: mapping.get(int(x), {}).get("name", "Unknown"))
        df["icon"] = df["item_id"].map(lambda x: mapping.get(int(x), {}).get("icon"))
        df["buy_limit"] = df["item_id"].map(lambda x: mapping.get(int(x), {}).get("limit"))
        df["icon_url"] = df["icon"].apply(
            lambda f: f"https://oldschool.runescape.wiki/images/{f}" if pd.notnull(f) else None
        )

    except Exception as e:
        print(f"⚠️ Failed to enrich item metadata: {e}")
        if "name" not in df.columns:
            df["name"] = df["item_id"].astype(str)
        df["icon_url"] = None
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

    # Defensive cleanup: keep only relevant numeric & core columns
    keep_core = [
        "item_id", "buy_price", "sell_price", "profit_pct",
        "predicted_profit_gp", "buy_limit", "volume_ratio", "volatility_1h"
    ]
    core_cols = [c for c in keep_core if c in df.columns]

    # Handle the case where metadata columns overshadow numeric fields
    # (e.g., from /mapping merges)
    numeric_df = df[core_cols].copy() if core_cols else pd.DataFrame()

    # Always keep item_id
    if "item_id" not in numeric_df.columns and "item_id" in df.columns:
        numeric_df["item_id"] = df["item_id"]

    # Enrich with metadata for dashboard
    numeric_df = enrich_with_metadata(numeric_df)

    print(f"✅ Loaded {len(numeric_df)} buy recs (cols: {list(numeric_df.columns)})")
    return numeric_df


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
    print(f"✅ Loaded {len(df)} sell recs (cols: {cols})")
    return df[cols]


# ---------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------
def df_to_safe_json(df: pd.DataFrame):
    """Safely convert any DataFrame to JSON-serializable dict."""
    def safe_val(x):
        if isinstance(x, (np.generic, np.bool_)):
            return x.item()
        if pd.isna(x):
            return None
        if isinstance(x, (pd.Timestamp, datetime)):
            return x.isoformat()
        if isinstance(x, (list, dict)):
            return json.dumps(x)
        return x

    return [
        {col: safe_val(val) for col, val in row.items()}
        for row in df.to_dict(orient="records")
    ]

@router.get("/flips/buy-recommendations")
def get_buy_recommendations():
    df = load_latest_predictions()
    if df.empty:
        return JSONResponse({"count": 0, "data": []})

    data = df_to_safe_json(df)
    return JSONResponse({"count": len(data), "data": data})


@router.get("/flips/sell-signals") 
def get_sell_recommendations():
    df = load_sell_signals()
    if df.empty:
        return JSONResponse({"count": 0, "data": []})

    data = df_to_safe_json(df)
    return JSONResponse({"count": len(data), "data": data})