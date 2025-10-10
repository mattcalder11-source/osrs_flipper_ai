#!/usr/bin/env python3
"""
flips.py — FastAPI backend for OSRS Flipping Dashboard

Serves:
  • /flips/buy-recommendations → latest_top_flips.csv (buy signals)
  • /flips/sell-signals → sell_signals.csv (active flips to evaluate)
  • /flips/active, /flips/add/{id}, /flips/close/{id}
Enriches all with item metadata (name, icon, buy_limit, icon_url) from item_mapping.json.
"""

import json
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
import numpy as np

router = APIRouter()
DATA_DIR = Path("/root/osrs_flipper_ai/osrs_flipper_ai/data")
PREDICTIONS_DIR = DATA_DIR / "predictions"
MAPPING_PATH = DATA_DIR / "item_mapping.json"
ACTIVE_FLIPS_PATH = DATA_DIR / "active_flips.csv"


# ---------------------------------------------------------------------
# Enrichment Utilities
# ---------------------------------------------------------------------
def enrich_with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Attach item name and icon URLs using item_mapping.json."""
    if df is None or df.empty:
        return df

    if not MAPPING_PATH.exists():
        print("⚠️ item_mapping.json missing — skipping enrichment.")
        df["name"] = df.get("name", df["item_id"].astype(str))
        df["icon_url"] = "/placeholder.png"
        return df

    try:
        with open(MAPPING_PATH, "r") as f:
            mapping = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load mapping: {e}")
        df["name"] = df.get("name", df["item_id"].astype(str))
        df["icon_url"] = "/placeholder.png"
        return df

    def get_field(item_id, field):
        try:
            return mapping.get(str(int(item_id)), {}).get(field, "")
        except Exception:
            return ""

    df["name"] = df["item_id"].apply(lambda x: get_field(x, "name") or str(x))
    df["icon"] = df["item_id"].apply(lambda x: get_field(x, "icon"))
    df["icon_url"] = df["item_id"].apply(lambda x: get_field(x, "icon_url") or "/placeholder.png")
    df["buy_limit"] = df["item_id"].apply(lambda x: get_field(x, "limit") or 0)

    return df


# ---------------------------------------------------------------------
# Safe JSON Conversion
# ---------------------------------------------------------------------
def df_to_safe_json(df: pd.DataFrame):
    """Safely convert any DataFrame to JSON-serializable list of dicts."""
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


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def load_latest_predictions() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "latest_top_flips.csv"
    if not path.exists() or path.stat().st_size == 0:
        print(f"⚠️ No buy recommendations found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    core = [
        "item_id", "buy_price", "sell_price", "profit_pct",
        "predicted_profit_gp", "buy_limit", "volatility_1h"
    ]
    df = df[[c for c in core if c in df.columns]]
    df = enrich_with_metadata(df)
    print(f"✅ Loaded {len(df)} buy recs (cols: {list(df.columns)})")
    return df


def load_sell_signals() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "sell_signals.csv"
    if not path.exists() or path.stat().st_size == 0:
        print(f"⚠️ No sell signals found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = enrich_with_metadata(df)
    cols = [
        "item_id", "name", "icon_url", "current_price", "should_sell",
        "reason", "confidence", "profit_pct", "hold_hours", "urgency_score"
    ]
    df = df[[c for c in cols if c in df.columns]]
    print(f"✅ Loaded {len(df)} sell recs (cols: {list(df.columns)})")
    return df


def load_active_flips() -> pd.DataFrame:
    if not ACTIVE_FLIPS_PATH.exists() or ACTIVE_FLIPS_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=[
            "item_id", "name", "icon_url", "entry_price",
            "entry_time", "current_price", "profit_pct", "profit_gp", "hold_hours"
        ])
    df = pd.read_csv(ACTIVE_FLIPS_PATH)
    df = enrich_with_metadata(df)
    return df


def save_active_flips(df: pd.DataFrame):
    df.to_csv(ACTIVE_FLIPS_PATH, index=False)
    print(f"💾 Saved {len(df)} active flips → {ACTIVE_FLIPS_PATH}")


# ---------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------
@router.get("/flips/buy-recommendations")
def get_buy_recommendations():
    df = load_latest_predictions()
    data = df_to_safe_json(df)
    return JSONResponse({"count": len(data), "data": data})


@router.get("/flips/sell-signals")
def get_sell_signals():
    df = load_sell_signals()
    data = df_to_safe_json(df)
    return JSONResponse({"count": len(data), "data": data})


@router.get("/flips/active")
def get_active():
    df = load_active_flips()
    data = df_to_safe_json(df)
    return JSONResponse({"count": len(data), "data": data})


@router.post("/flips/add/{item_id}")
def add_active(item_id: int):
    buys = load_latest_predictions()
    match = buys.loc[buys["item_id"] == item_id]
    if match.empty:
        return JSONResponse({"error": f"Item {item_id} not found"}, status_code=404)

    entry = match.iloc[0].to_dict()
    entry.update({
        "entry_price": float(entry.get("buy_price", 0)),
        "entry_time": datetime.utcnow().isoformat(),
        "current_price": entry.get("buy_price", 0),
        "profit_pct": 0.0,
        "profit_gp": 0.0,
        "hold_hours": 0.0,
    })

    df = load_active_flips()
    if item_id in df["item_id"].values:
        return JSONResponse({"status": "exists", "item_id": item_id})

    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    save_active_flips(df)
    return JSONResponse({"status": "added", "item_id": item_id})


@router.post("/flips/close/{item_id}")
def close_flip(item_id: int):
    df = load_active_flips()
    if df.empty or item_id not in df["item_id"].values:
        return JSONResponse({"status": "not_found", "item_id": item_id})

    df = df[df["item_id"] != item_id]
    save_active_flips(df)
    return JSONResponse({"status": "closed", "item_id": item_id})
