#!/usr/bin/env python3
"""
OSRS Flipper AI - Data Ingest Script

Fetches item mapping, price data (latest, 5m, 1h),
and daily volume from the OSRS Wiki API.
Produces a unified parquet snapshot for downstream features & modeling.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"
HEADERS = {"User-Agent": "osrs-flipper-ai (contact: matthew@example.com)"}
DATA_DIR = Path("/root/osrs_flipper_ai/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# Fetch helpers
# -----------------------------------------------------------

def fetch_mapping() -> pd.DataFrame:
    """Fetch item mapping (metadata)."""
    url = f"{BASE_URL}/mapping"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df.rename(columns={"id": "item_id"}, inplace=True)
        df["item_id"] = df["item_id"].astype(int)
        print(f"✅ Fetched {len(df)} mapping rows from /mapping")
        return df
    except Exception as e:
        print(f"⚠️ Failed to fetch /mapping: {e}")
        return pd.DataFrame(columns=["item_id"])


def fetch(endpoint: str) -> pd.DataFrame:
    """Generic helper for /latest, /5m, /1h."""
    url = f"{BASE_URL}/{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", {})
        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient="index").reset_index()
            df.rename(columns={"index": "item_id"}, inplace=True)
            df["item_id"] = df["item_id"].astype(int)
            print(f"✅ Fetched {len(df)} rows from /{endpoint}")
            return df
        else:
            print(f"⚠️ Unexpected structure from /{endpoint}")
            return pd.DataFrame(columns=["item_id"])
    except Exception as e:
        print(f"⚠️ Failed to fetch /{endpoint}: {e}")
        return pd.DataFrame(columns=["item_id"])


def fetch_volumes() -> pd.DataFrame:
    """Fetch 24h trade volume data from OSRS Wiki API."""
    url = f"{BASE_URL}/volumes"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", {})
        df = pd.DataFrame(list(data.items()), columns=["item_id", "daily_volume"])
        df["item_id"] = df["item_id"].astype(int)
        print(f"✅ Fetched {len(df)} daily volume entries from /volumes")
        return df
    except Exception as e:
        print(f"⚠️ Failed to fetch /volumes: {e}")
        return pd.DataFrame(columns=["item_id", "daily_volume"])


# -----------------------------------------------------------
# Snapshot builder
# -----------------------------------------------------------

def build_snapshot():
    """Fetch latest OSRS price + volume data and save parquet snapshot."""
    print("⏳ Fetching OSRS price data...")

    mapping = fetch_mapping()
    latest = fetch("latest")
    five = fetch("5m")
    one_hour = fetch("1h")
    volumes = fetch_volumes()

    # --- Merge all sources ---
    df = mapping.merge(latest, on="item_id", how="left", suffixes=("", "_latest"))
    df = df.merge(five, on="item_id", how="left", suffixes=("", "_5m"))
    df = df.merge(one_hour, on="item_id", how="left", suffixes=("", "_1h"))

    # --- Add daily volumes + liquidity ---
    if not volumes.empty:
        df = df.merge(volumes, on="item_id", how="left")
        df["liquidity_1h"] = df["daily_volume"] / 24.0
        df["liquidity_1h"] = df["liquidity_1h"].fillna(0)
        print("💧 Added liquidity_1h from daily volume data.")
    else:
        df["liquidity_1h"] = 0
        print("⚠️ No volume data available — liquidity_1h set to 0.")

    # --- Metadata + Save ---
    df["timestamp"] = datetime.utcnow()
    ts = int(time.time())

    out_path = DATA_DIR / f"snapshot_{ts}.parquet"
    latest_path = DATA_DIR / "snapshot_latest.parquet"

    df.to_parquet(out_path, index=False)
    df.to_parquet(latest_path, index=False)

    print(f"✅ Snapshot saved to {out_path}")
    print(f"🕓 Records: {len(df):,}  Columns: {len(df.columns)}")


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    print("🚀 Starting OSRS Wiki data ingestion...")
    build_snapshot()
    print("✅ Data ingestion complete.")
