#!/usr/bin/env python3
"""
OSRS Flipper AI - Data Ingest Script

Fetches item mapping, latest price data, short-term price history,
and daily volume from the OSRS Wiki API. Produces a unified parquet
snapshot for downstream feature generation and model training.
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
# Helpers
# -----------------------------------------------------------
def fetch(endpoint: str) -> pd.DataFrame:
    """Generic helper to fetch and parse JSON data from the Wiki API."""
    url = f"{BASE_URL}/{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", {})

        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient="index").reset_index()
            df.rename(columns={"index": "item_id"}, inplace=True)
        else:
            df = pd.DataFrame(data)

        df["item_id"] = df["item_id"].astype(int)
        print(f"✅ Fetched {len(df)} rows from /{endpoint}")
        return df

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

    mapping = fetch("mapping")
    latest = fetch("latest")
    five = fetch("5m")
    one_hour = fetch("1h")
    volumes = fetch_volumes()

    # Merge all dataframes
    df = mapping.merge(latest, on="item_id", how="left", suffixes=("", "_latest"))
    df = df.merge(five, on="item_id", how="left", suffixes=("", "_5m"))
    df = df.merge(one_hour, on="item_id", how="left", suffixes=("", "_1h"))

    # Merge daily volumes → derive liquidity_1h
    if not volumes.empty:
        df = df.merge(volumes, on="item_id", how="left")
        df["liquidity_1h"] = df["daily_volume"] / 24.0
        df["liquidity_1h"] = df["liquidity_1h"].fillna(0)
        print("💧 Added liquidity_1h from daily volume data.")
    else:
        df["liquidity_1h"] = 0
        print("⚠️ No volume data available — liquidity_1h set to 0.")

    # Add timestamp
    df["timestamp"] = datetime.utcnow()

    # Save snapshot
    ts = int(time.time())
    out_path = DATA_DIR / f"snapshot_{ts}.parquet"
    df.to_parquet(out_path, index=False)
    latest_path = DATA_DIR / "snapshot_latest.parquet"
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
