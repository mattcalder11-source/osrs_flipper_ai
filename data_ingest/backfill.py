"""
backfill_all.py - Fetches /timeseries data for *all* tradeable items from the OSRS Wiki API.
Stable version: retry logic, backoff, random jitter, and polite cooldowns.
"""

import os
import time
import json
import random
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from osrs_flipper_ai.config import BASE_URL, USER_AGENT, DATA_DIR

# --- Setup ---
load_dotenv()
HEADERS = {"User-Agent": f"{USER_AGENT}"}


# -----------------------------------------
# Safe GET with retries and exponential backoff
# -----------------------------------------
def safe_get(url, params=None, retries=1, backoff_factor=1.0, max_sleep=0.15):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=60)

            # Handle common error codes gracefully
            if r.status_code == 200:
                return r
            elif r.status_code == 400:
                # Permanent — skip item
                print(f"❌ 400 Bad Request for {params} — skipping.")
                return None
            elif r.status_code in (429, 502, 503):
                # Temporary (rate limit or timeout)
                wait = min(backoff_factor ** attempt + random.uniform(0.1, 0.15), max_sleep)
                print(f"⚠️ [{r.status_code}] Retry {attempt}/{retries}, sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue
            else:
                print(f"⚠️ Unexpected {r.status_code} from {url}: {r.text[:120]}")
                return None

        except requests.exceptions.RequestException as e:
            wait = min(backoff_factor ** attempt + random.uniform(0.1, 0.15), max_sleep)
            print(f"⚠️ Network error on attempt {attempt}: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    print(f"❌ Failed after {retries} attempts: {url}")
    return None


# -----------------------------------------
# API Wrappers
# -----------------------------------------
def get_mapping():
    r = safe_get(f"{BASE_URL}/mapping")
    return r.json() if r else []


def get_timeseries(item_id, timeframe="5m"):
    """Fetch historical price data for one item (updated for new API format)."""
    url = f"{BASE_URL}/timeseries"
    params = {"id": item_id, "timestep": timeframe}
    r = safe_get(url, params=params)
    if not r:
        return []
    try:
        data = r.json().get("data", [])
        if not isinstance(data, list):
            print(f"⚠️ Unexpected data format for item {item_id}")
            return []
        return data
    except Exception as e:
        print(f"⚠️ Failed to parse JSON for item {item_id}: {e}")
        return []


# -----------------------------------------
# Main backfill routine
# -----------------------------------------
def backfill_all_items(timeframe="5m", out_dir="data/raw/all_items", batch_size=10):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Fetching item mapping…")
    mapping = get_mapping()
    df_map = pd.DataFrame(mapping)

    # Handle schema changes
    tradable_field = None
    for field in ["tradeable_on_ge", "tradeable"]:
        if field in df_map.columns:
            tradable_field = field
            break

    if tradable_field:
        df_map = df_map[df_map[tradable_field] == True]
    else:
        print("⚠️ No tradability column found — proceeding with all items.")

    print(f"Found {len(df_map)} items to fetch.")

    for i, row in enumerate(df_map.itertuples(), 1):
        item_id = row.id
        name = getattr(row, "name", f"item_{item_id}")
        outfile = Path(out_dir) / f"{item_id}_{timeframe}.json"

        if outfile.exists():
            continue

        data = get_timeseries(item_id, timeframe)
        if not data:
            print(f"[{i}/{len(df_map)}] ⚠️ No data for {name} ({item_id})")
        else:
            with open(outfile, "w") as f:
                json.dump(data, f)
            print(f"[{i}/{len(df_map)}] ✅ {name} ({item_id}) — {len(data)} points")

        # Polite pacing
        time.sleep(random.uniform(0.1, 0.15))
        if i % batch_size == 0:
            cooldown = random.uniform(1, 3)
            print(f"🕐 Cooling down for {cooldown:.1f}s...")
            time.sleep(cooldown)

    print("\n✅ Backfill complete.")


# -----------------------------------------
# Merge all JSON → Parquet
# -----------------------------------------
def merge_to_parquet(in_dir="data/raw/all_items", timeframe="5m", out_path="data/osrs_all_timeseries_5m.parquet"):
    rows = []
    for file in Path(in_dir).glob(f"*_{timeframe}.json"):
        item_id = int(file.stem.split("_")[0])
        with open(file) as f:
            data = json.load(f)
        for d in data:
            d["item_id"] = item_id
        rows.extend(data)

    if not rows:
        print("⚠️ No data found to merge.")
        return None

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.sort_values(["item_id", "timestamp"], inplace=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="snappy")
    print(f"✅ Wrote {len(df):,} rows to {out_path}")
    return df


if __name__ == "__main__":
    backfill_all_items(timeframe="5m")
    merge_to_parquet(in_dir="data/raw/all_items", timeframe="5m", out_path="data/osrs_all_timeseries_5m.parquet")
