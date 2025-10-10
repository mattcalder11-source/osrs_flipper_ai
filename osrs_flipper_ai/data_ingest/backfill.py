"""
backfill.py - Fetches full historical OSRS GE data + daily volumes, matching ingest.py structure.
"""

import os
import time
import json
import random
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from osrs_flipper_ai.config import BASE_URL, USER_AGENT

# --- Setup ---
load_dotenv()
HEADERS = {"User-Agent": f"{USER_AGENT}"}

# Base output directories
BASE_DIR = Path(__file__).resolve().parents[1]  # /root/osrs_flipper_ai/osrs_flipper_ai
RAW_DIR = BASE_DIR / "data" / "raw"
ITEM_DIR = RAW_DIR / "all_items"
RAW_DIR.mkdir(parents=True, exist_ok=True)
ITEM_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# Safe GET with retries
# -----------------------------------------
def safe_get(url, params=None, retries=3, backoff_factor=1.5, max_sleep=2.0):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=60)
            if r.status_code == 200:
                return r
            elif r.status_code in (429, 502, 503):
                wait = min(backoff_factor**attempt + random.uniform(0.2, 0.5), max_sleep)
                print(f"⚠️ [{r.status_code}] Retry {attempt}/{retries}, sleeping {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"⚠️ Unexpected {r.status_code} from {url}")
                return None
        except requests.exceptions.RequestException as e:
            wait = min(backoff_factor**attempt + random.uniform(0.2, 0.5), max_sleep)
            print(f"⚠️ Network error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    print(f"❌ Failed after {retries} attempts: {url}")
    return None


# -----------------------------------------
# API wrappers
# -----------------------------------------
def get_mapping():
    r = safe_get(f"{BASE_URL}/mapping")
    return r.json() if r else []


def get_volumes():
    """Fetch 24h trade volumes."""
    r = safe_get(f"{BASE_URL}/volumes")
    if not r:
        return {}
    try:
        return r.json().get("data", {})
    except Exception:
        return {}


def get_timeseries(item_id, timeframe="5m"):
    url = f"{BASE_URL}/timeseries"
    params = {"id": item_id, "timestep": timeframe}
    r = safe_get(url, params=params)
    if not r:
        return []
    try:
        data = r.json().get("data", [])
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"⚠️ Failed to parse JSON for item {item_id}: {e}")
        return []


# -----------------------------------------
# Backfill all items
# -----------------------------------------
def backfill_all_items(timeframe="5m", batch_size=10):
    print("📦 Fetching item mapping…")
    mapping = get_mapping()
    df_map = pd.DataFrame(mapping)

    tradable_field = next((f for f in ["tradeable_on_ge", "tradeable"] if f in df_map.columns), None)
    if tradable_field:
        df_map = df_map[df_map[tradable_field] == True]

    print(f"🧩 Found {len(df_map)} tradeable items.")

    print("📊 Fetching 24h volumes…")
    volumes = get_volumes()
    print(f"✅ Retrieved {len(volumes)} volume entries.")

    for i, row in enumerate(df_map.itertuples(), 1):
        item_id = row.id
        name = getattr(row, "name", f"item_{item_id}")
        outfile = ITEM_DIR / f"{item_id}_{timeframe}.json"

        if outfile.exists():
            continue

        data = get_timeseries(item_id, timeframe)
        if not data:
            print(f"[{i}/{len(df_map)}] ⚠️ No data for {name} ({item_id})")
        else:
            for d in data:
                d["item_id"] = item_id
                d["name"] = name
                d["daily_volume"] = volumes.get(str(item_id), 0)

            with open(outfile, "w") as f:
                json.dump(data, f)

            print(f"[{i}/{len(df_map)}] ✅ {name} ({item_id}) — {len(data)} points")

        time.sleep(random.uniform(0.1, 0.2))
        if i % batch_size == 0:
            cooldown = random.uniform(1, 3)
            print(f"🕐 Cooling down for {cooldown:.1f}s...")
            time.sleep(cooldown)

    print("\n✅ Backfill complete.")


# -----------------------------------------
# Merge all JSON → Parquet
# -----------------------------------------
def merge_to_parquet_safe(timeframe="5m", batch_size=500):
    """
    Incrementally merges item JSONs into a single Parquet file without using too much RAM.
    """
    files = sorted(ITEM_DIR.glob(f"*_{timeframe}.json"))
    total = len(files)
    if total == 0:
        print("⚠️ No JSON files found.")
        return

    print(f"📂 Merging {total:,} files into snapshot parquet (batch size {batch_size})...")
    temp_parquets = []

    for i in range(0, total, batch_size):
        batch_files = files[i:i+batch_size]
        batch_rows = []
        for f in batch_files:
            try:
                with open(f) as j:
                    data = json.load(j)
                    batch_rows.extend(data)
            except Exception as e:
                print(f"⚠️ Skipping {f.name}: {e}")

        if not batch_rows:
            continue

        df_batch = pd.DataFrame(batch_rows)
        df_batch["timestamp"] = pd.to_datetime(df_batch["timestamp"], unit="s", errors="coerce")
        df_batch = df_batch.dropna(subset=["timestamp"])
        df_batch.sort_values(["item_id", "timestamp"], inplace=True)

        tmp_path = RAW_DIR / f"_tmp_batch_{i//batch_size}.parquet"
        df_batch.to_parquet(tmp_path, compression="snappy")
        temp_parquets.append(tmp_path)
        print(f"✅ Wrote batch {i//batch_size+1} → {len(df_batch):,} rows")

        del df_batch, batch_rows

    print("📦 Concatenating batch parquets...")
    df_iter = (pd.read_parquet(p) for p in temp_parquets)
    df = pd.concat(df_iter, ignore_index=True)

    ts = int(time.time())
    out_path = RAW_DIR / f"snapshot_{ts}.parquet"
    latest_path = RAW_DIR / "snapshot_latest.parquet"

    df.to_parquet(out_path, compression="snappy")
    df.to_parquet(latest_path, compression="snappy")

    print(f"✅ Wrote {len(df):,} total rows to {out_path}")

    # cleanup
    for p in temp_parquets:
        p.unlink(missing_ok=True)

    return df


# -----------------------------------------
# Entry point
# -----------------------------------------
if __name__ == "__main__":
    backfill_all_items(timeframe="5m")
    merge_to_parquet_safe(timeframe="5m")
