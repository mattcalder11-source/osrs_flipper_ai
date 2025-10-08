# src/ingest.py

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from osrs_flipper_ai.config import BASE_URL, USER_AGENT, DATA_DIR
from osrs_flipper_ai.features.features import compute_features 

# --- Ensure folders exist ---
os.makedirs(DATA_DIR, exist_ok=True)
FEATURE_DIR = os.path.join(DATA_DIR, "features")
LOG_DIR = "logs"
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

HEADERS = {"User-Agent": USER_AGENT}


# --- Fetch helper ---
def fetch(endpoint):
    """Fetch data from the OSRS Wiki API, handling both dict and list responses."""
    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    # /mapping returns a list
    if isinstance(data, list):
        return data

    # /latest, /5m, /1h return dicts with 'data' key
    if isinstance(data, dict) and "data" in data:
        return data["data"]

    # Fallback
    return data


# --- Main snapshot job ---
def snapshot():
    ts = int(datetime.now(timezone.utc).timestamp())

    try:
        print("⏳ Fetching OSRS price data...")
        mapping = fetch("mapping")
        latest = fetch("latest")
        five = fetch("5m")
        one_hour = fetch("1h")
    except Exception as e:
        log(f"❌ Fetch failed: {e}")
        return

    # Convert mapping list → dictionary
    mapping_dict = {str(item["id"]): item for item in mapping}

    rows = []
    for item_id, d in latest.items():
        name = mapping_dict.get(str(item_id), {}).get("name", "")
        row = {
            "ts_utc": ts,
            "item_id": int(item_id),
            "name": name,
            "high": d.get("high"),
            "low": d.get("low"),
            "highTime": d.get("highTime"),
            "lowTime": d.get("lowTime"),
        }

        # Merge 5m and 1h data
        if str(item_id) in five:
            row["avg_5m_high"] = five[str(item_id)].get("high")
            row["avg_5m_low"] = five[str(item_id)].get("low")
        if str(item_id) in one_hour:
            row["avg_1h_high"] = one_hour[str(item_id)].get("high")
            row["avg_1h_low"] = one_hour[str(item_id)].get("low")

        rows.append(row)

    # --- Save snapshot ---
    df = pd.DataFrame(rows)
    snapshot_file = os.path.join(DATA_DIR, f"snapshot_{ts}.parquet")
    df.to_parquet(snapshot_file, index=False)
    print(f"✅ Snapshot saved: {snapshot_file} ({len(df)} items)")
    log(f"Snapshot saved: {snapshot_file} ({len(df)} items)")

    # --- Compute features ---
    try:
        features = compute_features(df)
        feature_file = os.path.join(FEATURE_DIR, f"features_{ts}.parquet")
        features.to_parquet(feature_file, index=False)
        print(f"✨ Features computed and saved: {feature_file}")
        log(f"Features computed: {feature_file} ({len(features)} rows)")
    except Exception as e:
        print(f"❌ Error computing features: {e}")
        log(f"❌ Error computing features: {e}")
        return

    # --- Aggregate features ---
    try:
        aggregate_recent_features(FEATURE_DIR, days=7)
    except Exception as e:
        print(f"⚠ Error aggregating features: {e}")
        log(f"⚠ Error aggregating features: {e}")


# --- Aggregation helper ---
def aggregate_recent_features(feature_dir: str, days: int = 7):
    """Combine recent feature files into one rolling dataset."""
    files = sorted(
        [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".parquet")],
        key=os.path.getmtime,
        reverse=True,
    )

    cutoff = datetime.now() - timedelta(days=days)
    valid_files = [f for f in files if datetime.fromtimestamp(os.path.getmtime(f)) > cutoff]

    if not valid_files:
        print("⚠ No recent feature files to aggregate.")
        return

    dfs = []
    for f in valid_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"⚠ Skipping corrupt file {f}: {e}")

    if not dfs:
        print("⚠ No valid feature data to aggregate.")
        return

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["ts_utc", "item_id"])
    combined = combined.sort_values(["item_id", "ts_utc"])

    output_file = os.path.join(feature_dir, "latest_train.parquet")
    combined.to_parquet(output_file, index=False)
    print(f"📊 Aggregated {len(valid_files)} files → {output_file} ({len(combined)} rows)")
    log(f"Aggregated {len(valid_files)} → {output_file} ({len(combined)} rows)")


# --- Logging helper ---
def log(message: str):
    """Append timestamped messages to logs/runtime.log"""
    with open(os.path.join(LOG_DIR, "runtime.log"), "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


# --- Entry point ---
if __name__ == "__main__":
    snapshot()
