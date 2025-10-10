#!/usr/bin/env python3
"""
fetch_ge_limits.py — robust OSRS GE buy-limit fetcher (filters untradeables)
"""

import requests, json
from pathlib import Path

OUT_PATH = Path("/root/osrs_flipper_ai/osrs_flipper_ai/data/ge_limits.json")

def fetch_ge_limits():
    headers = {"User-Agent": "osrs-flipper-ai (contact: matt.calder11@gmail.com)"}
    url_mapping = "https://prices.runescape.wiki/api/v1/osrs/mapping"

    print("📦 Fetching OSRS GE buy limits...")
    try:
        limits_resp = requests.get(url_mapping, headers=headers, timeout=30)
        limits_resp.raise_for_status()
        limits_data = limits_resp.json().get("data", {})
        print(f"✅ Retrieved {len(limits_data):,} raw limit entries from /limits.")
    except Exception as e:
        print(f"❌ Failed to fetch /limits: {e}")
        limits_data = {}

    print("🗺️  Fetching item mapping (tradeable flags)...")
    try:
        mapping_resp = requests.get(url_mapping, headers=headers, timeout=30)
        mapping_resp.raise_for_status()
        mapping_data = mapping_resp.json()
        print(f"✅ Retrieved {len(mapping_data):,} items from /mapping.")
    except Exception as e:
        print(f"❌ Failed to fetch /mapping: {e}")
        mapping_data = []

    # Determine which items are tradeable
    tradeable_ids = {
        int(item["id"])
        for item in mapping_data
        if item.get("tradeable", False) or item.get("tradeable_on_ge", False)
    }

    # Build dictionary of item_id → limit
    limits = {}
    missing = 0
    for item_id_str, info in limits_data.items():
        item_id = int(item_id_str)
        if item_id not in tradeable_ids:
            continue  # skip untradeables / unavailable items
        limit = info.get("limit")
        if limit is None:
            missing += 1
            limit = 100  # fallback default
        limits[item_id] = limit

    # Save results
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(limits, f, indent=2)

    print(f"✅ Wrote {len(limits):,} tradeable item limits → {OUT_PATH}")
    print(f"🧹 Filtered out {len(limits_data) - len(limits):,} untradeable items.")
    print(f"⚠️ {missing:,} items missing explicit limits (defaulted to 100).")

if __name__ == "__main__":
    fetch_ge_limits()
