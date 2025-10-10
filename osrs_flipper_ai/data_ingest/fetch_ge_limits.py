#!/usr/bin/env python3
"""
fetch_ge_limits.py — Download and cache OSRS GE buy limits
(fully based on /mapping endpoint, filters untradeable/unavailable items)
"""

import requests, json
from pathlib import Path

OUT_PATH = Path("/root/osrs_flipper_ai/osrs_flipper_ai/data/ge_limits.json")
URL_MAPPING = "https://prices.runescape.wiki/api/v1/osrs/mapping"

def fetch_ge_limits():
    headers = {"User-Agent": "osrs-flipper-ai (contact: matthew@example.com)"}

    print("📦 Fetching item mapping (buy limits + tradeable flags)...")
    try:
        resp = requests.get(URL_MAPPING, headers=headers, timeout=45)
        resp.raise_for_status()
        mapping = resp.json()
        print(f"✅ Retrieved {len(mapping):,} total mapping entries.")
    except Exception as e:
        print(f"❌ Failed to fetch mapping: {e}")
        mapping = []

    # Build dictionary: item_id → buy_limit (for tradeable items only)
    limits = {}
    missing_limit = 0
    skipped_untradeable = 0

    for item in mapping:
        try:
            item_id = int(item["id"])
            # Skip untradeable or unavailable items
            if not (item.get("tradeable") or item.get("tradeable_on_ge")):
                skipped_untradeable += 1
                continue

            limit = item.get("buy_limit")
            # Skip invalid limits (null, 0, etc.)
            if limit is None or limit == 0:
                missing_limit += 1
                limit = 100  # default fallback to avoid NaNs

            limits[item_id] = limit
        except Exception:
            continue

    # Write file
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(limits, f, indent=2)

    print(f"✅ Wrote {len(limits):,} tradeable item limits → {OUT_PATH}")
    print(f"🧹 Skipped {skipped_untradeable:,} untradeable/unavailable items.")
    print(f"⚠️ {missing_limit:,} items had no explicit limit (defaulted to 100).")

if __name__ == "__main__":
    fetch_ge_limits()
