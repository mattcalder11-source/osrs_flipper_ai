#!/usr/bin/env python3
"""
fetch_ge_limits.py — Fetch OSRS GE buy limits from /mapping
Filters out items with NULL or zero buy limits.
"""

import requests, json
from pathlib import Path

OUT_PATH = Path("/root/osrs_flipper_ai/osrs_flipper_ai/data/ge_limits.json")
URL_MAPPING = "https://prices.runescape.wiki/api/v1/osrs/mapping"

def fetch_ge_limits():
    headers = {"User-Agent": "osrs-flipper-ai (contact: matthew@example.com)"}

    print("📦 Fetching OSRS item mapping (buy limits)...")
    try:
        resp = requests.get(URL_MAPPING, headers=headers, timeout=45)
        resp.raise_for_status()
        mapping = resp.json()
        print(f"✅ Retrieved {len(mapping):,} total items from /mapping.")
    except Exception as e:
        print(f"❌ Failed to fetch mapping: {e}")
        return

    limits = {}
    null_count = 0
    zero_count = 0

    for item in mapping:
        try:
            item_id = int(item.get("id"))
            limit = item.get("limit")
            if limit is None:
                null_count += 1
                continue
            if isinstance(limit, str) and not limit.strip():
                null_count += 1
                continue
            limit = int(limit)
            if limit <= 0:
                zero_count += 1
                continue
            limits[item_id] = limit
        except Exception:
            continue

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(limits, f, indent=2)

    print(f"✅ Wrote {len(limits):,} valid item limits → {OUT_PATH}")
    print(f"🧹 Skipped {null_count:,} items with NULL limits.")
    print(f"🧹 Skipped {zero_count:,} items with limit = 0.")

if __name__ == "__main__":
    fetch_ge_limits()
