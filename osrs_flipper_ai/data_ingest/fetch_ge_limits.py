#!/usr/bin/env python3
"""
fetch_ge_limits.py — Fetch OSRS item metadata and buy limits
Outputs:
  • ge_limits.json — {item_id: buy_limit}
  • item_mapping.json — full metadata for dashboard and enrichment
"""

import requests, json
from pathlib import Path

DATA_DIR = Path("/root/osrs_flipper_ai/osrs_flipper_ai/data")
OUT_LIMITS = DATA_DIR / "ge_limits.json"
OUT_MAPPING = DATA_DIR / "item_mapping.json"
URL_MAPPING = "https://prices.runescape.wiki/api/v1/osrs/mapping"

def fetch_ge_limits():
    headers = {"User-Agent": "osrs-flipper-ai (contact: matthew@example.com)"}

    print("📦 Fetching OSRS item mapping (buy limits + metadata)...")
    try:
        resp = requests.get(URL_MAPPING, headers=headers, timeout=45)
        resp.raise_for_status()
        mapping = resp.json()
        print(f"✅ Retrieved {len(mapping):,} total items from /mapping.")
    except Exception as e:
        print(f"❌ Failed to fetch mapping: {e}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Write full item mapping (for dashboard)
    with open(OUT_MAPPING, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"💾 Saved full item metadata → {OUT_MAPPING}")

    # Extract and save buy limits
    limits = {}
    null_count = 0
    zero_count = 0

    for item in mapping:
        try:
            item_id = int(item.get("id"))
            limit = item.get("limit")
            if limit is None or str(limit).strip() == "":
                null_count += 1
                continue
            limit = int(limit)
            if limit <= 0:
                zero_count += 1
                continue
            limits[item_id] = limit
        except Exception:
            continue

    with open(OUT_LIMITS, "w") as f:
        json.dump(limits, f, indent=2)

    print(f"✅ Wrote {len(limits):,} valid buy limits → {OUT_LIMITS}")
    print(f"🧹 Skipped {null_count:,} items with NULL limits.")
    print(f"🧹 Skipped {zero_count:,} items with limit = 0.")

if __name__ == "__main__":
    fetch_ge_limits()
