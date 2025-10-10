#!/usr/bin/env python3
"""
fetch_ge_limits.py — Fetch OSRS item metadata and buy limits
Outputs:
  • ge_limits.json — {item_id: buy_limit}
  • item_mapping.json — full metadata for dashboard and enrichment
"""

import requests
import json
import urllib.parse
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

    limits = {}
    enriched_mapping = {}
    null_count = 0
    zero_count = 0
    skipped_unusable = 0

    for item in mapping:
        try:
            item_id = int(item.get("id"))
            name = item.get("name")
            limit = item.get("limit")

            # Skip unusable entries
            if limit is None or str(limit).strip() == "":
                null_count += 1
                continue
            if int(limit) <= 0:
                zero_count += 1
                continue

            # Encode icon URL safely
            icon_name = item.get("icon", "")
            encoded_icon = urllib.parse.quote(icon_name.replace(" ", "_"))
            icon_url = f"https://oldschool.runescape.wiki/images/{encoded_icon}" if icon_name else ""

            # Build full mapping entry
            enriched_mapping[str(item_id)] = {
                "id": item_id,
                "name": name,
                "examine": item.get("examine"),
                "members": item.get("members"),
                "lowalch": item.get("lowalch"),
                "highalch": item.get("highalch"),
                "limit": int(limit),
                "icon": icon_name,
                "icon_url": icon_url,
            }

            limits[item_id] = int(limit)

        except Exception as e:
            skipped_unusable += 1
            continue

    # Write outputs
    with open(OUT_LIMITS, "w") as f:
        json.dump(limits, f, indent=2)
    with open(OUT_MAPPING, "w") as f:
        json.dump(enriched_mapping, f, indent=2)

    print(f"✅ Wrote {len(limits):,} valid buy limits → {OUT_LIMITS}")
    print(f"✅ Wrote {len(enriched_mapping):,} enriched items → {OUT_MAPPING}")
    print(f"🧹 Skipped {null_count:,} NULL-limit items, {zero_count:,} zero-limit items.")
    if skipped_unusable:
        print(f"🧹 Skipped {skipped_unusable:,} invalid entries during parsing.")


if __name__ == "__main__":
    fetch_ge_limits()
