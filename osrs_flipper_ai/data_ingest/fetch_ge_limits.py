#!/usr/bin/env python3
"""
fetch_ge_limits.py — Download and cache OSRS GE buy limits.
"""

import requests, json, os
from pathlib import Path

OUT_PATH = Path("/root/osrs_flipper_ai/osrs_flipper_ai/data/ge_limits.json")
URL = "https://prices.runescape.wiki/api/v1/osrs/mapping"

def fetch_ge_limits():
    print("📦 Fetching item mapping from RuneScape Wiki...")
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    mapping = r.json()

    # Build dict: item_id → limit
    limits = {int(item["id"]): item.get("limit", None) for item in mapping if "id" in item}
    with open(OUT_PATH, "w") as f:
        json.dump(limits, f, indent=2)
    print(f"✅ Wrote {len(limits):,} buy limits → {OUT_PATH}")

if __name__ == "__main__":
    fetch_ge_limits()
