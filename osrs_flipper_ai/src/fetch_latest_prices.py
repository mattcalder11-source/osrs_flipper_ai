"""
fetch_latest_prices.py — Live OSRS GE price fetcher (Wiki API)
--------------------------------------------------------------
Provides helper functions to fetch current item prices as a
dictionary {item_id: mid_price}.
"""

import requests
import pandas as pd
import time

WIKI_API = "https://prices.runescape.wiki/api/v1/osrs/latest"
USER_AGENT = "osrs-flipper-ai/1.0 (contact: youremail@example.com)"


def fetch_latest_prices_dict(retries=3, sleep_sec=1):
    """
    Fetch current OSRS item prices (mid price per item_id).

    Returns
    -------
    dict : {item_id: mid_price}
    """
    for attempt in range(retries):
        try:
            headers = {"User-Agent": USER_AGENT}
            r = requests.get(WIKI_API, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()["data"]

            prices = {}
            for item_id, v in data.items():
                try:
                    high = v.get("high")
                    low = v.get("low")
                    if high and low:
                        prices[int(item_id)] = (high + low) / 2
                except Exception:
                    continue

            if prices:
                print(f"✅ Loaded {len(prices):,} current prices from Wiki.")
                return prices

        except Exception as e:
            print(f"⚠️ Failed to fetch prices (attempt {attempt+1}/{retries}): {e}")
            time.sleep(sleep_sec)

    print("❌ Could not fetch live prices — returning empty dict.")
    return {}


def fetch_latest_prices_df():
    """
    Same as above, but returns a DataFrame with columns:
        item_id | high | low | mid_price
    """
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(WIKI_API, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]

    records = []
    for item_id, v in data.items():
        high = v.get("high")
        low = v.get("low")
        if high and low:
            records.append({
                "item_id": int(item_id),
                "high": high,
                "low": low,
                "mid_price": (high + low) / 2
            })
    return pd.DataFrame(records)
