"""
prepare_historical_features.py - Converts raw backfill data into feature-compatible format
for use with train_model.py.
"""

import pandas as pd
from osrs_flipper_ai.features.features import compute_features, compute_technical_indicators
from pathlib import Path

RAW_PATH = "data/osrs_all_timeseries_5m.parquet"
OUT_PATH = "data/features/latest_train.parquet"

def prepare_historical_features():
    print(f"📦 Loading {RAW_PATH}...")
    df = pd.read_parquet(RAW_PATH)

    # --- Normalize columns ---
    rename_map = {
        "avgHighPrice": "high",
        "avgLowPrice": "low",
        "highPriceVolume": "highPriceVolume",
        "lowPriceVolume": "lowPriceVolume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Add derived columns to match model expectations
    df["ts_utc"] = df["timestamp"].astype("int64") // 10**9
    df["mid_price"] = (df["high"] + df["low"]) / 2
    df["margin_pct"] = (df["high"] - df["low"]) / df["low"]  # proxy target for initial training

    # Compute engineered + technical features
    try:
        df = compute_features(df)
        print("✅ compute_features() applied.")
    except Exception as e:
        print(f"⚠️ compute_features() failed: {e}")

    try:
        df = compute_technical_indicators(df)
        print("✅ compute_technical_indicators() applied.")
    except Exception as e:
        print(f"⚠️ Technical indicators failed: {e}")

    # Drop incomplete rows
    df = df.dropna(subset=["mid_price", "margin_pct"])
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"💾 Saved feature dataset → {OUT_PATH} ({len(df):,} rows)")

if __name__ == "__main__":
    prepare_historical_features()
