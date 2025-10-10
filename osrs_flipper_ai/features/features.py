from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import gc
import numpy as np
import pandas as pd
from typing import Optional

EPS = 1e-9

# ----------------------------
# SCHEMA NORMALIZATION
# ----------------------------
def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw snapshot columns from historical or live data into a standard format
    expected by compute_features().
    Ensures presence of: high, low, volume, timestamp, item_id.
    """
    rename_map = {
        # Possible API variations
        "avgHighPrice": "avg_high_price",
        "avgLowPrice": "avg_low_price",
        "highPrice": "high",
        "lowPrice": "low",
        "avgHigh": "avg_high_price",
        "avgLow": "avg_low_price",
        "high": "high",
        "low": "low",
        "timestamp": "timestamp",
        "volume": "volume",
        "tradeVolume": "volume",
        "id": "item_id",
        "itemId": "item_id"
    }

    # Rename if present
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Derive missing price columns
    if "high" not in df.columns:
        if "avg_high_price" in df.columns:
            df["high"] = df["avg_high_price"]
        else:
            df["high"] = np.nan

    if "low" not in df.columns:
        if "avg_low_price" in df.columns:
            df["low"] = df["avg_low_price"]
        else:
            df["low"] = np.nan

    # Derive missing volume
    if "volume" not in df.columns:
        df["volume"] = np.nan

    # Convert timestamp if necessary
    if "timestamp" in df.columns:
        if np.issubdtype(df["timestamp"].dtype, np.number):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Ensure item_id exists
    if "item_id" not in df.columns:
        raise KeyError("❌ Missing 'item_id' column in snapshot — cannot continue.")

    # Drop rows without timestamp or item_id
    df = df.dropna(subset=["timestamp", "item_id"])

    print(f"✅ Normalized schema: {list(df.columns)}")
    return df


# ----------------------------
# TECHNICAL INDICATORS
# ----------------------------
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_schema(df)
    df = df.sort_values(["item_id", "timestamp"])

    results = []
    item_ids = df["item_id"].unique()

    for i, item_id in enumerate(item_ids):
        g = df.loc[df["item_id"] == item_id].copy()
        if g.empty:
            continue

        price = (
            g["avg_high_price"].ffill().fillna(g["high"])
            if "avg_high_price" in g.columns
            else g["high"]
        )

        # RSI
        delta = price.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(span=14, adjust=False).mean()
        roll_down = down.ewm(span=14, adjust=False).mean()
        rs = roll_up / (roll_down + EPS)
        g["rsi"] = (100 - (100 / (1 + rs))).fillna(50.0)

        # ROC & MACD
        g["roc"] = (price / price.shift(12) - 1) * 100
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        g["macd"] = macd
        g["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

        # Contrarian flags
        g["contrarian_flag"] = 0
        g.loc[g["rsi"] > 70, "contrarian_flag"] = -1
        g.loc[g["rsi"] < 30, "contrarian_flag"] = 1

        # Normalize columns (0-1)
        for col in ("rsi", "roc", "macd"):
            col_min = g[col].min(skipna=True)
            col_max = g[col].max(skipna=True)
            denom = (col_max - col_min) if (col_max - col_min) > EPS else EPS
            g[f"{col}_norm"] = (g[col] - col_min) / denom

        g["volatility_1h"] = g.get("volatility_1h", 0.0)

        # Composite technical score (0–5)
        liq_den = g["daily_volume"].max() + EPS
        vol_den = g["volatility_1h"].max() + EPS
        g["technical_score"] = (
            2 * g["rsi_norm"]
            + 1 * g["roc_norm"]
            + 1 * g["macd_norm"]
            + 0.5 * (g["daily_volume"] / liq_den)
            + 0.5 * (1 - (g["volatility_1h"] / vol_den))
            + g["contrarian_flag"] * 0.5
        ).clip(0, 5)

        g = g.astype({c: np.float32 for c in g.select_dtypes("float64")})
        results.append(g)

        # incremental cleanup to avoid memory balloon
        if i % 100 == 0:
            gc.collect()

    return pd.concat(results, ignore_index=True)


# ----------------------------
# FEATURE COMPUTATION
# ----------------------------
def compute_features_in_chunks(df, batch_size=1000, out_path=None):
    """
    Compute features in small chunks to prevent OOM kills.
    Writes incremental parquet batches if out_path is provided.
    """
    results = []
    item_ids = df["item_id"].unique()
    total = len(item_ids)
    print(f"🧩 Processing {total:,} unique items...")

    for i, item_id in enumerate(item_ids):
        g = df[df["item_id"] == item_id].copy()
        if g.empty:
            continue

        # ---- Same feature logic ----
        g["mid_price"] = ((g["high"] + g["low"]) / 2).astype(np.float32)
        g["spread"] = (g["high"] - g["low"]).astype(np.float32)
        g["spread_ratio"] = (g["spread"] / g["mid_price"]).astype(np.float32)
        g["volatility_1h"] = (
            g["mid_price"].pct_change(fill_method=None).rolling(12, min_periods=1).std()
        ).astype(np.float32)
        g["timestamp"] = pd.to_datetime(g["timestamp"], errors="coerce")
        g["target_profit_ratio"] = (g["high"] / g["low"] - 1).clip(-0.2, 2.0)

        results.append(g)

        # periodic write to disk
        if (i + 1) % batch_size == 0:
            batch_df = pd.concat(results, ignore_index=True)
            if out_path:
                mode = "a" if out_path.exists() else "w"
                batch_df.to_parquet(out_path, index=False)
            results = []  # clear from memory
            print(f"✅ Processed {i+1:,}/{total:,} items...")

    # final write
    if results and out_path:
        batch_df = pd.concat(results, ignore_index=True)
        batch_df.to_parquet(out_path, index=False)
        print(f"💾 Final batch written ({len(batch_df):,} rows).")
    elif results:
        return pd.concat(results, ignore_index=True)

# ----------------------------
# FEATURE LOADING
# ----------------------------
def load_recent_features(folder: str = "osrs_flipper_ai/data/features", days_back: int = 30, max_files: int = 100) -> pd.DataFrame:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"No such folder: {folder}")

    files = sorted(folder.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder}")

    now = datetime.now(timezone.utc)
    frames = []

    for p in files[:max_files]:
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        if (now - mtime).days > days_back:
            continue
        try:
            df = pd.read_parquet(p)
            if df.empty:
                continue
            df = normalize_schema(df)
            frames.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {p}: {e}")

    if not frames:
        raise ValueError("No recent valid feature snapshots found")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["item_id", "timestamp"])
    combined = combined.drop_duplicates(subset=["item_id", "timestamp"], keep="last").reset_index(drop=True)
    combined = compute_features_in_chunks(combined)
    gc.collect()
    return combined

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser(description="Compute OSRS item features from raw snapshot(s)")
    parser.add_argument("--input", type=str, default=None, help="Input .parquet path (or folder)")
    parser.add_argument("--output", type=str, default=None, help="Output .parquet path")
    args = parser.parse_args()

    RAW_DIR = Path("/root/osrs_flipper_ai/osrs_flipper_ai/data/raw")
    OUT_DIR = Path("/root/osrs_flipper_ai/osrs_flipper_ai/osrs_flipper_ai/data/features")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = Path(args.input)
    else:
        # Default: use latest snapshot
        snapshots = sorted(RAW_DIR.glob("snapshot_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not snapshots:
            raise FileNotFoundError(f"No snapshot parquet files found in {RAW_DIR}")
        input_path = snapshots[0]

    print(f"🚀 Loading snapshot: {input_path}")
    df_raw = pd.read_parquet(input_path)
    df_raw = normalize_schema(df_raw)

    print(f"🧠 Computing features for {len(df_raw):,} rows...")
    out_path = Path(args.output) if args.output else Path("/root/osrs_flipper_ai/osrs_flipper_ai/data/features/features_historical.parquet")
    df_features = compute_features_in_chunks(df_raw, batch_size=500)

    if df_features.empty:
        print("⚠️ No features generated — nothing to save.")
    else:
        ts = int(time.time())
        out_path = Path(args.output) if args.output else OUT_DIR / f"features_{ts}.parquet"
        latest_path = OUT_DIR / "features_latest.parquet"

        df_features.to_parquet(out_path, index=False)
        df_features.to_parquet(latest_path, index=False)
        print(f"✅ Saved {len(df_features):,} features → {out_path}")
