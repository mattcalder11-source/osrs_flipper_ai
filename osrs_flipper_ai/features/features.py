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
    df = df.copy()

    # Ensure item_id present
    if "item_id" not in df.columns:
        if "item_id" in df.index.names:
            df = df.reset_index()
        else:
            raise KeyError("Missing 'item_id' column")

    # Normalize timestamp
    if "timestamp" not in df.columns:
        for alt in ("ts_utc", "time", "datetime"):
            if alt in df.columns:
                df = df.rename(columns={alt: "timestamp"})
                break
        else:
            raise KeyError("Missing timestamp column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Price + volume columns
    rename_map = {
        "high_price": "high",
        "avg_high_price": "avg_high_price",
        "low_price": "low",
        "avg_low_price": "avg_low_price",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "volume" not in df.columns:
        for alt in ("trade_volume", "buy_volume", "sell_volume", "qty"):
            if alt in df.columns:
                df = df.rename(columns={alt: "volume"})
                break
        else:
            df["volume"] = 0.0

    for c in ("high", "low", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

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

        g["liquidity_1h"] = g.get("liquidity_1h", 0.0)
        g["volatility_1h"] = g.get("volatility_1h", 0.0)

        # Composite technical score (0–5)
        liq_den = g["liquidity_1h"].max() + EPS
        vol_den = g["volatility_1h"].max() + EPS
        g["technical_score"] = (
            2 * g["rsi_norm"]
            + 1 * g["roc_norm"]
            + 1 * g["macd_norm"]
            + 0.5 * (g["liquidity_1h"] / liq_den)
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
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_schema(df)
    df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    df["mid_price"] = ((df["high"] + df["low"]) / 2).astype(np.float32)
    df["spread"] = (df["high"] - df["low"]).astype(np.float32)
    df["spread_ratio"] = (df["spread"] / (df["mid_price"].abs() + EPS)).astype(np.float32)

    df["avg_high_price"] = (
        df.groupby("item_id")["high"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    ).astype(np.float32)
    df["avg_low_price"] = (
        df.groupby("item_id")["low"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    ).astype(np.float32)

    df["volatility_1h"] = (
        df.groupby("item_id")["mid_price"]
        .transform(lambda x: x.pct_change().rolling(12, min_periods=3).std())
        .fillna(0.0)
    ).astype(np.float32)

    df["liquidity_1h"] = (
        df.groupby("item_id")["volume"].transform(lambda x: x.rolling(12, min_periods=1).mean())
    ).fillna(0.0).astype(np.float32)

    df = compute_technical_indicators(df)
    gc.collect()
    return df


# ----------------------------
# FEATURE LOADING
# ----------------------------
def load_recent_features(folder: str = "data/features", days_back: int = 30, max_files: int = 100) -> pd.DataFrame:
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
    combined = compute_features(combined)
    gc.collect()
    return combined
