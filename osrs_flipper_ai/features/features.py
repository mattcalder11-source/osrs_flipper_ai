from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import gc
import numpy as np
import pandas as pd
from typing import Optional

EPS = 1e-9


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure minimal required columns exist, standardize names and types."""
    df = df.copy()

    # item_id
    if "item_id" not in df.columns and "item_id" in df.index.names:
        df = df.reset_index()
    if "item_id" not in df.columns:
        raise KeyError("Missing required 'item_id' column.")

    # timestamp: accept common alt names and coerce to timezone-aware datetime
    if "timestamp" not in df.columns:
        for alt in ("ts_utc", "time", "datetime"):
            if alt in df.columns:
                df = df.rename(columns={alt: "timestamp"})
                break
        else:
            raise KeyError("No timestamp-like column found (expected 'timestamp' or 'ts_utc' or 'time').")

    # convert timestamp -> pandas datetime (UTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        # If some rows failed to parse, try naive parse without utc then localize
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str), errors="coerce")
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
    if df["timestamp"].isna().all():
        raise KeyError("All timestamps failed to parse after normalization.")

    # price columns
    if "high" not in df.columns:
        for alt in ("high_price", "avg_high_price"):
            if alt in df.columns:
                df = df.rename(columns={alt: "high"})
                break
        else:
            # last resort: create from avg_high_price if available later
            df["high"] = 0.0

    if "low" not in df.columns:
        for alt in ("low_price", "avg_low_price"):
            if alt in df.columns:
                df = df.rename(columns={alt: "low"})
                break
        else:
            df["low"] = 0.0

    # avg_high_price / avg_low_price: keep as-is if present, otherwise compute a short rolling mean per item later
    # volume fallback
    if "volume" not in df.columns:
        for alt in ("trade_volume", "buy_volume", "sell_volume", "total_volume", "qty"):
            if alt in df.columns:
                df = df.rename(columns={alt: "volume"})
                break
        else:
            # Fill with zeros but keep column so downstream code doesn't KeyError
            df["volume"] = 0.0
            print("⚠️ No 'volume' column found — filling with zeros.")

    # Ensure numeric dtypes where possible (reduce memory)
    for c in ("high", "low", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    # safe defaults for avg columns
    if "avg_high_price" not in df.columns:
        df["avg_high_price"] = np.nan
    if "avg_low_price" not in df.columns:
        df["avg_low_price"] = np.nan

    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, ROC, MACD, contrarian flags, and a composite technical_score.
    Operates per-item_id (grouped).
    Returns a dataframe with new indicator columns appended.
    """
    df = normalize_schema(df)
    # keep working copy and sort by time
    df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True).copy()

    def _for_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy().reset_index(drop=True)

        # price baseline: prefer avg_high_price if present (non-NaN), else high
        if g["avg_high_price"].notna().any():
            price = g["avg_high_price"].fillna(method="ffill").fillna(g["high"])
        else:
            price = g["high"].astype(float)

        # RSI (14-period EWM method)
        delta = price.diff()
        up = delta.clip(lower=0.0)
        down = (-delta.clip(upper=0.0)).fillna(0.0)
        roll_up = up.ewm(span=14, adjust=False).mean()
        roll_down = down.ewm(span=14, adjust=False).mean()
        rs = roll_up / (roll_down + EPS)
        g["rsi"] = (100 - (100 / (1 + rs))).fillna(50.0)

        # ROC (12 period)
        g["roc"] = (price / price.shift(12) - 1) * 100

        # MACD
        ema_short = price.ewm(span=12, adjust=False).mean()
        ema_long = price.ewm(span=26, adjust=False).mean()
        macd = (ema_short - ema_long).fillna(0.0)
        g["macd"] = macd
        g["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

        # Contrarian signal: overbought/oversold
        g["contrarian_flag"] = 0
        g.loc[g["rsi"] > 70, "contrarian_flag"] = -1
        g.loc[g["rsi"] < 30, "contrarian_flag"] = +1

        # Normalizations per-group (0-1)
        for col in ("rsi", "roc", "macd"):
            col_min = g[col].min(skipna=True)
            col_max = g[col].max(skipna=True)
            denom = (col_max - col_min) if (col_max - col_min) > EPS else EPS
            g[f"{col}_norm"] = (g[col] - col_min) / denom

        # liquidity_1h and volatility_1h must exist in group (computed earlier by compute_features)
        # defensively fill missing cols
        if "liquidity_1h" not in g.columns:
            g["liquidity_1h"] = 0.0
        if "volatility_1h" not in g.columns:
            g["volatility_1h"] = 0.0

        # Combine into composite technical score (0–5)
        liq_den = g["liquidity_1h"].max() + EPS
        vol_den = g["volatility_1h"].max() + EPS
        g["technical_score"] = (
            2.0 * g["rsi_norm"]
            + 1.0 * g["roc_norm"]
            + 1.0 * g["macd_norm"]
            + 0.5 * (g["liquidity_1h"] / liq_den)
            + 0.5 * (1.0 - (g["volatility_1h"] / vol_den))
        )

        g["technical_score"] = (g["technical_score"] + g["contrarian_flag"] * 0.5).clip(lower=0.0, upper=5.0)

        # downcast numeric columns to float32 to save memory
        float_cols = [c for c in g.columns if g[c].dtype.kind == "f"]
        for c in float_cols:
            g[c] = g[c].astype(np.float32)

        return g

    # apply per-item; reset_index(drop=True) to avoid item_id duplication in both index & column
    out = df.groupby("item_id", group_keys=False).apply(_for_group).reset_index(drop=True)
    gc.collect()
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Top-level feature computation:
    - normalize schema
    - compute mid_price, spread, spread_ratio
    - compute per-item volatility (1h window ~ 12 samples) and liquidity (rolling mean)
    - compute avg_high_price/avg_low_price rolling 6 periods PER ITEM
    - then compute technical indicators
    """
    df = normalize_schema(df)
    df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    # mid & spread
    df["mid_price"] = ((df["high"].astype(float) + df["low"].astype(float)) / 2.0).astype(np.float32)
    df["spread"] = (df["high"].astype(float) - df["low"].astype(float)).astype(np.float32)

    # per-item avg_high_price / avg_low_price (rolling window = 6)
    df["avg_high_price"] = (
        df.groupby("item_id")["high"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    ).astype(np.float32)
    df["avg_low_price"] = (
        df.groupby("item_id")["low"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    ).astype(np.float32)

    # volatility (1h ~ 12 samples) using pct_change(fill_method=None) then rolling std
    df["volatility_1h"] = (
        df.groupby("item_id")["mid_price"]
        .transform(lambda x: x.pct_change(fill_method=None).rolling(12, min_periods=3).std())
    ).fillna(0.0).astype(np.float32)

    # liquidity: rolling mean of volume per item
    df["liquidity_1h"] = (
        df.groupby("item_id")["volume"]
        .transform(lambda x: x.rolling(12, min_periods=1).mean())
    ).fillna(0.0).astype(np.float32)

    # spread ratio
    df["spread_ratio"] = (df["spread"] / (df["mid_price"].abs() + EPS)).astype(np.float32)

    # Technical indicators appended
    df = compute_technical_indicators(df)

    # final tidy: ensure timestamp dtype and sensible column order
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # reduce numeric memory use where sensible
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)

    return df


def load_recent_features(folder: str = "data/features", days_back: int = 30, max_files: Optional[int] = 200) -> pd.DataFrame:
    """
    Load recent feature snapshots (parquet files) from `folder`.
    - only files <= days_back are kept
    - limits to max_files (most recent)
    - returns a combined dataframe (normalized), ready for training/prediction
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Feature folder not found: {folder}")

    files = sorted(folder.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No feature snapshots found in {folder}.")

    now = datetime.now(timezone.utc)
    selected = []
    for p in files:
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        age_days = (now - mtime).total_seconds() / 86400.0
        if age_days <= days_back:
            selected.append((p, mtime))
        else:
            # files are sorted newest-first; once we find older ones we can stop
            break
        if max_files and len(selected) >= max_files:
            break

    if not selected:
        raise ValueError(f"No snapshot files within last {days_back} days in {folder}.")

    frames = []
    for p, file_time in selected:
        try:
            df = pd.read_parquet(p)
            if df.empty:
                continue
            df = normalize_schema(df)
            df["file_time"] = file_time
            df["source_file"] = p.name
            # keep only necessary columns to reduce memory
            keep_cols = [
                "item_id",
                "timestamp",
                "high",
                "low",
                "volume",
                "avg_high_price",
                "avg_low_price",
                "file_time",
                "source_file",
            ]
            existing = [c for c in keep_cols if c in df.columns]
            df = df[existing].copy()
            # downcast
            df["item_id"] = df["item_id"].astype("category")
            for col in ("high", "low", "volume", "avg_high_price", "avg_low_price"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)
            frames.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {p}: {e}")

    if not frames:
        raise ValueError("No valid feature snapshots could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    # restore item_id to normal string/object if needed
    combined["item_id"] = combined["item_id"].astype(str)

    # deduplicate by (item_id, timestamp) keep most recent file_time
    combined = combined.sort_values(["item_id", "timestamp", "file_time"], ascending=[True, True, False])
    combined = combined.drop_duplicates(subset=["item_id", "timestamp"], keep="first").reset_index(drop=True)

    # compute the derived features (volatility, liquidity, technicals)
    combined = compute_features(combined)

    # final memory optimizations
    for c in combined.select_dtypes(include=["float64"]).columns:
        combined[c] = combined[c].astype(np.float32)

    gc.collect()
    return combined