import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from tqdm import tqdm

# ---------------------------------------------------------------
# Feature Engineering Utilities for OSRS Flipping AI
# ---------------------------------------------------------------

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all required columns exist with standardized names and safe defaults.
    This prevents KeyErrors across different data versions.
    """

    df = df.copy()

    # --- item_id ---
    if 'item_id' not in df.columns and 'item_id' in df.index.names:
        df = df.reset_index()
    if 'item_id' not in df.columns:
        raise KeyError("Missing required 'item_id' column.")

    # --- timestamp ---
    if 'timestamp' not in df.columns:
        for alt in ['ts_utc', 'time', 'datetime']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'timestamp'})
                break
        else:
            raise KeyError("No timestamp-like column found (expected 'timestamp' or 'ts_utc').")

    # --- high / low price ---
    if 'high' not in df.columns:
        for alt in ['high_price', 'avg_high_price']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'high'})
                break
        else:
            print("⚠️ Missing 'high' column — filling with zeros.")
            df['high'] = 0.0

    if 'low' not in df.columns:
        for alt in ['low_price', 'avg_low_price']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'low'})
                break
        else:
            print("⚠️ Missing 'low' column — filling with zeros.")
            df['low'] = 0.0

    # --- avg_high_price / avg_low_price ---
    if 'avg_high_price' not in df.columns:
        if 'high' in df.columns:
            df['avg_high_price'] = df['high'].rolling(6, min_periods=1).mean()
        else:
            df['avg_high_price'] = 0.0

    if 'avg_low_price' not in df.columns:
        if 'low' in df.columns:
            df['avg_low_price'] = df['low'].rolling(6, min_periods=1).mean()
        else:
            df['avg_low_price'] = 0.0

    # --- volume ---
    if 'volume' not in df.columns:
        for alt in ['trade_volume', 'buy_volume', 'sell_volume', 'total_volume', 'qty']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'volume'})
                break
        else:
            print("⚠️ No 'volume' column found — filling with zeros.")
            df['volume'] = 0.0

    # Fill NaNs with 0 for numeric stability
    df = df.fillna(0)

    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, ROC, MACD, contrarian flags, and composite technical_score.
    Safe for all pandas versions and avoids hangs / warnings.
    """

    df = normalize_schema(df)
    df = df.reset_index(drop=True).copy()

    def compute_indicators(group: pd.DataFrame) -> pd.DataFrame:
        # Ensure we're only processing the item’s data, not the key
        group = group.drop(columns=["item_id"], errors="ignore")

        price = group["avg_high_price"].astype(float)
        delta = price.diff()

        # RSI
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.ewm(span=14, adjust=False).mean()
        roll_down = down.ewm(span=14, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-9)
        group["rsi"] = 100 - (100 / (1 + rs))

        # ROC
        group["roc"] = (price / price.shift(12) - 1) * 100

        # MACD
        ema_short = price.ewm(span=12, adjust=False).mean()
        ema_long = price.ewm(span=26, adjust=False).mean()
        group["macd"] = ema_short - ema_long
        group["macd_signal"] = group["macd"].ewm(span=9, adjust=False).mean()

        # Contrarian signal
        group["contrarian_flag"] = 0
        group.loc[group["rsi"] > 70, "contrarian_flag"] = -1
        group.loc[group["rsi"] < 30, "contrarian_flag"] = +1

        # Normalize indicators safely
        for col in ["rsi", "roc", "macd"]:
            denom = group[col].max() - group[col].min() + 1e-9
            group[f"{col}_norm"] = (group[col] - group[col].min()) / denom

        # Technical score calculation (avoid division by zero)
        liq_max = group["liquidity_1h"].max() + 1e-9
        vol_max = group["volatility_1h"].max() + 1e-9

        group["technical_score"] = (
            2.0 * group["rsi_norm"]
            + 1.0 * group["roc_norm"]
            + 1.0 * group["macd_norm"]
            + 0.5 * (group["liquidity_1h"] / liq_max)
            + 0.5 * (1 - group["volatility_1h"] / vol_max)
        )

        # Apply contrarian adjustment
        group["technical_score"] += group["contrarian_flag"] * 0.5

        # ✅ Prevent hang: ensure numeric dtype, fill NaN, clip safely
        group["technical_score"] = (
            pd.to_numeric(group["technical_score"], errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=5)
        )

        return group

    # Apply group-by safely without deprecated params or recursion
    tqdm.pandas(desc="🧮 Computing indicators")

    df = (
        df.groupby("item_id", group_keys=False, sort=False)
        .progress_apply(lambda g: compute_indicators(g.copy()))
        .reset_index(drop=True)
    )

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all base and technical features from OSRS wiki data.
    Auto-detects and normalizes missing schema components.
    """

    df = normalize_schema(df)
    df = df.sort_values(['item_id', 'timestamp'])

    # --- Derived base features ---
    df['mid_price'] = (df['high'] + df['low']) / 2
    df['spread'] = df['high'] - df['low']

    # Volatility
    df['volatility_1h'] = (
        df.groupby('item_id')['mid_price']
          .transform(lambda x: x.pct_change(fill_method=None)
          .rolling(12, min_periods=3).std())
    )

    # Liquidity
    df['liquidity_1h'] = (
        df.groupby('item_id')['volume']
          .transform(lambda x: x.rolling(12, min_periods=3).mean())
    )

    # Spread ratio
    df['spread_ratio'] = df['spread'] / (df['mid_price'] + 1e-9)

    # Technical indicators
    df = compute_technical_indicators(df)

    return df


def load_recent_features(
    folder: str = "data/features",
    days_back: int = 30,
    decay_rate: float = 0.5
) -> pd.DataFrame:
    """
    Load all recent feature parquet files and combine with metadata.
    """

    files = sorted(glob.glob(os.path.join(folder, "features_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No feature files found in {folder}")

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue
            file_time = datetime.fromtimestamp(os.path.getmtime(f), tz=timezone.utc)
            df["file_time"] = file_time
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {f}: {e}")

    if not dfs:
        raise ValueError("No valid feature files loaded.")

    combined = pd.concat(dfs, ignore_index=True)
    combined = normalize_schema(combined)

    if "file_time" not in combined.columns:
        combined["file_time"] = datetime.utcnow()

    now = datetime.now(timezone.utc)
    combined["age_days"] = (now - combined["file_time"]).dt.total_seconds() / 86400
    combined["weight"] = np.exp(-decay_rate * combined["age_days"])

    combined = combined.sort_values("file_time", ascending=False)
    combined = combined.drop_duplicates(subset=["item_id", "timestamp"], keep="first")

    return combined


if __name__ == "__main__":
    df = load_recent_features()
    print(f"✅ Loaded {len(df)} rows across {df['item_id'].nunique()} items")
