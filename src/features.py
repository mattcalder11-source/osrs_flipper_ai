# src/features.py
import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engineered features for each item:
    - Spread, mid_price, momentum, margin, volatility
    - Liquidity proxy (liquidity_1h): count of price updates in the last hour
    - Dynamic slippage based on liquidity
    """

    # Ensure expected columns exist (so code doesn't KeyError)
    for col in ['avg_5m_high', 'avg_5m_low', 'avg_1h_high', 'avg_1h_low', 'high', 'low', 'ts_utc', 'item_id', 'name']:
        if col not in df.columns:
            df[col] = np.nan

    # Sort and prepare timestamp column
    df = df.sort_values(['item_id', 'ts_utc']).copy()
    df['timestamp'] = pd.to_datetime(df['ts_utc'], unit='s', errors='coerce')
    df = df.dropna(subset=['timestamp'])  # drop rows with bad timestamps

    # Basic price features
    df['spread'] = df['high'] - df['low']
    df['mid_price'] = (df['high'] + df['low']) / 2
    # avoid division by zero
    df['spread_ratio'] = np.where(df['mid_price'] != 0, df['spread'] / df['mid_price'], 0.0)
    # momentum and potential profit (guard against NaN)
    df['momentum'] = (df['avg_5m_low'] - df['avg_1h_low']) / df['avg_1h_low'].replace({0: np.nan})
    df['potential_profit'] = df['avg_5m_high'] - df['avg_5m_low']
    df['margin_pct'] = np.where(df['avg_5m_low'] != 0, df['potential_profit'] / df['avg_5m_low'], 0.0)
    df['hour'] = df['timestamp'].dt.hour

    # Liquidity: count of price changes within a rolling 1-hour window for each item
    def calc_liquidity(group):
        # group may or may not contain the grouping column in future pandas versions;
        # get group key via group.name, and ensure item_id column exists
        group = group.sort_values('timestamp').copy()
        group_item_id = group.name
        if 'item_id' not in group.columns:
            group['item_id'] = group_item_id

        # set timestamp as index for time-aware rolling
        group = group.set_index('timestamp')

        # price change indicator (1 if low changed compared to prev snapshot)
        group['price_change'] = (group['low'] != group['low'].shift(1)).astype(int)

        # rolling 1-hour sum of price changes => liquidity proxy
        # this requires a DatetimeIndex
        group['liquidity_1h'] = group['price_change'].rolling('1h').sum().fillna(0)

        # reset index to restore timestamp column
        group = group.reset_index()
        return group

    # Use groupby.apply WITHOUT include_groups (works on older pandas).
    # calc_liquidity is written to be robust whether item_id is present or not.
    df = df.groupby('item_id', group_keys=False).apply(calc_liquidity)

    # Volatility: 1h rolling std of low prices (approx 12 * 5-min samples)
    df['volatility_1h'] = (
        df.groupby('item_id')['low']
        .transform(lambda x: x.rolling(12, min_periods=3).std())
        .fillna(0)
    )

    # Dynamic slippage: map liquidity -> slippage factor (higher liquidity => lower slippage)
    # This is a simple monotonic transform; tune coefficients to taste.
    # We clip to keep values sensible.
    df['slippage_factor'] = np.clip(1 - (1 / (1 + df['liquidity_1h'])) * 0.03, 0.90, 1.00)

    # Adjusted profit and margin
    df['adj_profit'] = df['potential_profit'] * df['slippage_factor']
    df['adj_margin_pct'] = np.where(df['avg_5m_low'] != 0, df['adj_profit'] / df['avg_5m_low'], 0.0)

    # Replace any remaining inf/-inf and fill NaNs with 0 for downstream compatibility
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Add technical indicators
    df = compute_technical_indicators(df)

    # Columns to keep for model/analysis
    cols_to_keep = [
        'ts_utc', 'item_id', 'name', 'high', 'low',
        'spread', 'mid_price', 'spread_ratio', 'momentum',
        'potential_profit', 'margin_pct', 'hour',
        'liquidity_1h', 'volatility_1h',
        'slippage_factor', 'adj_profit', 'adj_margin_pct'
    ]

    # Some columns may not exist if filled earlier, so intersect with df.columns
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    return df[cols_to_keep].reset_index(drop=True)

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators and assigns a composite technical score (0–5).
    """

    df = df.sort_values(['item_id', 'ts_utc']).copy()

    def compute_indicators(group):
        # Compute RSI
        delta = group['mid_price'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        roll_up = pd.Series(gain).rolling(14, min_periods=3).mean()
        roll_down = pd.Series(loss).rolling(14, min_periods=3).mean()

        rs = roll_up / (roll_down + 1e-9)
        group['rsi'] = 100 - (100 / (1 + rs))

        # Rate of Change (ROC)
        group['roc'] = group['mid_price'].pct_change(periods=6) * 100  # 6 * 5min ≈ 30min lookback

        # MACD
        ema_short = group['mid_price'].ewm(span=12, adjust=False).mean()
        ema_long = group['mid_price'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema_short - ema_long
        group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()

        # Contrarian signal: detect extremes
        group['contrarian_flag'] = 0
        group.loc[group['rsi'] > 70, 'contrarian_flag'] = -1  # overbought
        group.loc[group['rsi'] < 30, 'contrarian_flag'] = +1  # oversold

        # Normalize indicators to 0–1
        for col in ['rsi', 'roc', 'macd']:
            group[f'{col}_norm'] = (group[col] - group[col].min()) / (group[col].max() - group[col].min() + 1e-9)

        # Combine indicators into 0–5 technical score
        group['technical_score'] = (
            2.0 * group['rsi_norm']
            + 1.0 * group['roc_norm']
            + 1.0 * group['macd_norm']
            + 0.5 * (group['liquidity_1h'] / (group['liquidity_1h'].max() + 1e-9))
            + 0.5 * (1 - group['volatility_1h'] / (group['volatility_1h'].max() + 1e-9))
        )

        # Adjust by contrarian signal
        group['technical_score'] += group['contrarian_flag'] * 0.5
        group['technical_score'] = group['technical_score'].clip(0, 5)

        return group

    df = df.groupby('item_id', group_keys=False).apply(compute_indicators)
    return df