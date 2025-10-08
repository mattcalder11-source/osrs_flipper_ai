import pandas as pd
import numpy as np
from datetime import timedelta

def label_future_profit(df,
                        horizon_hours=1,
                        min_profit_gp=1000,
                        stack_cap_gp=125_000_000):
    """
    Label data for flip profitability with dynamic slippage and stack caps.

    Parameters
    ----------
    df : DataFrame
        Must include columns: ['item_id', 'ts_utc', 'high', 'low', 'liquidity_1h']
    horizon_hours : int
        How far ahead (in hours) to look for a potential sell.
    min_profit_gp : int
        Minimum profit per item (gp) for a flip to count as "profitable".
    stack_cap_gp : int
        Maximum gold allocated per flip simulation.
    """

    # Ensure timestamps are in datetime format
    df['ts_utc'] = pd.to_datetime(df['ts_utc'], unit='s', errors='coerce')

    out_rows = []
    grouped = df.groupby('item_id')

    for item_id, g in grouped:
        g = g.sort_values('ts_utc').copy()

        median_price = g['low'].median()
        if np.isnan(median_price) or median_price <= 0:
            continue

        # how many items we can buy under the stack cap
        max_qty = int(stack_cap_gp // median_price)
        if max_qty == 0:
            max_qty = 1

        for i, row in g.iterrows():
            t = row['ts_utc']
            liquidity = row.get('liquidity_1h', np.nan)

            # --- Dynamic slippage model ---
            # Lower liquidity = higher slippage
            if pd.isna(liquidity):
                buy_slip = 0.015
                sell_slip = 0.02
            elif liquidity > 20:
                buy_slip = 0.005
                sell_slip = 0.005
            elif liquidity > 10:
                buy_slip = 0.0075
                sell_slip = 0.01
            elif liquidity > 5:
                buy_slip = 0.01
                sell_slip = 0.015
            else:
                buy_slip = 0.015
                sell_slip = 0.025

            # Apply slippage to prices
            buy_price = row['low'] * (1 + buy_slip)
            end_time = t + timedelta(hours=horizon_hours)
            future = g[(g['ts_utc'] > t) & (g['ts_utc'] <= end_time)]

            if future.empty:
                continue

            sell_price = future['high'].max() * (1 - sell_slip)

            profit_per_item = sell_price - buy_price
            profit_total = profit_per_item * max_qty
            profit_pct = profit_per_item / buy_price if buy_price > 0 else 0
            profitable = int(profit_per_item >= min_profit_gp / max_qty)

            out_rows.append({
                'item_id': item_id,
                'ts_utc': t,
                'buy_slip': buy_slip,
                'sell_slip': sell_slip,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'profit_per_item': profit_per_item,
                'profit_total': profit_total,
                'profit_pct': profit_pct,
                'max_qty': max_qty,
                'profitable': profitable
            })

    return pd.DataFrame(out_rows)


if __name__ == "__main__":
    # Example usage
    df = pd.read_parquet("data/features.parquet")

    labels = label_future_profit(
        df,
        horizon_hours=2,
        min_profit_gp=500,
        stack_cap_gp=5_000_000
    )

    labels.to_parquet("data/labels_dynamic_slippage.parquet", index=False)
    print("✅ Labels generated:", len(labels))
    print(labels.head())
