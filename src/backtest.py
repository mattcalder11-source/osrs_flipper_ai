# src/backtest.py (simplified)
import pandas as pd
import heapq

def simulate_signals(signal_df, wallet=50_000_000, max_concurrent=5, slippage_pct=0.02):
    # signal_df must have columns: ts_utc, item_id, buy_price, target_sell_price (or predicted profit)
    wallet = float(wallet)
    open_positions = []  # min-heap by expected exit time or priority
    history = []
    for idx, row in signal_df.sort_values('ts_utc').iterrows():
        # Clean up positions whose horizon expired (sell at last known high)
        # For simplicity: just track buys and sells by label
        # Implementation depends on how you computed sells; use your labeling logic to simulate
        pass
    # compute returns, win rate, drawdown...
