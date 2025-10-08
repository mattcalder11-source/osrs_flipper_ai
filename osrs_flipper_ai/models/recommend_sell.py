"""
recommend_sell.py — OSRS AI flipping sell signal module
--------------------------------------------------------
Enhanced with confidence-weighted "urgency" ranking system
so dashboards can prioritize which flips to sell first.
"""

import numpy as np
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
SELL_RULES = {
    "min_profit_pct": 0.005,      # Sell if ≥ 0.5% profit
    "target_profit_pct": 0.015,   # Ideal profit target (1.5%)
    "stop_loss_pct": -0.01,       # Cut losses if −1%
    "spread_compression": 0.5,    # Sell if spread < 50% of entry spread
    "max_hold_hours": 12,         # Timeout-based sell signal
    "rsi_overbought": 65,         # RSI threshold for momentum reversal
}


# ---------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------
def recommend_sell(entry_price, current_price, features, entry_time=None, now=None):
    """
    Decide whether to SELL based on current conditions.

    Returns
    -------
    dict:
        {
          "should_sell": bool,
          "reason": str,
          "confidence": float,
          "profit_pct": float,
          "hold_hours": float,
          "urgency_score": float
        }
    """
    if now is None:
        now = datetime.utcnow()

    spread_ratio = features.get("spread_ratio", 0)
    momentum = features.get("momentum", 0)
    rsi = features.get("rsi", 50)

    profit_pct = (current_price - entry_price) / entry_price
    hold_hours = None
    if entry_time:
        hold_hours = (now - entry_time).total_seconds() / 3600.0

    reason, confidence, urgency = None, 0.0, 0.0
    sell = False

    # --- Primary SELL logic ---
    if profit_pct >= SELL_RULES["target_profit_pct"]:
        sell, reason, confidence = True, "target_hit", 0.95

    elif profit_pct <= SELL_RULES["stop_loss_pct"]:
        sell, reason, confidence = True, "stop_loss", 0.9

    elif spread_ratio < SELL_RULES["spread_compression"] * features.get("entry_spread_ratio", spread_ratio):
        sell, reason, confidence = True, "spread_compressed", 0.8

    elif rsi >= SELL_RULES["rsi_overbought"] or momentum < 0:
        sell, reason, confidence = True, "momentum_reversal", 0.7

    elif hold_hours and hold_hours > SELL_RULES["max_hold_hours"]:
        sell, reason, confidence = True, "timeout_exit", 0.6

    elif profit_pct >= SELL_RULES["min_profit_pct"]:
        sell, reason, confidence = True, "min_profit_exit", 0.5

    # --- Urgency Scoring ---
    # Combines profit distance, time decay, and momentum pressure into a 0–1 scale
    urgency = compute_urgency_score(
        profit_pct=profit_pct,
        confidence=confidence,
        momentum=momentum,
        rsi=rsi,
        hold_hours=hold_hours,
    )

    return {
        "should_sell": sell,
        "reason": reason,
        "confidence": confidence,
        "profit_pct": round(profit_pct, 4),
        "hold_hours": None if hold_hours is None else round(hold_hours, 2),
        "urgency_score": round(urgency, 3),
    }


# ---------------------------------------------------------------------
# URGENCY SCORING FUNCTION
# ---------------------------------------------------------------------
def compute_urgency_score(profit_pct, confidence, momentum, rsi, hold_hours):
    """
    Combine multiple dimensions into a single [0,1] urgency score.
    High = sell sooner.
    """
    # Normalize inputs
    profit_score = np.clip(profit_pct / SELL_RULES["target_profit_pct"], 0, 1)
    loss_score = np.clip(-profit_pct / abs(SELL_RULES["stop_loss_pct"]), 0, 1)
    momentum_risk = np.clip(-momentum, 0, 1)   # negative momentum = risk
    rsi_risk = np.clip((rsi - SELL_RULES["rsi_overbought"]) / 15, 0, 1)
    time_risk = np.clip((hold_hours or 0) / SELL_RULES["max_hold_hours"], 0, 1)

    # Weighted combination (sum ≈ 1.0)
    urgency = (
        0.30 * confidence +        # base signal confidence
        0.15 * profit_score +      # profit nearing target
        0.25 * loss_score +        # deepening losses → urgent
        0.15 * momentum_risk +     # momentum turning negative
        0.10 * rsi_risk +          # overbought exhaustion
        0.05 * time_risk           # stale flip
    )

    return float(np.clip(urgency, 0, 1))

# ---------------------------------------------------------------------
# BATCH INTERFACE
# ---------------------------------------------------------------------
def batch_recommend_sell(active_flips_df, latest_prices_dict):
    """
    Given active flips + live prices, produce SELL recommendations.

    active_flips_df: DataFrame with columns:
        ['item_id', 'entry_price', 'entry_time', 'spread_ratio', 'rsi', 'momentum']
    latest_prices_dict: dict mapping item_id → current mid/high price
    """
    recs = []

    for _, row in active_flips_df.iterrows():
        item_id = row["item_id"]
        current_price = latest_prices_dict.get(item_id, np.nan)
        if np.isnan(current_price):
            continue

        features = {
            "spread_ratio": row.get("spread_ratio", 0),
            "rsi": row.get("rsi", 50),
            "momentum": row.get("momentum", 0),
            "entry_spread_ratio": row.get("entry_spread_ratio", row.get("spread_ratio", 0)),
        }

        entry_time = None
        if "entry_time" in row and pd.notna(row["entry_time"]):
            entry_time = pd.to_datetime(row["entry_time"], errors="coerce")

        result = recommend_sell(
            entry_price=row["entry_price"],
            current_price=current_price,
            features=features,
            entry_time=entry_time,
        )

        recs.append({**{"item_id": item_id, "current_price": current_price}, **result})

    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.sort_values("urgency_score", ascending=False).reset_index(drop=True)

    return df
