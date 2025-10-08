import streamlit as st
import pandas as pd
from datetime import datetime

from src.fetch_latest_prices import fetch_latest_prices_dict
from models.recommend_sell import batch_recommend_sell

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BUY_FILE = "data/predictions/latest_top_flips.csv"
ACTIVE_FILE = "data/active_flips.csv"
REFRESH_MINUTES = 5

st.set_page_config(page_title="OSRS AI Flipping Dashboard", layout="wide")

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def load_data():
    try:
        buys = pd.read_csv(BUY_FILE)
        buys["selected"] = False
    except Exception:
        buys = pd.DataFrame()
    return buys

def load_active():
    try:
        df = pd.read_csv(ACTIVE_FILE)
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        return df
    except Exception:
        return pd.DataFrame(columns=["item_id", "entry_price", "entry_time"])

def save_active(df):
    df.to_csv(ACTIVE_FILE, index=False)

# ---------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------
st.title("💹 OSRS AI Flipping Dashboard")
st.caption("Track buys, holds, and sell recommendations in real time")

# Sidebar: Refresh
if st.button("🔄 Refresh Prices / Recompute"):
    st.session_state["latest_prices"] = fetch_latest_prices_dict()

if "latest_prices" not in st.session_state:
    st.session_state["latest_prices"] = fetch_latest_prices_dict()

# Load data
buy_recs = load_data()
active_flips = load_active()

# ---------------------------------------------------------------------
# SECTION 1 — BUY RECOMMENDATIONS
# ---------------------------------------------------------------------
st.subheader("🛒 Buy Recommendations")

if buy_recs.empty:
    st.warning("No buy recommendations found — run predict_flips first.")
else:
    st.write("Select flips you actually executed:")
    selected = st.multiselect(
        "Choose items you bought",
        options=buy_recs["item_id"].tolist(),
        format_func=lambda x: f"{x} - {buy_recs.loc[buy_recs['item_id'] == x, 'name'].values[0] if 'name' in buy_recs.columns else ''}",
    )

    if st.button("✅ Mark Selected as Implemented"):
        now = datetime.utcnow()
        new_entries = buy_recs[buy_recs["item_id"].isin(selected)].copy()
        new_entries["entry_price"] = new_entries["low"]
        new_entries["entry_time"] = now
        new_entries = new_entries[
            ["item_id", "name", "entry_price", "entry_time", "spread_ratio", "rsi", "momentum"]
        ]

        active_flips = pd.concat([active_flips, new_entries], ignore_index=True).drop_duplicates("item_id", keep="last")
        save_active(active_flips)
        st.success(f"Added {len(new_entries)} items to active tracking!")

    st.dataframe(
        buy_recs[["item_id", "name", "potential_profit", "spread_ratio", "liquidity_1h", "rsi"]].head(20),
        hide_index=True,
        use_container_width=True,
    )

# ---------------------------------------------------------------------
# SECTION 2 — ACTIVE FLIPS + SELL SIGNALS
# ---------------------------------------------------------------------
st.subheader("💰 Active Flips & Sell Recommendations")

if active_flips.empty:
    st.info("No active flips yet. Add items from above.")
else:
    latest_prices = st.session_state["latest_prices"]
    sell_recs = batch_recommend_sell(active_flips, latest_prices)

    # Merge sell urgency into active flips
    active_display = active_flips.merge(sell_recs, on="item_id", how="left")

    # Conditional formatting by urgency
    def highlight_row(row):
        urgency = row.get("urgency_score", 0)
        if urgency >= 0.8:
            return ["background-color: #ffcccc"] * len(row)
        elif urgency >= 0.6:
            return ["background-color: #fff2cc"] * len(row)
        return ["" for _ in row]

    st.dataframe(
        active_display.style.apply(highlight_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Option to remove sold flips
    sold_ids = st.multiselect("Mark flips as SOLD", active_flips["item_id"].tolist())
    if st.button("💸 Remove Sold Items"):
        active_flips = active_flips[~active_flips["item_id"].isin(sold_ids)]
        save_active(active_flips)
        st.success(f"Removed {len(sold_ids)} sold flips!")

st.caption("Auto-refreshes live prices every few minutes or on demand.")
