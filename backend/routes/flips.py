from fastapi import APIRouter
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from osrs_flipper_ai.models.recommend_sell import batch_recommend_sell
from osrs_flipper_ai.src.fetch_latest_prices import fetch_latest_prices_dict

router = APIRouter(prefix="/flips", tags=["flips"])

# ----------------------------------------------------
# Path discovery
# ----------------------------------------------------
def get_data_dir():
    """Find the correct data directory (handles both repo layouts)."""
    base_dir = Path(__file__).resolve().parents[2]
    candidates = [
        base_dir / "data",
        base_dir / "osrs_flipper_ai" / "data",
    ]
    for d in candidates:
        if (d / "predictions").exists():
            print(f"📂 Using data directory: {d}")
            return d
    raise FileNotFoundError("No valid data directory found.")

DATA_DIR = get_data_dir()

# ----------------------------------------------------
# CSV helpers
# ----------------------------------------------------
def load_latest_predictions():
    """Load the main predictions file directly."""
    latest = Path("/root/osrs_flipper_ai/data/predictions/latest_top_flips.csv")
    print(f"✅ Loading predictions from: {latest}")

    if not latest.exists():
        print(f"⚠️ File not found: {latest}")
        return pd.DataFrame()

    df = pd.read_csv(latest)

    rename_map = {
        "avg_low_price": "low",
        "avg_high_price": "high",
        "predicted_profit": "potential_profit",
        "predicted_profit_pct": "potential_profit",
        "item_name": "name",
    }
    df = df.rename(columns=rename_map)

    # Replace invalid values
    import numpy as np
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Ensure all required columns exist
    for col in ["low", "high", "potential_profit", "name"]:
        if col not in df.columns:
            df[col] = 0

    return df


def load_csv(path, cols=None):
    if not path.exists():
        return pd.DataFrame(columns=cols or [])
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")
        return pd.DataFrame(columns=cols or [])


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ----------------------------------------------------
# API routes
# ----------------------------------------------------
@router.get("/buy-recommendations")
def get_buy_recommendations():
    """Return latest model-predicted flips."""
    df = load_latest_predictions()
    return df.to_dict(orient="records")


@router.get("/active")
def get_active_flips():
    active_file = DATA_DIR / "active_flips.csv"
    return load_csv(active_file).to_dict(orient="records")


@router.post("/add/{item_id}")
def add_active_flip(item_id: int):
    pred_file = DATA_DIR / "predictions" / "latest_top_flips.csv"
    active_file = DATA_DIR / "active_flips.csv"
    buys = load_csv(pred_file)
    active = load_csv(active_file)

    row = buys[buys["item_id"] == item_id]
    if row.empty:
        return {"error": "Item not found"}

    row = row.copy()
    row["entry_price"] = row["low"]
    row["entry_time"] = datetime.utcnow()
    active = pd.concat([active, row], ignore_index=True).drop_duplicates("item_id", keep="last")
    save_csv(active, active_file)
    return {"status": "added", "item_id": item_id}


@router.delete("/remove/{item_id}")
def remove_active_flip(item_id: int):
    active_file = DATA_DIR / "active_flips.csv"
    active = load_csv(active_file)
    active = active[active["item_id"] != item_id]
    save_csv(active, active_file)
    return {"status": "removed", "item_id": item_id}


@router.get("/sell-signals")
def get_sell_signals():
    active_file = DATA_DIR / "active_flips.csv"
    active = load_csv(active_file)
    if active.empty:
        return []

    latest_prices = fetch_latest_prices_dict()
    sell_recs = batch_recommend_sell(active, latest_prices)
    return sell_recs.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records")
