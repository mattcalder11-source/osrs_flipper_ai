from fastapi import APIRouter
from pathlib import Path
import pandas as pd
from datetime import datetime
from glob import glob
import numpy as np

# Import your core model logic
from osrs_flipper_ai.models.recommend_sell import batch_recommend_sell
from osrs_flipper_ai.src.fetch_latest_prices import fetch_latest_prices_dict

router = APIRouter(prefix="/flips", tags=["flips"])

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def get_latest_prediction_file() -> Path:
    """Find the most recent predictions CSV in /data/predictions/."""
    pred_dir = DATA_DIR / "predictions"
    if not pred_dir.exists():
        return None
    files = sorted(
        pred_dir.glob("*.csv"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None

def load_latest_predictions():
    latest_file = get_latest_prediction_file()
    if latest_file is None:
        print("⚠️ No prediction files found in /data/predictions/")
        return pd.DataFrame()
    print(f"📂 Loading predictions from: {latest_file.name}")
    return load_csv(latest_file)

ACTIVE_FILE = DATA_DIR / "active_flips.csv"
PRED_FILE = get_latest_prediction_file()

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
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
# API Routes
# ----------------------------------------------------

@router.get("/buy-recommendations")
def get_buy_recommendations():
    """Return the latest model-predicted buy flips."""
    df = load_latest_predictions()
    return df.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records")

@router.get("/active")
def get_active_flips():
    df = load_csv(ACTIVE_FILE)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient="records")

@router.post("/add/{item_id}")
def add_active_flip(item_id: int):
    """Mark a recommended flip as implemented (bought)."""
    buys = load_csv(PRED_FILE)
    active = load_csv(ACTIVE_FILE)
    row = buys[buys["item_id"] == item_id]
    if row.empty:
        return {"error": "Item not found"}

    row = row.copy()
    row["entry_price"] = row["low"]
    row["entry_time"] = datetime.utcnow()
    active = pd.concat([active, row], ignore_index=True).drop_duplicates("item_id", keep="last")
    save_csv(active, ACTIVE_FILE)
    return {"status": "added", "item_id": item_id}

@router.delete("/remove/{item_id}")
def remove_active_flip(item_id: int):
    """Mark an active flip as sold."""
    active = load_csv(ACTIVE_FILE)
    active = active[active["item_id"] != item_id]
    save_csv(active, ACTIVE_FILE)
    return {"status": "removed", "item_id": item_id}

@router.get("/sell-signals")
def get_sell_signals():
    """Return sell recommendations for active flips."""
    active = load_csv(ACTIVE_FILE)
    if active.empty:
        return []
    latest_prices = fetch_latest_prices_dict()
    sell_recs = batch_recommend_sell(active, latest_prices)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return sell_recs.to_dict(orient="records")
