import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from recommend_sell import batch_recommend_sell

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = Path("/root/osrs_flipper_ai/osrs_flipper_ai/data")
MODEL_DIR = Path("/root/osrs_flipper_ai/osrs_flipper_ai/models/trained_models")
BLACKLIST_PATH = DATA_DIR / "item_blacklist.txt"
MAPPING_PATH = DATA_DIR / "item_mapping.json"
LATEST_PRICES_PATH = DATA_DIR / "raw/latest_prices.json"
PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------
def load_latest_model(model_dir=MODEL_DIR):
    model_files = sorted(model_dir.glob("*.pkl"), key=os.path.getmtime, reverse=True)
    if not model_files:
        raise FileNotFoundError("No trained model found.")
    latest_model = model_files[0]
    model = joblib.load(latest_model)
    print(
        f"📦 Loaded model: {latest_model}\n"
        f"   Trained {model.get('metadata', {}).get('timestamp', 'Unknown')} "
        f"(R²={model.get('metadata', {}).get('r2', 'N/A')})"
    )
    return model

# ---------------------------------------------------------------------
# LOAD BLACKLIST
# ---------------------------------------------------------------------
def load_blacklist():
    if not BLACKLIST_PATH.exists():
        print("⚠️ No blacklist.txt found, skipping blacklist filtering.")
        return []

    with open(BLACKLIST_PATH, "r", encoding="utf-8") as f:
        lines = [x.strip().lower() for x in f.readlines() if x.strip()]
    print(f"🧱 Loaded blacklist ({len(lines)} items): {lines}")
    return lines

# ---------------------------------------------------------------------
# LOAD ITEM MAPPING
# ---------------------------------------------------------------------
def load_item_mapping():
    if not MAPPING_PATH.exists():
        print("⚠️ No item_mapping.json found — proceeding without enrichment.")
        return pd.DataFrame()

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list-of-dicts and dict formats
    if isinstance(data, dict):
        data = data.get("items", list(data.values())) if "items" in data else list(data.values())

    df = pd.DataFrame(data)
    if "name" not in df.columns:
        print("⚠️ 'name' not found in item_mapping.json — skipping name-based blacklist.")
    else:
        df["name"] = df["name"].astype(str).str.lower()

    return df

# ---------------------------------------------------------------------
# MAIN PREDICT FUNCTION
# ---------------------------------------------------------------------
def predict_flips(model_dict, df, top_n=100):
    model = model_dict["model"]
    mode = model_dict.get("mode", "ratio")

    # ------------------------------------------------------------
    # 🧹 Clean data
    # ------------------------------------------------------------
    df = df.dropna()
    if "item_id" not in df.columns:
        raise KeyError("Missing 'item_id' column in features DataFrame.")

    # Remove low-quality or inactive items
    df = df[df["mid_price"] > 50].dropna(subset=["daily_volume", "volatility_1h"])
    df = df[df["daily_volume"] > 0]

    # ------------------------------------------------------------
    # 🧱 Blacklist filtering
    # ------------------------------------------------------------
    blacklist = load_blacklist()
    mapping_df = load_item_mapping()
    if not mapping_df.empty and "name" in mapping_df.columns and len(blacklist) > 0:
        name_to_id = (
            mapping_df[mapping_df["name"].isin(blacklist)]["id"].astype(int).tolist()
            if "id" in mapping_df.columns
            else []
        )
        before = len(df)
        df = df[~df["item_id"].isin(name_to_id)]
        after = len(df)
        print(f"🚫 Blacklist filter: {before} → {after} rows (filtered {before - after})")
    else:
        print("⚠️ No name-based filtering applied (missing 'name' column or empty blacklist).")

    # ------------------------------------------------------------
    # 📈 Predict
    # ------------------------------------------------------------
    feature_cols = [
        c
        for c in df.columns
        if c not in ["item_id", "timestamp", "pred", "buy_price", "sell_price"]
    ]
    preds = model.predict(df[feature_cols])
    df["pred"] = preds * 1.1  # calibration bias (+10%)

    print(f"📊 Pred summary:\n{df['pred'].describe()}")

    # ------------------------------------------------------------
    # 💰 Interpret predictions into buy/sell
    # ------------------------------------------------------------
    if mode == "ratio":
        df["sell_price"] = df["mid_price"] * (1 + df["pred"])
    else:
        df["sell_price"] = df["mid_price"] + df["pred"]

    df["buy_price"] = df["mid_price"]
    df["predicted_profit_gp"] = df["sell_price"] - df["buy_price"]
    df["profit_pct"] = 100 * df["predicted_profit_gp"] / df["buy_price"]

    # Clamp outliers
    df["sell_price"] = df["sell_price"].clip(
        lower=df["buy_price"] * 0.95, upper=df["buy_price"] * 3.0
    )
    df["predicted_profit_gp"] = df["sell_price"] - df["buy_price"]
    df["profit_pct"] = 100 * df["predicted_profit_gp"] / df["buy_price"]

    # ------------------------------------------------------------
    # 🎯 Group by item for diversity
    # ------------------------------------------------------------
    df = df.sort_values("predicted_profit_gp", ascending=False)
    df = df.groupby("item_id", as_index=False).first()
    raw_top_flips = df.head(top_n).copy()

    # ------------------------------------------------------------
    # 🗺️ Load latest prices for enrichment
    # ------------------------------------------------------------
    latest_prices_dict = {}
    if LATEST_PRICES_PATH.exists():
        with open(LATEST_PRICES_PATH, "r") as f:
            latest_prices_dict = json.load(f)
        print(f"✅ Loaded {len(latest_prices_dict)} current prices from Wiki.")
    else:
        print("⚠️ No latest_prices.json found; skipping live enrichment.")

    # ------------------------------------------------------------
    # 🧠 Batch recommend sells
    # ------------------------------------------------------------
    top_flips = batch_recommend_sell(raw_top_flips, latest_prices_dict)

    # ------------------------------------------------------------
    # 💾 Save predictions
    # ------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = PREDICTIONS_DIR / f"top_flips_{timestamp}.csv"
    latest_path = PREDICTIONS_DIR / "latest_top_flips.csv"
    top_flips.to_csv(out_path, index=False)
    top_flips.to_csv(latest_path, index=False)
    print(f"💾 Saved {len(top_flips)} predictions → {out_path}")
    return top_flips

# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting flip prediction pipeline...")
    model_dict = load_latest_model()
    features_path = DATA_DIR / "features/features_latest.parquet"
    df = pd.read_parquet(features_path)
    top_flips = predict_flips(model_dict, df, top_n=100)
    print("✅ Prediction complete.")
