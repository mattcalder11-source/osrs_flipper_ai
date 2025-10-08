import os
import time
import pandas as pd
from datetime import datetime, timezone
from osrs_flipper_ai.config import DATA_DIR
from data_ingest.ingest import snapshot
from features.features import compute_features

def run_pipeline():
    """
    Runs ingestion + feature generation automatically.
    """
    print("🚀 Starting OSRS Flipper AI pipeline...")

    # Step 1 — Run data ingestion
    print("⏳ Collecting snapshot...")
    snapshot()  # saves new snapshot file in DATA_DIR

    # Step 2 — Find latest snapshot
    files = sorted(
        [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet")],
        key=os.path.getmtime,
        reverse=True,
    )
    if not files:
        print("❌ No snapshot files found.")
        return

    latest = files[0]
    print(f"📦 Latest snapshot: {latest}")

    # Step 3 — Compute features
    try:
        df = pd.read_parquet(latest)
        features = compute_features(df)

        # Save features with timestamp
        ts = int(datetime.now(timezone.utc).timestamp())
        out_path = os.path.join(DATA_DIR, f"features_{ts}.parquet")
        features.to_parquet(out_path, index=False)

        print(f"✅ Features saved: {out_path} ({len(features)} rows)")

        # Optional: append to combined master file
        master_path = os.path.join(DATA_DIR, "features_master.parquet")
        if os.path.exists(master_path):
            old = pd.read_parquet(master_path)
            combined = pd.concat([old, features], ignore_index=True)
        else:
            combined = features
        combined.to_parquet(master_path, index=False)
        print(f"📈 Master feature table updated: {len(combined)} rows total")

    except Exception as e:
        print(f"❌ Error computing features: {e}")
        with open("logs/pipeline.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] ERROR: {e}\n")

    # Step 4 — Log successful run
    with open("logs/pipeline.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] SUCCESS: snapshot + features\n")


if __name__ == "__main__":
    run_pipeline()
