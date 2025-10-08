from osrs_flipper_ai.data_ingest.backfill import backfill_all_items, merge_to_parquet
from osrs_flipper_ai.data_ingest.ingest import snapshot

if __name__ == "__main__":
    # 1. Fetch all  items and backfill /timeseries
    item_ids = backfill_all_items(timeframe="5m")

    # 2. Merge all JSONs into one Parquet dataset
    df = merge_to_parquet(timeframe="5m")

    print("\n✅ Backfill complete!")
    print(df.head())
