import pandas as pd
vol = pd.read_parquet("/root/osrs_flipper_ai/osrs_flipper_ai/data/raw/snapshot_latest.parquet")
print(vol[["item_id", "daily_volume"]].head(10))
print(vol["daily_volume"].describe())
