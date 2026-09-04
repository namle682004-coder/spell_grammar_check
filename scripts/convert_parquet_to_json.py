# scripts/convert_parquet_to_json.py
import pandas as pd
import json
from pathlib import Path

# Đọc parquet
df = pd.read_parquet("data/raw/shynbui/data/train-00000-of-00001.parquet")

# Chia train/val/test (80-10-10)
total = len(df)
train_size = int(0.8 * total)
val_size = int(0.1 * total)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

# Lưu thành JSON
output_dir = Path("data/raw/shynbui/json")
output_dir.mkdir(parents=True, exist_ok=True)

for name, data_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    records = data_df.to_dict(orient="records")
    with open(output_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(records)} to {name}.json")
