from pathlib import Path

import pandas as pd

idx_dir = Path("data/indices")
u = pd.read_parquet("data/meta/universe.parquet")
indices = u[u["asset_type"] == "index"]

missing, corrupted = [], []
for _, row in indices.iterrows():
    p = idx_dir / f"{row['code']}.parquet"
    if not p.exists():
        missing.append((row["code"], row["name"]))
        continue
    if p.stat().st_size < 100:
        corrupted.append((row["code"], row["name"], p.stat().st_size))

print(f"missing: {len(missing)}")
for item in missing:
    print(" ", item)
print(f"corrupted (<100 bytes): {len(corrupted)}")
for item in corrupted:
    print(" ", item)
