import re
from pathlib import Path

import pandas as pd

u = pd.read_parquet("data/meta/universe.parquet")
missing_idx, missing_etf = [], []
for _, r in u.iterrows():
    base = Path("data/indices") if r["asset_type"] == "index" else Path("data/etf")
    if not (base / f"{r['code']}.parquet").exists():
        (missing_idx if r["asset_type"] == "index" else missing_etf).append(r["code"])

numeric = [c for c in missing_idx if re.fullmatch(r"\d{6}", c)]
alpha = [c for c in missing_idx if not re.fullmatch(r"\d{6}", c)]
print(f"done index: {len(u[u.asset_type=='index']) - len(missing_idx)}")
print(f"done etf: {len(u[u.asset_type=='etf']) - len(missing_etf)}")
print(f"missing index: {len(missing_idx)} (numeric {len(numeric)}, alpha {len(alpha)})")
print(f"missing etf: {len(missing_etf)}")
print("alpha sample:", alpha[:10])
