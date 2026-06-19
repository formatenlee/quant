import re
from pathlib import Path

import pandas as pd

u = pd.read_parquet("data/meta/universe.parquet")
idx_dir = Path("data/indices")
etf_dir = Path("data/etf")


def has_file(row) -> bool:
    base = idx_dir if row["asset_type"] == "index" else etf_dir
    p = base / f"{row['code']}.parquet"
    return p.exists() and p.stat().st_size > 0


u["has_file"] = u.apply(has_file, axis=1)
missing = u[~u["has_file"]]
done = u[u["has_file"]]

print("=== 文件状态 ===")
print(f"总计 {len(u)} | 已完成 {len(done)} | 未完成 {len(missing)}")
print(
    f"指数: {len(done[done.asset_type=='index'])}/{len(u[u.asset_type=='index'])}"
)
print(f"ETF: {len(done[done.asset_type=='etf'])}/{len(u[u.asset_type=='etf'])}")

# 模拟 skip-existing 过滤
pending = missing  # skip-existing 只保留 missing
print(f"\n=== skip-existing 应下载 ===")
print(f"待下载数量: {len(pending)} (仅 index 过滤: {len(pending[pending.asset_type=='index'])})")

# 检查日志
log_path = Path("data/meta/download_resume.log")
if log_path.exists():
    log = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"跳过已有数据 (\d+) 个", log)
    m2 = re.search(r"共 (\d+) 个标的", log)
    if m:
        print(f"\n=== 当前任务日志 ===")
        print(f"日志记录跳过: {m.group(1)} 个")
        print(f"日志记录待下载: {m2.group(1) if m2 else '?'} 个")

    # 非跳过的下载项
    new_codes = []
    for line in log.splitlines():
        if "已跳过" in line:
            continue
        hit = re.search(r"\[\d+/\d+\] (\S+) ", line)
        if hit:
            new_codes.append(hit.group(1))

    done_codes = set(done["code"])
    overlap = set(new_codes) & done_codes
    print(f"本次实际请求下载: {len(new_codes)} 个")
    print(f"与已有文件重叠 (应为0): {len(overlap)}")
    if overlap:
        print(f"重叠代码: {sorted(overlap)[:20]}")
