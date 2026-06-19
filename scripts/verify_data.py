"""核查 universe 中所有标的本地 parquet 完整性。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLS = {"date", "open", "high", "low", "close", "volume", "asset_type"}


def validate_parquet(path: Path, name: str = "") -> tuple[bool, str, str]:
    """返回 (是否有效, 级别 ok|warn|error, 说明)。"""
    if not path.exists():
        return False, "error", "文件不存在"
    size = path.stat().st_size
    if size < 100:
        return False, "error", f"文件过小 ({size} bytes)"

    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        return False, "error", f"无法读取: {exc}"

    if df.empty:
        return False, "error", "数据为空"
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        return False, "error", f"缺少列: {missing_cols}"
    if df["date"].isna().all():
        return False, "error", "日期全为空"
    if df["close"].isna().all():
        return False, "error", "收盘价全为空"

    rows = len(df)
    last = df["date"].max()
    if rows < 5:
        tag = "新上市" if name.startswith("N") or "新" in name else "历史较短"
        return True, "warn", f"{tag} rows={rows} last={last}"
    return True, "ok", f"rows={rows} last={last}"


def main() -> None:
    u = pd.read_parquet("data/meta/universe.parquet")
    bad: list[dict] = []
    warns: list[dict] = []

    for _, row in u.iterrows():
        base = Path("data/indices") if row["asset_type"] == "index" else Path("data/etf")
        path = base / f"{row['code']}.parquet"
        ok, level, msg = validate_parquet(path, str(row["name"]))
        item = {
            "code": row["code"],
            "name": row["name"],
            "asset_type": row["asset_type"],
            "path": str(path),
            "reason": msg,
        }
        if not ok:
            bad.append(item)
        elif level == "warn":
            warns.append(item)

    print(f"universe: {len(u)}")
    print(f"ok: {len(u) - len(bad) - len(warns)}")
    print(f"warn (新上市/历史短): {len(warns)}")
    print(f"error: {len(bad)}")
    if warns:
        print("\n--- 提示（数据有效但历史较短）---")
        for item in warns:
            print(f"  [{item['asset_type']}] {item['code']} {item['name']}: {item['reason']}")
    if bad:
        print("\n--- 异常列表 ---")
        for item in bad:
            print(f"  [{item['asset_type']}] {item['code']} {item['name']}: {item['reason']}")

        out = Path("data/meta/invalid_data.parquet")
        pd.DataFrame(bad).to_parquet(out, index=False)
        print(f"\n已保存: {out}")
    if warns:
        Path("data/meta").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(warns).to_parquet("data/meta/warn_data.parquet", index=False)


if __name__ == "__main__":
    main()
