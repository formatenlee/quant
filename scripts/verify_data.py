"""核查 universe 中所有标的本地 parquet 完整性。"""

from __future__ import annotations

from quant_cursor.config import load_config
from quant_cursor.validation import validate_universe


def main() -> None:
    config = load_config()
    report = validate_universe(config)
    print(f"universe: {len(report)}")
    print(f"ok: {(report['status'] == 'ok').sum()}")
    print(f"warn: {(report['status'] == 'warn').sum()}")
    print(f"error: {(report['status'] == 'error').sum()}")
    bad = report[report["status"] == "error"]
    if not bad.empty:
        print("\n--- 异常列表 ---")
        for _, item in bad.head(20).iterrows():
            print(f"  [{item['asset_type']}] {item['code']} {item['name']}: {item['message']}")


if __name__ == "__main__":
    main()
