"""
bpc_v4 快速本地测试（已弃用独立实现，转调 train 入口）。

推荐:
  python -m quant_cursor.bpc_v4.train --dev --device cpu

本模块保留为兼容包装:
  python -m quant_cursor.bpc_v4.quick_test
"""

from __future__ import annotations

import sys


def main() -> int:
    from .train import main as train_main

    if len(sys.argv) <= 1:
        sys.argv.extend(["--dev", "--device", "cpu", "--num-workers", "0"])
    return train_main()


if __name__ == "__main__":
    raise SystemExit(main())
