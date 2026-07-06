"""
Kronos 预计算入口（与 BPC 物化/训练解耦）。

示例（项目根目录，PYTHONPATH=src 或 pip install -e .）:

  # 全市场标的，指定日期窗口
  python -m quant_cursor.bpc_v4.precompute_kronos \\
    --full --start 2015-01-01 --end 2024-12-31 \\
    --device cuda \\
    --kronos-path /path/to/Kronos-Tokenizer-base \\
    --output-dir data/kronos_cache

  # 增量：已存在分片跳过；强制重建
  python -m quant_cursor.bpc_v4.precompute_kronos --full --force-rebuild

物化/训练时引用:
  python -m quant_cursor.bpc_v4.train ... --kronos-cache-dir data/kronos_cache
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from quant_cursor.bpc.dataset import load_qlib_instruments
from quant_cursor.config import load_config

from .config import GlobalConfig
from .dataset import BPCV4InstrumentStore
from .kronos import resolve_kronos_local_path, sync_kronos_config
from .kronos_cache import build_kronos_cache

logger = logging.getLogger("quant_cursor.bpc_v4.precompute_kronos")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BPC-v4 Kronos 预计算（z_q + s1_ids 分片缓存）")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--full", action="store_true", help="使用 manifest 全部标的")
    p.add_argument("--max-instruments", type=int, default=None)
    p.add_argument("--min-rows", type=int, default=60)
    p.add_argument("--instruments", nargs="*", default=None)
    p.add_argument("--output-dir", type=str, default=None, help="默认 data/kronos_cache")
    p.add_argument("--kronos-path", type=str, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cpu-threads", type=int, default=4, help="CPU 预计算线程数（默认 4）")
    p.add_argument("--device", type=str, default="cpu", help="已弃用：预计算固定使用 CPU")
    p.add_argument("--force-rebuild", action="store_true", help="覆盖已有标的分片")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    qc_cfg = load_config(Path(args.config) if args.config else None)

    start = args.start or "2015-01-01"
    end = args.end or datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(args.output_dir) if args.output_dir else qc_cfg.data_dir / "kronos_cache"

    if args.instruments:
        instruments = args.instruments
    else:
        manifest = qc_cfg.meta_dir / "qlib_manifest.parquet"
        instruments = load_qlib_instruments(
            manifest,
            max_instruments=None if args.full else args.max_instruments,
            min_rows=args.min_rows,
        )

    if not instruments:
        raise RuntimeError("无可用标的")

    config = GlobalConfig()
    config.qlib.provider_uri = qc_cfg.qlib_data_dir
    config.qlib.start_date = start
    config.qlib.end_date = end
    config.qlib.instruments = instruments
    if args.seq_len is not None:
        config.kronos.seq_len = args.seq_len

    if args.kronos_path:
        config.kronos.local_path = args.kronos_path
    else:
        resolved = resolve_kronos_local_path(None)
        if resolved:
            config.kronos.local_path = resolved
        else:
            logger.warning("未找到本地 Kronos，将尝试 HuggingFace")

    sync_kronos_config(config)

    logger.info(
        "Kronos precompute: instruments=%d window=%s..%s out=%s cpu_threads=%d",
        len(instruments),
        start,
        end,
        out_dir,
        args.cpu_threads,
    )

    store = BPCV4InstrumentStore(
        instruments=instruments,
        start=start,
        end=end,
        provider_uri=config.qlib.provider_uri,
        pad_missing_amount=config.kronos.amount_pad_zero,
    )

    meta = build_kronos_cache(
        store,
        config,
        out_dir,
        batch_size=args.batch_size,
        device="cpu",
        force=args.force_rebuild,
        instruments=list(store._cache.keys()),
        cpu_threads=args.cpu_threads,
    )
    logger.info(
        "Done: %d instruments, %d windows -> %s",
        meta.n_instruments,
        meta.n_windows,
        out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
