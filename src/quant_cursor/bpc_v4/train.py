"""
BPC-v4 训练入口（与 v3 相同，从项目根目录启动）。

示例:
  python -m quant_cursor.bpc_v4.train --dev --device cuda

小样本（200 标的，2019 至今）:
  python -m quant_cursor.bpc_v4.train \\
    --max-instruments 200 \\
    --start 2019-01-01 \\
    --max-samples 2000 \\
    --epochs 3 \\
    --batch-size 64 \\
    --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from quant_cursor.bpc.dataset import load_qlib_instruments
from quant_cursor.config import load_config

logger = logging.getLogger("quant_cursor.bpc_v4.train")


def _apply_dev_preset(args: argparse.Namespace) -> None:
    if not args.dev:
        return
    args.start = args.start or "2019-01-01"
    args.end = args.end or datetime.now().strftime("%Y-%m-%d")
    if args.max_instruments is None:
        args.max_instruments = 200
    if args.max_samples is None:
        args.max_samples = 2000
    if args.epochs == 50:
        args.epochs = 3
    if args.batch_size == 256:
        args.batch_size = 64
    logger.info(
        "Dev preset: start=%s end=%s max_instruments=%s max_samples=%s epochs=%s batch_size=%s",
        args.start,
        args.end,
        args.max_instruments,
        args.max_samples,
        args.epochs,
        args.batch_size,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BPC-v4 训练（qlib + Kronos）")
    p.add_argument("--config", type=str, default=None, help="quant_cursor config.yaml 路径")
    p.add_argument(
        "--dev",
        action="store_true",
        help="小样本测试：2019~今天 + 200标的 + max-samples=2000 + 3 epochs",
    )
    p.add_argument("--start", type=str, default=None, help="起始日期，默认 2019-01-01")
    p.add_argument("--end", type=str, default=None, help="结束日期，默认今天")
    p.add_argument("--max-instruments", type=int, default=None, help="从 qlib manifest 取前 N 只标的")
    p.add_argument("--max-samples", type=int, default=None, help="限制预计算窗口总数（小样本加速）")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-dir", type=str, default=None, help="checkpoint 目录")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--kronos-path", type=str, default=None, help="Kronos 本地模型目录")
    p.add_argument("--min-rows", type=int, default=60, help="manifest 过滤最少行数")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    _apply_dev_preset(args)

    qc_cfg = load_config(Path(args.config) if args.config else None)
    start = args.start or "2019-01-01"
    end = args.end or datetime.now().strftime("%Y-%m-%d")

    manifest = qc_cfg.meta_dir / "qlib_manifest.parquet"
    if args.max_instruments is not None:
        instruments = load_qlib_instruments(
            manifest,
            max_instruments=args.max_instruments,
            min_rows=args.min_rows,
        )
        logger.info(
            "Instrument source: qlib manifest (max_instruments=%s, loaded=%d)",
            args.max_instruments,
            len(instruments),
        )
    else:
        instruments = load_qlib_instruments(manifest, min_rows=args.min_rows)
        logger.info("Instrument source: qlib manifest (all filtered, n=%d)", len(instruments))

    if not instruments:
        raise RuntimeError(
            f"manifest 未返回任何标的，请检查 {manifest} 是否存在且 qlib 数据已导出。"
        )

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) if args.save_dir else qc_cfg.data_dir / "checkpoints" / "bpc_v4" / run_name

    from bpc_v4.config import GlobalConfig
    from bpc_v4.train import run_training

    config = GlobalConfig()
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.train.device = args.device
    config.train.save_dir = save_dir
    config.train.save_dir.mkdir(parents=True, exist_ok=True)
    config.qlib.provider_uri = qc_cfg.qlib_data_dir
    config.qlib.start_date = start
    config.qlib.end_date = end
    config.qlib.instruments = instruments
    config.qlib.max_samples = args.max_samples
    if args.kronos_path:
        config.kronos.local_path = args.kronos_path

    logger.info(
        "BPC-v4: %d instruments, %s ~ %s, qlib=%s, save_dir=%s",
        len(instruments),
        start,
        end,
        config.qlib.provider_uri,
        save_dir,
    )

    run_training(config, resume=args.resume)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
